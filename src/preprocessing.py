
from pathlib import Path
from collections import Counter
import gzip
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000

# Preserve -1 as a TCP window sentinel.
SENTINEL_FEATURES = {
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"
}


# --------------------------------------------------
# DATASET SPLITTING
# --------------------------------------------------

def get_partition(filename):

    name = filename.lower()

    if any(day in name for day in [
        "monday", "tuesday", "wednesday"
    ]):
        return "train"

    if "thursday" in name:
        return "validation"

    if "friday" in name:
        return "test"

    raise ValueError(
        f"Unknown dataset file: {filename}"
    )


# --------------------------------------------------
# LOAD AUDIT INFORMATION
# --------------------------------------------------

def load_constant_features():

    audit_file = REPORT_DIR / "data_quality_features.csv"

    if not audit_file.exists():
        raise FileNotFoundError(
            "Run data_quality_audit.py first."
        )

    audit = pd.read_csv(audit_file)

    constant = audit[
        audit["Constant_Feature"].astype(str).str.lower() == "true"
    ]["Feature"].tolist()

    return constant


def duplicate_column_verified():

    audit_file = REPORT_DIR / "duplicate_column_audit.csv"

    if not audit_file.exists():
        return False

    audit = pd.read_csv(audit_file)

    if audit.empty:
        return False

    row = audit.iloc[0]

    return (
        int(row["Total_Comparisons"]) > 0
        and int(row["Mismatched_Rows"]) == 0
    )


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def preprocess_dataset():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))

    if len(files) != 8:
        raise ValueError(
            f"Expected 8 CSV files, found {len(files)}"
        )

    # Remove previous generated partitions.
    for partition in ["train", "validation", "test"]:
        output = PROCESSED_DIR / f"{partition}.csv.gz"

        if output.exists():
            output.unlink()

    constant_features = load_constant_features()

    # Verify whether the duplicate header feature
    # can safely be eliminated.
    drop_duplicate_header = duplicate_column_verified()

    files_by_partition = {
        "train": [],
        "validation": [],
        "test": []
    }

    # Validate schemas.
    reference_columns = None

    for file in files:

        columns = pd.read_csv(
            file,
            nrows=0,
            encoding="utf-8-sig",
            skipinitialspace=True
        ).columns

        columns = columns.astype(str).str.strip().tolist()

        if "Label" not in columns:
            raise ValueError(f"Label missing in {file.name}")

        if reference_columns is None:
            reference_columns = columns

        elif set(columns) != set(reference_columns):
            raise ValueError(
                f"Schema mismatch in {file.name}"
            )

        partition = get_partition(file.name)

        files_by_partition[partition].append(file)

    if any(not group for group in files_by_partition.values()):
        raise ValueError("One or more partitions are empty.")

    # Final feature selection before model-level selection.
    feature_columns = [
        col for col in reference_columns
        if col != "Label"
        and col not in constant_features
    ]

    if drop_duplicate_header:
        feature_columns = [
            col for col in feature_columns
            if col != "Fwd Header Length.1"
        ]

    missing_features = (
        set(SENTINEL_FEATURES) - set(feature_columns)
    )

    if missing_features:
        raise ValueError(
            f"Expected sentinel features missing: {missing_features}"
        )

    print("\n" + "=" * 65)
    print("INTRUSION DETECTION PREPROCESSING")
    print("=" * 65)

    print(f"Original feature count: {len(reference_columns) - 1}")
    print(f"Constant features removed: {len(constant_features)}")
    print(
        f"Duplicate header removed: "
        f"{drop_duplicate_header}"
    )
    print(f"Remaining features: {len(feature_columns)}")

    # Partition statistics.
    row_counts = Counter()
    label_counts = {
        name: Counter()
        for name in files_by_partition
    }

    invalid_negative_counts = Counter()

    for partition, partition_files in files_by_partition.items():

        output_file = PROCESSED_DIR / f"{partition}.csv.gz"

        print(f"\nProcessing {partition.upper()}")

        first_write = True

        for file in partition_files:

            print(f"  Reading {file.name}")

            for chunk in pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                encoding="utf-8-sig",
                skipinitialspace=True
            ):

                chunk.columns = (
                    chunk.columns.astype(str).str.strip()
                )

                # Normalize target labels.
                labels = (
                    chunk["Label"]
                    .astype("string")
                    .str.strip()
                )

                valid_labels = (
                    labels.notna()
                    & labels.ne("")
                )

                # Exclude records with no target label.
                labels = labels[valid_labels]

                # Convert candidate input features to numeric.
                X = chunk.loc[
                    valid_labels,
                    feature_columns
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # Replace positive and negative infinity.
                X = X.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                # Replace invalid negative measurements.
                for col in feature_columns:

                    if col in SENTINEL_FEATURES:
                        continue

                    negative_mask = X[col] < 0

                    invalid_negative_counts[col] += int(
                        negative_mask.sum()
                    )

                    X[col] = X[col].mask(
                        negative_mask,
                        np.nan
                    )

                # Create binary classification target.
                y = (
                    labels.str.upper() != "BENIGN"
                ).astype("int8")

                # Preserve original attack labels for analysis.
                X["Label"] = labels.to_numpy()
                X["target"] = y.to_numpy()

                X.to_csv(
                    output_file,
                    index=False,
                    mode="wt" if first_write else "at",
                    header=first_write,
                    compression="gzip"
                )

                first_write = False

                row_counts[partition] += len(X)

                label_counts[partition].update(
                    labels.tolist()
                )

        print(
            f"  Rows written: {row_counts[partition]:,}"
        )

    # --------------------------------------------------
    # SAVE FEATURE SCHEMA
    # --------------------------------------------------

    schema = {
        "project": "Intrusion Detection Using Evolutionary Feature Selection",
        "original_feature_count": len(reference_columns) - 1,
        "final_feature_count": len(feature_columns),
        "features": feature_columns,
        "removed_constant_features": constant_features,
        "removed_duplicate_feature": (
            "Fwd Header Length.1"
            if drop_duplicate_header
            else None
        ),
        "target_mapping": {
            "BENIGN": 0,
            "ALL_ATTACKS": 1
        },
        "sentinel_features": list(SENTINEL_FEATURES),
        "split_strategy": {
            "train": "Monday, Tuesday, Wednesday",
            "validation": "Thursday",
            "test": "Friday"
        }
    }

    with open(
        PROCESSED_DIR / "feature_schema.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(schema, f, indent=4)

    # --------------------------------------------------
    # SAVE DATA SUMMARY
    # --------------------------------------------------

    summary = []

    for partition in ["train", "validation", "test"]:

        summary.append({
            "Partition": partition,
            "Total_Rows": row_counts[partition],
            "Benign": label_counts[partition].get("BENIGN", 0),
            "Attack": (
                row_counts[partition]
                - label_counts[partition].get("BENIGN", 0)
            )
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        REPORT_DIR / "preprocessing_summary.csv",
        index=False
    )

    negative_report = pd.DataFrame([
        {
            "Feature": feature,
            "Invalid_Negative_Values": count
        }
        for feature, count in invalid_negative_counts.items()
    ])

    negative_report.to_csv(
        REPORT_DIR / "preprocessing_negative_values.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETED")
    print("=" * 65)

    print(summary_df.to_string(index=False))

    print("\nFinal features:", len(feature_columns))
    print("Output directory:", PROCESSED_DIR)


if __name__ == "__main__":
    preprocess_dataset()
