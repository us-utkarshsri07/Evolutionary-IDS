
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000

# Global exact-duplicate detection uses additional RAM.
CHECK_EXACT_DUPLICATES = True


def audit_dataset():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found.")

    missing_counts = Counter()
    invalid_counts = Counter()
    infinite_counts = Counter()

    min_values = {}
    max_values = {}

    label_counts = Counter()

    total_rows = 0
    missing_labels = 0
    duplicate_rows = 0

    seen_hashes = set() if CHECK_EXACT_DUPLICATES else None

    reference_columns = None

    print("\n" + "=" * 70)
    print("CIC-IDS2017 DATA QUALITY AUDIT")
    print("=" * 70)

    for file in files:

        print(f"\nAuditing: {file.name}")

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
            skipinitialspace=True
        ):

            chunk.columns = chunk.columns.astype(str).str.strip()

            if "Label" not in chunk.columns:
                raise ValueError(f"Missing Label column: {file.name}")

            if reference_columns is None:
                reference_columns = list(chunk.columns)

            elif set(chunk.columns) != set(reference_columns):
                raise ValueError(f"Column mismatch: {file.name}")

            rows = len(chunk)
            total_rows += rows

            # 1. Label integrity
            labels = chunk["Label"].astype("string").str.strip()

            label_missing_mask = labels.isna() | labels.eq("")
            missing_labels += int(label_missing_mask.sum())

            label_counts.update(
                labels[~label_missing_mask].value_counts().to_dict()
            )

            # 2. Numeric feature audit
            features = chunk.drop(columns=["Label"])

            for column in features.columns:

                raw = features[column]

                values = pd.to_numeric(raw, errors="coerce")

                missing_counts[column] += int(values.isna().sum())

                invalid_mask = raw.notna() & values.isna()
                invalid_counts[column] += int(invalid_mask.sum())

                array = values.to_numpy(
                    dtype=np.float64,
                    na_value=np.nan
                )

                inf_mask = np.isinf(array)
                infinite_counts[column] += int(inf_mask.sum())

                finite = array[np.isfinite(array)]

                if len(finite):

                    current_min = float(finite.min())
                    current_max = float(finite.max())

                    min_values[column] = min(
                        min_values.get(column, current_min),
                        current_min
                    )

                    max_values[column] = max(
                        max_values.get(column, current_max),
                        current_max
                    )

            # 3. Exact duplicate rows
            if CHECK_EXACT_DUPLICATES:

                hashes = pd.util.hash_pandas_object(
                    chunk,
                    index=False
                ).to_numpy()

                unique_hashes = set(map(int, hashes))

                duplicates_in_chunk = len(hashes) - len(unique_hashes)

                duplicates_from_previous = sum(
                    h in seen_hashes for h in unique_hashes
                )

                duplicate_rows += (
                    duplicates_in_chunk + duplicates_from_previous
                )

                seen_hashes.update(unique_hashes)

        print(f"Completed: {file.name}")

    # 4. Generate feature quality report
    feature_report = []

    for column in reference_columns:

        if column == "Label":
            continue

        minimum = min_values.get(column, np.nan)
        maximum = max_values.get(column, np.nan)

        all_missing = pd.isna(minimum)

        feature_report.append({

            "Feature": column,

            "Missing_Or_Invalid_Total":
                missing_counts[column],

            "Invalid_Numeric":
                invalid_counts[column],

            "Infinite_Values":
                infinite_counts[column],

            "Minimum": minimum,

            "Maximum": maximum,

            "All_Missing": all_missing,

            "Constant_Feature":
                bool(
                    not all_missing and minimum == maximum
                )

        })

    feature_df = pd.DataFrame(feature_report)

    feature_df.to_csv(
        REPORT_DIR / "data_quality_features.csv",
        index=False
    )

    # 5. Save exact class counts
    label_df = pd.DataFrame([
        {"Label": label, "Count": count}
        for label, count in label_counts.items()
    ])

    label_df.to_csv(
        REPORT_DIR / "audited_labels.csv",
        index=False
    )

    # 6. Save duplicate summary
    duplicate_report = pd.DataFrame([{

        "Total_Rows": total_rows,

        "Missing_Labels": missing_labels,

        "Exact_Duplicate_Rows":
            duplicate_rows if CHECK_EXACT_DUPLICATES else "Not checked",

        "Candidate_Features":
            len(reference_columns) - 1

    }])

    duplicate_report.to_csv(
        REPORT_DIR / "data_quality_summary.csv",
        index=False
    )

    # 7. Print results
    print("\n" + "=" * 70)
    print("DATA QUALITY AUDIT RESULTS")
    print("=" * 70)

    print(f"Total rows: {total_rows:,}")
    print(f"Candidate features: {len(reference_columns) - 1}")
    print(f"Missing labels: {missing_labels:,}")

    if CHECK_EXACT_DUPLICATES:
        print(f"Exact duplicate rows: {duplicate_rows:,}")

    print("\nLabel distribution:")

    for label, count in label_counts.most_common():
        print(f"{label!r}: {count:,}")

    print("\nFeature quality summary:")

    print(
        feature_df[
            [
                "Feature",
                "Missing_Or_Invalid_Total",
                "Infinite_Values",
                "Constant_Feature"
            ]
        ].to_string(index=False)
    )

    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    audit_dataset()
