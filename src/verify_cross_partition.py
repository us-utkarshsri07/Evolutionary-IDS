
from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000

PARTITIONS = {
    "train": 1,
    "validation": 2,
    "test": 4
}


def load_features():

    with open(
        DATA_DIR / "feature_schema.json",
        encoding="utf-8"
    ) as f:
        return json.load(f)["features"]


def prepare_features(chunk, features):

    X = chunk[features].apply(
        pd.to_numeric,
        errors="coerce"
    ).astype("float64")

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    values = X.to_numpy(copy=True)

    # Canonicalize missing values and signed zero.
    values[np.isnan(values)] = np.nan
    values[values == 0] = 0.0

    hashes = pd.util.hash_pandas_object(
        X,
        index=False
    ).to_numpy(dtype=np.uint64)

    return values, hashes


def main():

    features = load_features()

    print("\nEXACT CROSS-PARTITION VERIFICATION")
    print("=" * 65)

    # ------------------------------------------------
    # PASS 1: FIND HASHES PRESENT IN MULTIPLE PARTITIONS
    # ------------------------------------------------

    partition_hashes = {}

    for partition in PARTITIONS:

        print(f"\nFirst pass: {partition}")

        arrays = []

        file = DATA_DIR / f"{partition}.csv.gz"

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=features
        ):

            _, hashes = prepare_features(
                chunk,
                features
            )

            arrays.append(hashes)

        partition_hashes[partition] = np.unique(
            np.concatenate(arrays)
        )

    train_hashes = partition_hashes["train"]
    validation_hashes = partition_hashes["validation"]
    test_hashes = partition_hashes["test"]

    tv = np.intersect1d(
        train_hashes,
        validation_hashes
    )

    tt = np.intersect1d(
        train_hashes,
        test_hashes
    )

    vt = np.intersect1d(
        validation_hashes,
        test_hashes
    )

    candidate_hashes = np.unique(
        np.concatenate([tv, tt, vt])
    )

    print(
        f"\nCandidate shared hashes: "
        f"{len(candidate_hashes):,}"
    )

    # ------------------------------------------------
    # PASS 2: VERIFY ACTUAL NUMERICAL FEATURE VECTORS
    # ------------------------------------------------

    candidate_set = set(map(int, candidate_hashes))

    verified = {}

    for partition, partition_bit in PARTITIONS.items():

        print(f"\nExact verification: {partition}")

        file = DATA_DIR / f"{partition}.csv.gz"

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=features + ["target", "Label"]
        ):

            values, hashes = prepare_features(
                chunk,
                features
            )

            mask = np.isin(
                hashes,
                candidate_hashes,
                assume_unique=False
            )

            selected_indices = np.flatnonzero(mask)

            for i in selected_indices:

                # Full numerical row comparison.
                payload = values[i].tobytes()

                hash_value = int(hashes[i])

                label = str(chunk["Label"].iloc[i]).strip()

                target = int(chunk["target"].iloc[i])

                record = verified.setdefault(
                    payload,
                    {
                        "hash": hash_value,
                        "partition_mask": 0,
                        "label_mask": 0,
                        "rows": {
                            "train": 0,
                            "validation": 0,
                            "test": 0
                        },
                        "labels": set()
                    }
                )

                record["partition_mask"] |= partition_bit

                record["label_mask"] |= (
                    1 if target == 0 else 2
                )

                record["rows"][partition] += 1

                record["labels"].add(label)

        print(f"Completed: {partition}")

    # ------------------------------------------------
    # ANALYZE EXACT MATCHES
    # ------------------------------------------------

    def shared(mask):
        return mask.bit_count() >= 2

    exact_shared = [
        item for item in verified.values()
        if shared(item["partition_mask"])
    ]

    exact_tv = sum(
        (item["partition_mask"] & 3) == 3
        for item in exact_shared
    )

    exact_tt = sum(
        (item["partition_mask"] & 5) == 5
        for item in exact_shared
    )

    exact_vt = sum(
        (item["partition_mask"] & 6) == 6
        for item in exact_shared
    )

    exact_all = sum(
        item["partition_mask"] == 7
        for item in exact_shared
    )

    conflicts = [
        item for item in exact_shared
        if item["label_mask"] == 3
    ]

    verified_hashes = {
        item["hash"] for item in exact_shared
    }

    unverified_hashes = (
        candidate_set - verified_hashes
    )

    # ------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------

    summary = pd.DataFrame([

        {
            "Metric": "Candidate shared hashes",
            "Value": len(candidate_hashes)
        },

        {
            "Metric": "Exactly verified shared feature vectors",
            "Value": len(exact_shared)
        },

        {
            "Metric": "Exact train-validation matches",
            "Value": exact_tv
        },

        {
            "Metric": "Exact train-test matches",
            "Value": exact_tt
        },

        {
            "Metric": "Exact validation-test matches",
            "Value": exact_vt
        },

        {
            "Metric": "Exact matches across all three",
            "Value": exact_all
        },

        {
            "Metric": "Exact vectors with conflicting labels",
            "Value": len(conflicts)
        },

        {
            "Metric": "Shared hashes not exactly verified",
            "Value": len(unverified_hashes)
        }

    ])

    summary.to_csv(
        REPORT_DIR / "exact_cross_partition_audit.csv",
        index=False
    )

    conflict_rows = []

    for item in conflicts:

        conflict_rows.append({

            "Hash": item["hash"],

            "Partitions": ",".join([
                p for p, bit in PARTITIONS.items()
                if item["partition_mask"] & bit
            ]),

            "Labels": " | ".join(
                sorted(item["labels"])
            ),

            "Train_Rows": item["rows"]["train"],

            "Validation_Rows": item["rows"]["validation"],

            "Test_Rows": item["rows"]["test"]

        })

    pd.DataFrame(conflict_rows).to_csv(
        REPORT_DIR / "verified_label_conflicts.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("EXACT VERIFICATION RESULTS")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nReports saved:")
    print("exact_cross_partition_audit.csv")
    print("verified_label_conflicts.csv")


if __name__ == "__main__":
    main()
