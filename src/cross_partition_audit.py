
from pathlib import Path

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


def main():

    schema = pd.read_json(
        DATA_DIR / "feature_schema.json",
        typ="series"
    )

    features = schema["features"]

    all_hashes = []
    all_partitions = []
    all_labels = []

    total_rows = 0

    print("\nCROSS-PARTITION DUPLICATE AUDIT")
    print("=" * 65)

    for partition, partition_bit in PARTITIONS.items():

        file = DATA_DIR / f"{partition}.csv.gz"

        print(f"\nProcessing {partition}...")

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=features + ["target"]
        ):

            # Normalize numerical dtypes before hashing.
            X = chunk[features].astype("float64")

            hashes = pd.util.hash_pandas_object(
                X,
                index=False
            ).to_numpy(dtype=np.uint64)

            labels = chunk["target"].to_numpy(
                dtype=np.uint8
            )

            all_hashes.append(hashes)

            all_partitions.append(
                np.full(
                    len(chunk),
                    partition_bit,
                    dtype=np.uint8
                )
            )

            all_labels.append(labels)

            total_rows += len(chunk)

        print(f"Completed: {partition}")

    print("\nAggregating feature-vector hashes...")

    hashes = np.concatenate(all_hashes)

    partition_codes = np.concatenate(all_partitions)

    labels = np.concatenate(all_labels)

    unique_hashes, inverse, counts = np.unique(
        hashes,
        return_inverse=True,
        return_counts=True
    )

    partition_masks = np.zeros(
        len(unique_hashes),
        dtype=np.uint8
    )

    label_masks = np.zeros(
        len(unique_hashes),
        dtype=np.uint8
    )

    # Partition bits: train=1, validation=2, test=4
    np.bitwise_or.at(
        partition_masks,
        inverse,
        partition_codes
    )

    # Label bits: benign=1, attack=2
    label_bits = np.where(
        labels == 0,
        1,
        2
    ).astype(np.uint8)

    np.bitwise_or.at(
        label_masks,
        inverse,
        label_bits
    )

    # Cross-partition intersections
    train_validation = (
        (partition_masks & 3) == 3
    )

    train_test = (
        (partition_masks & 5) == 5
    )

    validation_test = (
        (partition_masks & 6) == 6
    )

    all_three = partition_masks == 7

    cross_partition = (
        train_validation |
        train_test |
        validation_test
    )

    # Same feature vector assigned both benign and attack.
    conflicting_labels = label_masks == 3

    cross_partition_conflicts = (
        cross_partition & conflicting_labels
    )

    repeated_vectors = counts > 1

    summary = [

        ("Total records", total_rows),

        ("Unique feature-vector hashes", len(unique_hashes)),

        (
            "Rows repeated beyond first occurrence",
            int(total_rows - len(unique_hashes))
        ),

        (
            "Feature vectors repeated within or across partitions",
            int(repeated_vectors.sum())
        ),

        (
            "Train-validation shared vectors",
            int(train_validation.sum())
        ),

        (
            "Train-test shared vectors",
            int(train_test.sum())
        ),

        (
            "Validation-test shared vectors",
            int(validation_test.sum())
        ),

        (
            "Vectors shared across all three partitions",
            int(all_three.sum())
        ),

        (
            "Vectors with conflicting labels",
            int(conflicting_labels.sum())
        ),

        (
            "Cross-partition vectors with conflicting labels",
            int(cross_partition_conflicts.sum())
        )

    ]

    report = pd.DataFrame(
        summary,
        columns=["Metric", "Value"]
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = REPORT_DIR / "cross_partition_audit.csv"

    report.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 65)
    print("AUDIT RESULTS")
    print("=" * 65)

    print(report.to_string(index=False))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
