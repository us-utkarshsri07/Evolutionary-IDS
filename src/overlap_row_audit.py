
from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000


def get_hashes(chunk, features):

    X = chunk[features].apply(
        pd.to_numeric,
        errors="coerce"
    ).astype("float64")

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return pd.util.hash_pandas_object(
        X,
        index=False
    ).to_numpy(dtype=np.uint64)


def main():

    with open(
        DATA_DIR / "feature_schema.json",
        encoding="utf-8"
    ) as f:

        features = json.load(f)["features"]

    # -----------------------------------------
    # Build reference hashes from full training
    # partition, not only the 250k model sample.
    # -----------------------------------------

    print("\nCollecting training feature vectors...")

    train_hashes = []

    train_labels = []

    for chunk in pd.read_csv(
        DATA_DIR / "train.csv.gz",
        chunksize=CHUNK_SIZE,
        usecols=features + ["target"]
    ):

        hashes = get_hashes(chunk, features)

        labels = chunk["target"].to_numpy(
            dtype=np.uint8
        )

        train_hashes.append(hashes)

        train_labels.append(labels)

    hashes = np.concatenate(train_hashes)

    labels = np.concatenate(train_labels)

    unique_hashes, inverse = np.unique(
        hashes,
        return_inverse=True
    )

    label_masks = np.zeros(
        len(unique_hashes),
        dtype=np.uint8
    )

    # 1 = Benign, 2 = Attack
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

    print(
        f"Unique training patterns: {len(unique_hashes):,}"
    )

    # -----------------------------------------
    # Evaluate overlap in validation and test.
    # -----------------------------------------

    results = []

    for partition in ["validation", "test"]:

        print(f"\nAnalyzing {partition}...")

        total = 0
        overlapping = 0
        novel = 0
        conflicting = 0

        for chunk in pd.read_csv(
            DATA_DIR / f"{partition}.csv.gz",
            chunksize=CHUNK_SIZE,
            usecols=features + ["target"]
        ):

            current_hashes = get_hashes(
                chunk,
                features
            )

            targets = chunk["target"].to_numpy(
                dtype=np.uint8
            )

            positions = np.searchsorted(
                unique_hashes,
                current_hashes
            )

            valid = positions < len(unique_hashes)

            safe_positions = np.minimum(
                positions,
                len(unique_hashes) - 1
            )

            matched = (
                valid &
                (
                    unique_hashes[safe_positions]
                    == current_hashes
                )
            )

            total += len(chunk)

            overlapping += int(matched.sum())

            novel += int((~matched).sum())

            # Check whether a matching training pattern
            # has the opposite binary target.
            matched_positions = safe_positions[matched]

            matching_labels = label_masks[
                matched_positions
            ]

            current_targets = targets[matched]

            conflict = (
                ((current_targets == 0) & ((matching_labels & 2) != 0))
                |
                ((current_targets == 1) & ((matching_labels & 1) != 0))
            )

            conflicting += int(conflict.sum())

        results.append({

            "Partition": partition,

            "Total_Rows": total,

            "Overlapping_Rows": overlapping,

            "Novel_Feature_Rows": novel,

            "Overlap_Percentage": (
                overlapping / total * 100
            ),

            "Conflicting_Label_Rows": conflicting,

            "Novel_Percentage": (
                novel / total * 100
            )

        })

    # -----------------------------------------
    # Save results
    # -----------------------------------------

    report = pd.DataFrame(results)

    output = REPORT_DIR / "overlap_row_audit.csv"

    report.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 70)
    print("ROW-LEVEL OVERLAP RESULTS")
    print("=" * 70)

    print(
        report.to_string(index=False)
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
