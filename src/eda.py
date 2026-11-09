
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

PARTITIONS = ["train", "validation", "test"]

CHUNK_SIZE = 100_000
SAMPLE_PER_CHUNK = 3000

sns.set_theme(style="whitegrid")


def run_eda():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    label_counts = {
        p: Counter() for p in PARTITIONS
    }

    target_counts = {
        p: Counter() for p in PARTITIONS
    }

    missing_counts = Counter()

    train_samples = []
    total_train_rows = 0

    print("\n" + "=" * 65)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    # --------------------------------------------------
    # 1. LOAD PARTITIONS IN CHUNKS
    # --------------------------------------------------

    for partition in PARTITIONS:

        file = DATA_DIR / f"{partition}.csv.gz"

        if not file.exists():
            raise FileNotFoundError(file)

        print(f"\nAnalyzing {partition.upper()}")

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE
        ):

            labels = (
                chunk["Label"]
                .astype("string")
                .str.strip()
            )

            label_counts[partition].update(
                labels.value_counts().to_dict()
            )

            target_counts[partition].update(
                chunk["target"].value_counts().to_dict()
            )

            # Only training data is used for
            # feature statistics and correlation.
            if partition == "train":

                features = chunk.drop(
                    columns=["Label", "target"]
                )

                missing_counts.update(
                    features.isna().sum().to_dict()
                )

                total_train_rows += len(chunk)

                n = min(
                    SAMPLE_PER_CHUNK,
                    len(features)
                )

                sample = features.sample(
                    n=n,
                    random_state=42
                )

                train_samples.append(sample)

        print(f"Completed {partition}")

    # --------------------------------------------------
    # 2. ATTACK DISTRIBUTION REPORT
    # --------------------------------------------------

    distribution = []

    for partition in PARTITIONS:

        for label, count in label_counts[partition].items():

            distribution.append({
                "Partition": partition,
                "Label": label,
                "Count": count
            })

    distribution_df = pd.DataFrame(distribution)

    distribution_df.to_csv(
        REPORT_DIR / "eda_label_distribution.csv",
        index=False
    )

    # Binary partition distribution
    binary_data = []

    for partition in PARTITIONS:

        benign = target_counts[partition].get(0, 0)
        attack = target_counts[partition].get(1, 0)

        binary_data.append({
            "Partition": partition,
            "Benign": benign,
            "Attack": attack
        })

    binary_df = pd.DataFrame(binary_data)

    binary_df.to_csv(
        REPORT_DIR / "eda_binary_distribution.csv",
        index=False
    )

    # --------------------------------------------------
    # 3. PLOT CLASS DISTRIBUTION
    # --------------------------------------------------

    plot_df = binary_df.set_index("Partition")[
        ["Benign", "Attack"]
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(9, 5),
        stacked=True
    )

    ax.set_title("Benign vs Attack Traffic")
    ax.set_ylabel("Number of network flows")
    ax.set_xlabel("Dataset partition")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "binary_class_distribution.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------
    # 4. ATTACK FAMILY DISTRIBUTION
    # --------------------------------------------------

    overall_labels = Counter()

    for partition in PARTITIONS:
        overall_labels.update(
            label_counts[partition]
        )

    attack_counts = {
        label: count
        for label, count in overall_labels.items()
        if label.strip().upper() != "BENIGN"
    }

    attack_df = pd.DataFrame(
        list(attack_counts.items()),
        columns=["Attack", "Count"]
    ).sort_values(
        "Count",
        ascending=False
    )

    attack_df.to_csv(
        REPORT_DIR / "eda_attack_distribution.csv",
        index=False
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=attack_df,
        x="Count",
        y="Attack"
    )

    plt.xscale("log")
    plt.title("Attack Family Distribution (Log Scale)")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "attack_family_distribution.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------
    # 5. MISSING VALUE ANALYSIS
    # --------------------------------------------------

    missing_df = pd.DataFrame([
        {
            "Feature": feature,
            "Missing_Count": count,
            "Missing_Percentage": (
                count / total_train_rows * 100
            )
        }
        for feature, count in missing_counts.items()
    ])

    missing_df = missing_df.sort_values(
        "Missing_Count",
        ascending=False
    )

    missing_df.to_csv(
        REPORT_DIR / "eda_missing_values.csv",
        index=False
    )

    top_missing = missing_df[
        missing_df["Missing_Count"] > 0
    ].head(20)

    if not top_missing.empty:

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=top_missing,
            x="Missing_Count",
            y="Feature"
        )

        plt.title("Missing Values in Training Features")
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "missing_values.png",
            dpi=200
        )

        plt.close()

    # --------------------------------------------------
    # 6. FEATURE STATISTICS
    # --------------------------------------------------

    print("\nCombining training samples...")

    sample_df = pd.concat(
        train_samples,
        ignore_index=True
    )

    sample_df = sample_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    statistics = sample_df.describe().T

    statistics.to_csv(
        REPORT_DIR / "eda_feature_statistics.csv"
    )

    # --------------------------------------------------
    # 7. CORRELATION ANALYSIS
    # --------------------------------------------------

    variances = sample_df.var(
        numeric_only=True
    ).dropna()

    top_features = variances.nlargest(20).index.tolist()

    correlation = sample_df[
        top_features
    ].corr()

    correlation.to_csv(
        REPORT_DIR / "eda_correlation_matrix.csv"
    )

    plt.figure(figsize=(14, 11))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.2
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_heatmap.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------
    # 8. FEATURE DISTRIBUTION
    # --------------------------------------------------

    selected_features = [
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Packet Length Mean"
    ]

    for feature in selected_features:

        if feature not in sample_df.columns:
            continue

        values = sample_df[feature].dropna()

        if len(values) == 0:
            continue

        plt.figure(figsize=(9, 5))

        sns.histplot(
            values.clip(
                upper=values.quantile(0.99)
            ),
            bins=50,
            kde=False
        )

        plt.title(
            f"Distribution of {feature} "
            "(99th percentile capped for visualization)"
        )

        plt.xlabel(feature)
        plt.tight_layout()

        safe_name = (
            feature.lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        plt.savefig(
            FIGURE_DIR / f"{safe_name}_distribution.png",
            dpi=200
        )

        plt.close()

    print("\n" + "=" * 65)
    print("EDA COMPLETED")
    print("=" * 65)

    print(f"Training rows: {total_train_rows:,}")
    print(f"EDA sample size: {len(sample_df):,}")
    print(f"Features analyzed: {len(sample_df.columns)}")

    print("\nBinary distribution:")
    print(binary_df.to_string(index=False))

    print("\nReports:", REPORT_DIR)
    print("Figures:", FIGURE_DIR)


if __name__ == "__main__":
    run_eda()
