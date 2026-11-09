
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000

PARTITIONS = ["train", "validation", "test"]


# ============================================================
# FEATURE HASHING
# ============================================================

def prepare_features(df, features):

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    ).astype("float64")

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X


def calculate_hashes(X):

    return pd.util.hash_pandas_object(
        X,
        index=False
    ).to_numpy(dtype=np.uint64)


# ============================================================
# BUILD REFERENCE HASH INDEXES
# ============================================================

def build_reference_indexes(
    baseline_features,
    ga_features
):

    indexes = {}

    for partition in PARTITIONS:

        print(f"\nBuilding reference index: {partition}")

        base_parts = []
        ga_parts = []

        file = DATA_DIR / f"{partition}.csv.gz"

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            usecols=baseline_features
        ):

            # Guarantee identical feature ordering.
            X_base = prepare_features(
                chunk,
                baseline_features
            )

            X_ga = X_base[ga_features]

            base_parts.append(
                calculate_hashes(X_base)
            )

            ga_parts.append(
                calculate_hashes(X_ga)
            )

        indexes[partition] = {

            "baseline": np.unique(
                np.concatenate(base_parts)
            ),

            "ga": np.unique(
                np.concatenate(ga_parts)
            )

        }

        print(
            f"Baseline unique patterns: "
            f"{len(indexes[partition]['baseline']):,}"
        )

        print(
            f"GA unique patterns: "
            f"{len(indexes[partition]['ga']):,}"
        )

    return indexes


# ============================================================
# BUILD EVALUATION REFERENCE SETS
# ============================================================

def prepare_reference_sets(indexes):

    # Validation novelty:
    # Must be unseen in training.

    validation_reference = {
        "baseline": indexes["train"]["baseline"],
        "ga": indexes["train"]["ga"]
    }

    # Test novelty:
    # Must be unseen in training AND validation.

    test_reference = {

        "baseline": np.union1d(
            indexes["train"]["baseline"],
            indexes["validation"]["baseline"]
        ),

        "ga": np.union1d(
            indexes["train"]["ga"],
            indexes["validation"]["ga"]
        )

    }

    return {
        "validation": validation_reference,
        "test": test_reference
    }


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    model_name,
    partition,
    subset,
    feature_count,
    elapsed_time
):

    if len(y_true) == 0:
        return None

    y_pred = (
        probabilities >= 0.50
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    return {

        "Model": model_name,

        "Partition": partition,

        "Subset": subset,

        "Records": len(y_true),

        "Features": feature_count,

        "Accuracy": accuracy_score(
            y_true, y_pred
        ),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "Attack_Precision": precision_score(
            y_true, y_pred,
            zero_division=0
        ),

        "Attack_Recall": recall_score(
            y_true, y_pred,
            zero_division=0
        ),

        "Attack_F1": f1_score(
            y_true, y_pred,
            zero_division=0
        ),

        "Macro_F1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),

        "False_Positive_Rate": fpr,

        "ROC_AUC": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),

        "PR_AUC": (
            average_precision_score(
                y_true, probabilities
            )
            if np.sum(y_true) > 0
            else np.nan
        ),

        "MCC": matthews_corrcoef(
            y_true, y_pred
        ),

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp),

        "Prediction_Time_Seconds": elapsed_time

    }


# ============================================================
# EVALUATE A PARTITION
# ============================================================

def evaluate_partition(
    partition,
    reference,
    baseline_package,
    ga_package
):

    baseline_model = baseline_package["pipeline"]
    ga_model = ga_package["pipeline"]

    base_features = baseline_package["features"]
    ga_features = ga_package["features"]

    thresholds = {
        baseline_package.get("threshold", 0.50),
        ga_package.get("threshold", 0.50)
    }

    if thresholds != {0.50}:
        raise ValueError(
            "Both models must use the same 0.50 threshold."
        )

    baseline_true = []
    baseline_prob = []

    ga_true = []
    ga_prob = []

    counts = {
        "Total_Rows": 0,
        "Baseline_Seen_Rows": 0,
        "GA_Seen_Rows": 0,
        "Common_Novel_Rows": 0,
        "Baseline_Only_Novel_Rows": 0,
        "GA_Only_Novel_Rows": 0
    }

    baseline_prediction_time = 0.0
    ga_prediction_time = 0.0

    data_file = DATA_DIR / f"{partition}.csv.gz"

    print(f"\nEvaluating partition: {partition}")

    for chunk in pd.read_csv(
        data_file,
        chunksize=CHUNK_SIZE,
        usecols=base_features + ["target"]
    ):

        X_base = prepare_features(
            chunk,
            base_features
        )

        X_ga = X_base[ga_features]

        y = chunk["target"].to_numpy(
            dtype=np.int8
        )

        # Construct novelty masks for both models.
        base_hashes = calculate_hashes(X_base)
        ga_hashes = calculate_hashes(X_ga)

        base_seen = np.isin(
            base_hashes,
            reference["baseline"],
            assume_unique=False
        )

        ga_seen = np.isin(
            ga_hashes,
            reference["ga"],
            assume_unique=False
        )

        common_novel = ~(
            base_seen | ga_seen
        )

        counts["Total_Rows"] += len(chunk)

        counts["Baseline_Seen_Rows"] += int(
            base_seen.sum()
        )

        counts["GA_Seen_Rows"] += int(
            ga_seen.sum()
        )

        counts["Common_Novel_Rows"] += int(
            common_novel.sum()
        )

        counts["Baseline_Only_Novel_Rows"] += int(
            (~base_seen).sum()
        )

        counts["GA_Only_Novel_Rows"] += int(
            (~ga_seen).sum()
        )

        # Baseline prediction
        start = time.perf_counter()

        p_base = baseline_model.predict_proba(
            X_base
        )[:, 1]

        baseline_prediction_time += (
            time.perf_counter() - start
        )

        # GA prediction
        start = time.perf_counter()

        p_ga = ga_model.predict_proba(
            X_ga
        )[:, 1]

        ga_prediction_time += (
            time.perf_counter() - start
        )

        baseline_true.append(y)
        baseline_prob.append(p_base)

        ga_true.append(y)
        ga_prob.append(p_ga)

    # Combine predictions from all chunks.
    y_true = np.concatenate(baseline_true)

    probabilities_base = np.concatenate(
        baseline_prob
    )

    probabilities_ga = np.concatenate(
        ga_prob
    )

    baseline_seen_parts = []
    ga_seen_parts = []

    # Reconstruct novelty masks for metric subsets.
    for chunk in pd.read_csv(
        data_file,
        chunksize=CHUNK_SIZE,
        usecols=base_features
    ):

        X_base = prepare_features(
            chunk,
            base_features
        )

        X_ga = X_base[ga_features]

        base_seen = np.isin(
            calculate_hashes(X_base),
            reference["baseline"]
        )

        ga_seen = np.isin(
            calculate_hashes(X_ga),
            reference["ga"]
        )

        baseline_seen_parts.append(base_seen)
        ga_seen_parts.append(ga_seen)

    base_seen_all = np.concatenate(
        baseline_seen_parts
    )

    ga_seen_all = np.concatenate(
        ga_seen_parts
    )

    common_novel = ~(
        base_seen_all | ga_seen_all
    )

    result_rows = []

    for model_name, probabilities, feature_count, elapsed in [

        (
            "Baseline",
            probabilities_base,
            len(base_features),
            baseline_prediction_time
        ),

        (
            "GA",
            probabilities_ga,
            len(ga_features),
            ga_prediction_time
        )

    ]:

        # Complete partition evaluation.
        result_rows.append(
            calculate_metrics(
                y_true,
                probabilities,
                model_name,
                partition,
                "Full",
                feature_count,
                elapsed
            )
        )

        # Common novel subset evaluation.
        result_rows.append(
            calculate_metrics(
                y_true[common_novel],
                probabilities[common_novel],
                model_name,
                partition,
                "Common_Novel",
                feature_count,
                elapsed
            )
        )

    coverage = {

        "Partition": partition,

        **counts,

        "Common_Novel_Percentage": (
            counts["Common_Novel_Rows"]
            / counts["Total_Rows"] * 100
        ),

        "Baseline_Seen_Percentage": (
            counts["Baseline_Seen_Rows"]
            / counts["Total_Rows"] * 100
        ),

        "GA_Seen_Percentage": (
            counts["GA_Seen_Rows"]
            / counts["Total_Rows"] * 100
        )

    }

    return result_rows, coverage


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("NOVEL-PATTERN GENERALIZATION EVALUATION")
    print("=" * 70)

    baseline_package = joblib.load(
        MODEL_DIR / "baseline_random_forest.joblib"
    )

    ga_package = joblib.load(
        MODEL_DIR / "ga_random_forest.joblib"
    )

    baseline_features = baseline_package["features"]

    ga_features = ga_package["features"]

    # Build reference feature hashes for all partitions.
    indexes = build_reference_indexes(
        baseline_features,
        ga_features
    )

    references = prepare_reference_sets(indexes)

    all_results = []
    all_coverage = []

    for partition in ["validation", "test"]:

        results, coverage = evaluate_partition(
            partition,
            references[partition],
            baseline_package,
            ga_package
        )

        all_results.extend(results)

        all_coverage.append(coverage)

        print("\nCoverage summary:")
        print(pd.DataFrame([coverage]).to_string(index=False))

    results_df = pd.DataFrame(all_results)

    coverage_df = pd.DataFrame(all_coverage)

    results_df.to_csv(
        REPORT_DIR / "novel_pattern_model_comparison.csv",
        index=False
    )

    coverage_df.to_csv(
        REPORT_DIR / "novel_pattern_coverage.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("NOVEL-PATTERN EVALUATION COMPLETED")
    print("=" * 70)

    columns = [
        "Model",
        "Partition",
        "Subset",
        "Records",
        "Macro_F1",
        "Attack_Precision",
        "Attack_Recall",
        "False_Positive_Rate"
    ]

    print(
        results_df[columns].to_string(index=False)
    )

    print("\nSaved:")
    print("novel_pattern_model_comparison.csv")
    print("novel_pattern_coverage.csv")


if __name__ == "__main__":
    main()
