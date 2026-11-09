
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

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000


def load_features():

    with open(
        DATA_DIR / "feature_schema.json",
        encoding="utf-8"
    ) as f:
        return json.load(f)["features"]


def load_model_packages():

    baseline = joblib.load(
        MODEL_DIR / "baseline_random_forest.joblib"
    )

    ga = joblib.load(
        MODEL_DIR / "ga_random_forest.joblib"
    )

    with open(
        REPORT_DIR / "threshold_recommendations.json",
        encoding="utf-8"
    ) as f:
        recommendations = json.load(f)

    thresholds = {
        "Baseline": float(
            recommendations["Baseline"]["Threshold"]
        ),
        "GA": float(
            recommendations["GA"]["Threshold"]
        )
    }

    return baseline, ga, thresholds


def prepare_features(chunk, features):

    X = chunk[features].apply(
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


def build_reference_indexes(
    baseline_features,
    ga_features
):

    indexes = {
        "Baseline": [],
        "GA": []
    }

    for partition in ["train", "validation"]:

        print(f"Building reference: {partition}")

        path = DATA_DIR / f"{partition}.csv.gz"

        for chunk in pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            usecols=baseline_features
        ):

            X_base = prepare_features(
                chunk,
                baseline_features
            )

            X_ga = X_base[ga_features]

            indexes["Baseline"].append(
                calculate_hashes(X_base)
            )

            indexes["GA"].append(
                calculate_hashes(X_ga)
            )

    return {
        model: np.unique(
            np.concatenate(parts)
        )
        for model, parts in indexes.items()
    }


def compute_metrics(
    y_true,
    probabilities,
    threshold
):

    if len(y_true) == 0:
        raise ValueError("Empty evaluation subset.")

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    return {

        "Records": len(y_true),

        "Accuracy": accuracy_score(
            y_true, predictions
        ),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),

        "Attack_Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Attack_Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Attack_F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Macro_F1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "False_Positive_Rate": fpr,

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "PR_AUC": average_precision_score(
            y_true,
            probabilities
        ),

        "MCC": matthews_corrcoef(
            y_true,
            predictions
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)

    }


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    baseline_package, ga_package, thresholds = (
        load_model_packages()
    )

    baseline_model = baseline_package["pipeline"]
    ga_model = ga_package["pipeline"]

    baseline_features = baseline_package["features"]
    ga_features = ga_package["features"]

    print("\nFINAL CALIBRATED TEST EVALUATION")
    print("=" * 70)

    print("Baseline threshold:", thresholds["Baseline"])
    print("GA threshold:", thresholds["GA"])

    # Build training + validation reference hashes.
    references = build_reference_indexes(
        baseline_features,
        ga_features
    )

    # Accumulate test results.
    predictions = {

        "Baseline": {
            "probabilities": [],
            "time": 0.0
        },

        "GA": {
            "probabilities": [],
            "time": 0.0
        }

    }

    labels = []
    common_novel_parts = []

    test_file = DATA_DIR / "test.csv.gz"

    print("\nEvaluating Friday test data...")

    for chunk in pd.read_csv(
        test_file,
        chunksize=CHUNK_SIZE,
        usecols=baseline_features + ["target"]
    ):

        X_base = prepare_features(
            chunk,
            baseline_features
        )

        X_ga = X_base[ga_features]

        y = chunk["target"].to_numpy(
            dtype=np.int8
        )

        base_hashes = calculate_hashes(X_base)
        ga_hashes = calculate_hashes(X_ga)

        base_seen = np.isin(
            base_hashes,
            references["Baseline"]
        )

        ga_seen = np.isin(
            ga_hashes,
            references["GA"]
        )

        common_novel = ~(
            base_seen | ga_seen
        )

        common_novel_parts.append(
            common_novel
        )

        labels.append(y)

        start = time.perf_counter()

        base_prob = baseline_model.predict_proba(
            X_base
        )[:, 1]

        predictions["Baseline"]["time"] += (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        ga_prob = ga_model.predict_proba(
            X_ga
        )[:, 1]

        predictions["GA"]["time"] += (
            time.perf_counter() - start
        )

        predictions["Baseline"]["probabilities"].append(
            base_prob
        )

        predictions["GA"]["probabilities"].append(
            ga_prob
        )

    y_test = np.concatenate(labels)

    common_novel = np.concatenate(
        common_novel_parts
    )

    print(f"\nTotal test records: {len(y_test):,}")

    print(
        f"Common novel records: "
        f"{common_novel.sum():,}"
    )

    results = []

    for model_name in ["Baseline", "GA"]:

        model_threshold = thresholds[model_name]

        probabilities = np.concatenate(
            predictions[model_name]["probabilities"]
        )

        elapsed = predictions[model_name]["time"]

        feature_count = (
            len(baseline_features)
            if model_name == "Baseline"
            else len(ga_features)
        )

        for subset_name, mask in [
            ("Full", np.ones(len(y_test), dtype=bool)),
            ("Common_Novel", common_novel)
        ]:

            metrics = compute_metrics(
                y_test[mask],
                probabilities[mask],
                model_threshold
            )

            metrics.update({

                "Model": model_name,

                "Subset": subset_name,

                "Features": feature_count,

                "Threshold": model_threshold,

                "Prediction_Time_Seconds": elapsed

            })

            results.append(metrics)

            print(
                f"\n{model_name} | {subset_name}"
            )

            for key, value in metrics.items():

                if isinstance(value, (int, float)):
                    print(f"{key}: {value}")

    # Save final metrics.
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "final_calibrated_test_results.csv",
        index=False
    )

    coverage = pd.DataFrame([{

        "Test_Rows": len(y_test),

        "Common_Novel_Rows": int(common_novel.sum()),

        "Common_Novel_Percentage": (
            common_novel.mean() * 100
        ),

        "Baseline_Threshold": thresholds["Baseline"],

        "GA_Threshold": thresholds["GA"]

    }])

    coverage.to_csv(
        REPORT_DIR / "final_test_coverage.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 70)

    print("\nSaved:")
    print("final_calibrated_test_results.csv")
    print("final_test_coverage.csv")


if __name__ == "__main__":
    main()
