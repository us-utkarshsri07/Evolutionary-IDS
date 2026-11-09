
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000

THRESHOLDS = [
    0.50, 0.30, 0.20, 0.10,
    0.05, 0.02, 0.01, 0.005,
    0.002, 0.001, 0.0005
]


def calculate_metrics(y_true, probabilities, threshold):

    y_pred = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    attack_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) else 0
    )

    return {
        "Threshold": threshold,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": attack_f1,
        "Macro_F1": macro_f1,
        "False_Positive_Rate": fpr,
        "Predicted_Attack_Count": int(y_pred.sum()),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def main():

    print("\nTHRESHOLD CALIBRATION")
    print("=" * 65)

    baseline_package = joblib.load(
        MODEL_DIR / "baseline_random_forest.joblib"
    )

    ga_package = joblib.load(
        MODEL_DIR / "ga_random_forest.joblib"
    )

    baseline_model = baseline_package["pipeline"]
    ga_model = ga_package["pipeline"]

    baseline_features = baseline_package["features"]
    ga_features = ga_package["features"]

    y_parts = []

    baseline_parts = []
    ga_parts = []

    validation_file = DATA_DIR / "validation.csv.gz"

    print("\nGenerating validation probabilities...")

    for chunk in pd.read_csv(
        validation_file,
        chunksize=CHUNK_SIZE,
        usecols=baseline_features + ["target"]
    ):

        X_base = chunk[baseline_features].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X_base = X_base.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X_ga = X_base[ga_features]

        baseline_parts.append(
            baseline_model.predict_proba(X_base)[:, 1]
        )

        ga_parts.append(
            ga_model.predict_proba(X_ga)[:, 1]
        )

        y_parts.append(
            chunk["target"].to_numpy(dtype=np.int8)
        )

    y_true = np.concatenate(y_parts)

    probabilities = {
        "Baseline": np.concatenate(baseline_parts),
        "GA": np.concatenate(ga_parts)
    }

    all_results = []
    recommendations = {}

    for model_name, probs in probabilities.items():

        print(f"\nAnalyzing {model_name}...")

        model_results = []

        for threshold in THRESHOLDS:

            result = calculate_metrics(
                y_true,
                probs,
                threshold
            )

            result["Model"] = model_name

            model_results.append(result)

        # Select best attack F1.
        # Break ties using lower false-positive rate.
        best = sorted(
            model_results,
            key=lambda x: (
                -x["Attack_F1"],
                x["False_Positive_Rate"]
            )
        )[0]

        recommendations[model_name] = best

        all_results.extend(model_results)

        print(
            f"Recommended threshold: {best['Threshold']}"
        )

        print(
            f"Attack F1: {best['Attack_F1']:.5f}"
        )

        print(
            f"Attack recall: {best['Attack_Recall']:.5f}"
        )

        print(
            f"False positive rate: "
            f"{best['False_Positive_Rate']:.5f}"
        )

    results_df = pd.DataFrame(all_results)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        REPORT_DIR / "threshold_calibration.csv",
        index=False
    )

    with open(
        REPORT_DIR / "threshold_recommendations.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            recommendations,
            f,
            indent=4
        )

    print("\n" + "=" * 65)
    print("THRESHOLD CALIBRATION COMPLETED")
    print("=" * 65)

    print(
        results_df[
            [
                "Model",
                "Threshold",
                "Attack_Precision",
                "Attack_Recall",
                "Attack_F1",
                "False_Positive_Rate"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print("outputs/reports/threshold_calibration.csv")
    print("outputs/reports/threshold_recommendations.json")


if __name__ == "__main__":
    main()
