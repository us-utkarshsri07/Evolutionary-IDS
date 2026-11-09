
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "baseline_random_forest.joblib"
VALIDATION_PATH = BASE_DIR / "data" / "processed" / "validation.csv.gz"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

CHUNK_SIZE = 100_000

THRESHOLDS = [
    0.50, 0.30, 0.20, 0.10,
    0.05, 0.02, 0.01, 0.005,
    0.002, 0.001, 0.0005
]


def main():

    print("Loading baseline model...")

    package = joblib.load(MODEL_PATH)

    model = package["pipeline"]
    features = package["features"]

    y_true_parts = []
    probabilities_parts = []

    print("Reading validation dataset...")

    for chunk in pd.read_csv(
        VALIDATION_PATH,
        chunksize=CHUNK_SIZE,
        usecols=features + ["target"]
    ):

        X = chunk[features].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        probabilities = model.predict_proba(X)[:, 1]

        probabilities_parts.append(probabilities)

        y_true_parts.append(
            chunk["target"].to_numpy()
        )

    y_true = np.concatenate(y_true_parts)

    y_prob = np.concatenate(probabilities_parts)

    results = []

    for threshold in THRESHOLDS:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true, y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true, y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred,
            zero_division=0
        )

        fpr = fp / (fp + tn)

        results.append({

            "Threshold": threshold,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "False_Positive_Rate": fpr,

            "True_Positive": tp,

            "False_Positive": fp,

            "False_Negative": fn,

            "True_Negative": tn

        })

    report = pd.DataFrame(results)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = REPORT_DIR / "baseline_threshold_analysis.csv"

    report.to_csv(
        output,
        index=False
    )

    print("\nVALIDATION THRESHOLD COMPARISON")
    print("=" * 85)

    print(
        report.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
