
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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

SEED = 42

CHUNK_SIZE = 100_000
TRAIN_SAMPLE_FRACTION = 0.15
MAX_TRAIN_ROWS = 250_000

THRESHOLD = 0.50


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_training_sample(features):

    train_file = DATA_DIR / "train.csv.gz"

    columns = features + ["target"]

    samples = []

    print("\nReconstructing baseline training sample...")

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            train_file,
            chunksize=CHUNK_SIZE,
            usecols=columns
        )
    ):

        chunk = chunk[columns]

        sample = chunk.sample(
            frac=TRAIN_SAMPLE_FRACTION,
            random_state=SEED + chunk_index
        )

        samples.append(sample)

    train_df = pd.concat(
        samples,
        ignore_index=True
    )

    if len(train_df) > MAX_TRAIN_ROWS:

        train_df = train_df.sample(
            n=MAX_TRAIN_ROWS,
            random_state=SEED
        )

    X = train_df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = train_df["target"].astype("int8")

    print(f"Training records: {len(X):,}")

    return X, y


# ============================================================
# CREATE GA MODEL
# ============================================================

def build_ga_model():

    return Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "classifier",
            RandomForestClassifier(

                n_estimators=100,

                max_depth=20,

                min_samples_leaf=2,

                class_weight="balanced_subsample",

                max_features="sqrt",

                random_state=SEED,

                n_jobs=-1

            )
        )

    ])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    features,
    partition,
    model_name
):

    data_file = DATA_DIR / f"{partition}.csv.gz"

    y_true_parts = []
    probability_parts = []

    print(
        f"\nEvaluating {model_name} on {partition.upper()}..."
    )

    start = time.perf_counter()

    for chunk in pd.read_csv(
        data_file,
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

        y = chunk["target"].to_numpy(dtype=np.int8)

        probabilities = model.predict_proba(X)[:, 1]

        y_true_parts.append(y)
        probability_parts.append(probabilities)

    prediction_time = time.perf_counter() - start

    y_true = np.concatenate(y_true_parts)

    y_probability = np.concatenate(
        probability_parts
    )

    y_pred = (
        y_probability >= THRESHOLD
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

    result = {

        "Model": model_name,

        "Partition": partition,

        "Features": len(features),

        "Records": len(y_true),

        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "Balanced_Accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),

        "Attack_Precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "Attack_Recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "Attack_F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "False_Positive_Rate": float(fpr),

        "ROC_AUC": float(
            roc_auc_score(
                y_true,
                y_probability
            )
        ),

        "PR_AUC": float(
            average_precision_score(
                y_true,
                y_probability
            )
        ),

        "MCC": float(
            matthews_corrcoef(
                y_true,
                y_pred
            )
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        "Prediction_Time_Seconds": float(
            prediction_time
        ),

        "Threshold": THRESHOLD

    }

    # Save confusion matrix
    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual_Benign", "Actual_Attack"],
        columns=["Predicted_Benign", "Predicted_Attack"]
    )

    cm.to_csv(
        REPORT_DIR /
        f"{model_name.lower()}_{partition}_confusion_matrix.csv"
    )

    print(
        f"Macro F1: {result['Macro_F1']:.5f}"
    )

    print(
        f"Attack Recall: {result['Attack_Recall']:.5f}"
    )

    print(
        f"False Positive Rate: {result['False_Positive_Rate']:.5f}"
    )

    return result


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("BASELINE VS GA FEATURE SELECTION")
    print("=" * 70)

    # Load previously trained baseline model
    baseline_package = joblib.load(
        MODEL_DIR / "baseline_random_forest.joblib"
    )

    baseline_model = baseline_package["pipeline"]

    baseline_features = baseline_package["features"]

    # Load GA feature selection output
    ga_result = load_json(
        MODEL_DIR / "ga_selected_features.json"
    )

    ga_features = ga_result["selected_features"]

    # Verify feature consistency
    if not set(ga_features).issubset(
        set(baseline_features)
    ):

        raise ValueError(
            "GA selected features do not match baseline schema."
        )

    print(f"\nBaseline features: {len(baseline_features)}")

    print(f"GA features: {len(ga_features)}")

    print(
        f"Reduction: "
        f"{(1 - len(ga_features) / len(baseline_features)) * 100:.2f}%"
    )

    # Recreate exact baseline training sample
    X_train, y_train = load_training_sample(
        baseline_features
    )

    # Train GA-selected Random Forest
    ga_model = build_ga_model()

    X_train_ga = X_train[ga_features]

    print("\nTraining GA-selected Random Forest...")

    start = time.perf_counter()

    ga_model.fit(
        X_train_ga,
        y_train
    )

    ga_training_time = time.perf_counter() - start

    print(
        f"GA model training completed: "
        f"{ga_training_time:.2f} seconds"
    )

    # Save GA model
    ga_package = {

        "pipeline": ga_model,

        "features": ga_features,

        "threshold": THRESHOLD,

        "experiment": "Random Forest with GA-selected features"

    }

    joblib.dump(
        ga_package,
        MODEL_DIR / "ga_random_forest.joblib",
        compress=3
    )

    # Evaluate both models on identical partitions
    results = []

    for partition in ["validation", "test"]:

        baseline_result = evaluate_model(
            baseline_model,
            baseline_features,
            partition,
            "Baseline"
        )

        ga_evaluation = evaluate_model(
            ga_model,
            ga_features,
            partition,
            "GA"
        )

        results.append(baseline_result)
        results.append(ga_evaluation)

    # Add training metadata
    metadata = {

        "Training_Records": len(X_train),

        "Baseline_Features": len(baseline_features),

        "GA_Features": len(ga_features),

        "Feature_Reduction_Percentage": (
            1 - len(ga_features) / len(baseline_features)
        ) * 100,

        "GA_Training_Time_Seconds": ga_training_time,

        "Threshold": THRESHOLD,

        "Random_State": SEED

    }

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    with open(
        REPORT_DIR / "comparison_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    display_columns = [
        "Model",
        "Partition",
        "Features",
        "Accuracy",
        "Macro_F1",
        "Attack_Precision",
        "Attack_Recall",
        "False_Positive_Rate",
        "Prediction_Time_Seconds"
    ]

    print(
        comparison[display_columns].to_string(
            index=False
        )
    )

    print("\nSaved models:")
    print("models/baseline_random_forest.joblib")
    print("models/ga_random_forest.joblib")

    print("\nSaved comparison:")
    print("outputs/reports/model_comparison.csv")


if __name__ == "__main__":
    main()
