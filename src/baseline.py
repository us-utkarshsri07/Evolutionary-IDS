
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
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

SEED = 42
CHUNK_SIZE = 100_000

TRAIN_SAMPLE_FRACTION = 0.15
MAX_TRAIN_ROWS = 250_000

THRESHOLD = 0.50


# ==================================================
# LOAD FEATURE SCHEMA
# ==================================================

def load_feature_schema():

    schema_path = DATA_DIR / "feature_schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            "feature_schema.json not found. "
            "Run preprocessing.py first."
        )

    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    features = schema["features"]

    print(f"Features loaded: {len(features)}")

    return features


# ==================================================
# LOAD TRAINING SAMPLE
# ==================================================

def load_training_sample(features):

    train_file = DATA_DIR / "train.csv.gz"

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    print("\nLoading training data in chunks...")

    samples = []

    selected_columns = features + ["target"]

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            train_file,
            chunksize=CHUNK_SIZE,
            usecols=selected_columns
        )
    ):

        chunk = chunk[selected_columns]

        # Uniform sampling within each chunk.
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

    X_train = train_df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y_train = train_df["target"].astype("int8")

    if y_train.nunique() != 2:
        raise ValueError(
            "Training sample must contain both classes."
        )

    print(f"Training sample size: {len(X_train):,}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    return X_train, y_train


# ==================================================
# BUILD RANDOM FOREST PIPELINE
# ==================================================

def build_model():

    model = Pipeline([
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

    return model


# ==================================================
# EVALUATION
# ==================================================

def evaluate_partition(model, features, partition):

    data_file = DATA_DIR / f"{partition}.csv.gz"

    print(f"\nEvaluating {partition.upper()}...")

    y_true_parts = []
    y_pred_parts = []
    y_prob_parts = []

    start_time = time.perf_counter()

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

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        y_true_parts.append(y)

        y_pred_parts.append(predictions)

        y_prob_parts.append(probabilities)

    prediction_time = time.perf_counter() - start_time

    y_true = np.concatenate(y_true_parts)

    y_pred = np.concatenate(y_pred_parts)

    y_prob = np.concatenate(y_prob_parts)

    # Confusion matrix:
    # TN FP
    # FN TP

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    metrics = {

        "partition": partition,

        "total_records": int(len(y_true)),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),

        "attack_precision": float(
            precision_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "attack_recall": float(
            recall_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "attack_f1": float(
            f1_score(
                y_true, y_pred,
                pos_label=1,
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),

        "average_precision": float(
            average_precision_score(y_true, y_prob)
        ),

        "matthews_correlation": float(
            matthews_corrcoef(y_true, y_pred)
        ),

        "true_negative": int(tn),

        "false_positive": int(fp),

        "false_negative": int(fn),

        "true_positive": int(tp),

        "prediction_time_seconds": float(
            prediction_time
        ),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            output_dict=True,
            zero_division=0
        )

    }

    print("\n" + "=" * 60)

    print(f"{partition.upper()} RESULTS")

    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, (int, float)):

            print(f"{key:30}: {value}")

    print("\nConfusion Matrix:")

    print(
        np.array([
            [tn, fp],
            [fn, tp]
        ])
    )

    # Save confusion matrix
    cm_df = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp]
        ],
        index=["Actual Benign", "Actual Attack"],
        columns=["Predicted Benign", "Predicted Attack"]
    )

    cm_df.to_csv(
        REPORT_DIR / f"baseline_confusion_matrix_{partition}.csv"
    )

    return metrics


# ==================================================
# MAIN TRAINING FUNCTION
# ==================================================

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("INTRUSION DETECTION - RANDOM FOREST BASELINE")
    print("=" * 65)

    # 1. Feature schema
    features = load_feature_schema()

    # 2. Training sample
    X_train, y_train = load_training_sample(features)

    # 3. Model
    model = build_model()

    print("\nTraining Random Forest...")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print("\nTraining completed.")

    print(
        f"Training time: {training_time:.2f} seconds"
    )

    # 4. Save fitted model
    model_package = {

        "pipeline": model,

        "features": features,

        "threshold": THRESHOLD,

        "experiment": "Day-based Random Forest baseline",

        "random_state": SEED

    }

    model_path = MODEL_DIR / "baseline_random_forest.joblib"

    joblib.dump(
        model_package,
        model_path,
        compress=3
    )

    print(f"\nModel saved: {model_path}")

    # 5. Evaluate complete validation and test partitions
    results = {}

    for partition in ["validation", "test"]:

        results[partition] = evaluate_partition(
            model,
            features,
            partition
        )

    # Add training metadata
    results["training"] = {

        "training_sample_size": len(X_train),

        "original_training_rows": 1_668_530,

        "number_of_features": len(features),

        "training_time_seconds": training_time,

        "threshold": THRESHOLD

    }

    # 6. Save metrics
    metrics_path = REPORT_DIR / "baseline_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # 7. Feature importance
    classifier = model.named_steps["classifier"]

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance": classifier.feature_importances_

    }).sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        REPORT_DIR / "baseline_feature_importance.csv",
        index=False
    )

    print("\n" + "=" * 65)

    print("BASELINE EXPERIMENT COMPLETED")

    print("=" * 65)

    print("\nTop 15 important features:")

    print(
        importance_df.head(15).to_string(index=False)
    )

    print("\nSaved files:")
    print("models/baseline_random_forest.joblib")
    print("outputs/reports/baseline_metrics.json")
    print("outputs/reports/baseline_feature_importance.csv")


if __name__ == "__main__":
    main()
