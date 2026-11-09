from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
INFERENCE_CHUNK_SIZE = 50_000
SENTINEL_FEATURES = {
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
}
st.set_page_config(
    page_title="Evolutionary IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #0e1117;
    }
    [data-testid="stSidebar"] {
        background: #1b1b1b;
        border-right: 1px solid #30343b;
    }
    [data-testid="stMetric"] {
        background: #171b24;
        border: 1px solid #293241;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #60a5fa;
    }
    div.stButton > button[kind="primary"] {
        border-radius: 9px;
        font-weight: 600;
    }
    .section-caption {
        color: #9ca3af;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# ARTIFACT AND REPORT LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading trained models...")
def load_model_artifacts() -> tuple[dict[str, Any], ...]:
    """Load the two trained models and their metadata."""
    required_paths = [
        MODEL_DIR / "baseline_random_forest.joblib",
        MODEL_DIR / "ga_random_forest.joblib",
        MODEL_DIR / "ga_selected_features.json",
        REPORT_DIR / "threshold_recommendations.json",
        DATA_DIR / "feature_schema.json",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required project artifacts are missing:\n- "
            + "\n- ".join(missing_paths)
        )
    baseline_package = joblib.load(
        MODEL_DIR / "baseline_random_forest.joblib"
    )
    ga_package = joblib.load(
        MODEL_DIR / "ga_random_forest.joblib"
    )
    with (REPORT_DIR / "threshold_recommendations.json").open(
        "r", encoding="utf-8"
    ) as file:
        threshold_recommendations = json.load(file)
    with (DATA_DIR / "feature_schema.json").open(
        "r", encoding="utf-8"
    ) as file:
        feature_schema = json.load(file)
    with (MODEL_DIR / "ga_selected_features.json").open(
        "r", encoding="utf-8"
    ) as file:
        ga_metadata = json.load(file)
    return (
        baseline_package,
        ga_package,
        threshold_recommendations,
        feature_schema,
        ga_metadata,
    )
@st.cache_data(show_spinner=False)
def load_report(filename: str) -> pd.DataFrame | dict | None:
    """Load a report from the project's reports directory."""
    path = REPORT_DIR / filename
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return None
try:
    (
        baseline_package,
        ga_package,
        threshold_recommendations,
        feature_schema,
        ga_metadata,
    ) = load_model_artifacts()
    baseline_model = baseline_package["pipeline"]
    ga_model = ga_package["pipeline"]
    baseline_features = list(baseline_package["features"])
    ga_features = list(ga_package["features"])
    baseline_threshold = float(
        threshold_recommendations.get("Baseline", {}).get(
            "Threshold", baseline_package.get("threshold", 0.50)
        )
    )
    ga_threshold = float(
        threshold_recommendations.get("GA", {}).get(
            "Threshold", ga_package.get("threshold", 0.50)
        )
    )
    if not set(ga_features).issubset(set(baseline_features)):
        raise ValueError(
            "The GA feature list is not a subset of the baseline feature list."
        )
except Exception as exc:
    st.error("The application could not load its model artifacts.")
    st.exception(exc)
    st.stop()
# ============================================================
# GENERAL HELPERS
# ============================================================
def get_upload_compression(filename: str) -> str | None:
    """Return the compression format for supported uploaded files."""
    return "gzip" if filename.lower().endswith(".gz") else None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and reject duplicate names."""
    result = df.copy()
    result.columns = result.columns.astype(str).str.strip()
    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "The uploaded CSV contains duplicate column names after whitespace "
            f"normalization: {duplicated}"
        )
    return result
def prepare_model_input(
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Apply the same numeric and invalid-value handling as preprocessing."""
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
# The training preprocessing converted negative measurements to NaN,
# except for the two TCP initial-window fields where -1 is a sentinel.
    for feature in feature_names:
        if feature in SENTINEL_FEATURES:
            continue
        X[feature] = X[feature].mask(X[feature] < 0, np.nan)
    return X
def get_attack_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return the model's class-1 Random Forest score."""
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The loaded model does not contain attack class 1.")
    attack_index = classes.index(1)
    return probabilities[:, attack_index]
def evaluate_binary_arrays(
    actual: np.ndarray,
    predictions: np.ndarray,
    attack_scores: np.ndarray,
    unlabeled_records: int,
) -> dict[str, Any] | None:
    """Calculate the existing ground-truth metrics over all labeled chunks."""
    if len(actual) == 0:
        return None
    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1],
    ).ravel()
    has_both_classes = len(np.unique(actual)) == 2
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "labeled_records": int(len(actual)),
        "unlabeled_records": int(unlabeled_records),
        "actual_benign": int((actual == 0).sum()),
        "actual_attack": int((actual == 1).sum()),
        "accuracy": float(accuracy_score(actual, predictions)),
        "attack_precision": float(precision_score(actual, predictions, zero_division=0)),
        "attack_recall": float(recall_score(actual, predictions, zero_division=0)),
        "attack_f1": float(f1_score(actual, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(actual, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "false_positive_rate": false_positive_rate,
        "balanced_accuracy": (
            float(balanced_accuracy_score(actual, predictions))
            if has_both_classes else None
        ),
        "roc_auc": float(roc_auc_score(actual, attack_scores)) if has_both_classes else None,
        "pr_auc": (
            float(average_precision_score(actual, attack_scores))
            if np.any(actual == 1) else None
        ),
        "mcc": float(matthews_corrcoef(actual, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "has_both_classes": bool(has_both_classes),
    }
def run_chunked_inference(
    uploaded_file,
    model,
    feature_names: list[str],
    threshold: float,
    status_placeholder,
) -> dict[str, Any]:
    """Predict every CSV row in bounded-memory chunks and write results to disk."""
    uploaded_file.seek(0)
    compression = get_upload_compression(uploaded_file.name)
    chunk_reader = pd.read_csv(
        uploaded_file,
        chunksize=INFERENCE_CHUNK_SIZE,
        encoding="utf-8-sig",
        skipinitialspace=True,
        engine="c",
        low_memory=True,
        compression=compression,
    )
    output_handle = tempfile.NamedTemporaryFile(
        mode="w", suffix="_intrusion_predictions.csv", delete=False,
        encoding="utf-8-sig", newline="",
    )
    output_path = output_handle.name
    total_records = 0
    attacks_detected = 0
    benign_records = 0
    unlabeled_records = 0
    preview_parts: list[pd.DataFrame] = []
    preview_record_count = 0
    actual_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    label_column = None
    first_chunk = True
    try:
        for chunk_number, chunk in enumerate(chunk_reader, start=1):
            chunk = normalize_columns(chunk)
            if first_chunk:
                missing_features = [name for name in feature_names if name not in chunk.columns]
                if missing_features:
                    raise ValueError(
                        "The uploaded CSV is missing required model features: "
                        + ", ".join(missing_features)
                    )
                label_column = next(
                    (column for column in chunk.columns if column.strip().lower() == "label"),
                    None,
                )
                first_chunk = False
            if chunk.empty:
                continue
            X = prepare_model_input(chunk, feature_names)
            attack_scores = get_attack_scores(model, X)
            predictions = (attack_scores >= threshold).astype(np.int8)
            chunk_results = chunk.copy()
            chunk_results["Attack_Score"] = attack_scores
            chunk_results["Predicted_Class"] = np.where(predictions == 1, "ATTACK", "BENIGN")
            chunk_results["Decision_Threshold"] = float(threshold)
            attacks_detected += int(predictions.sum())
            benign_records += int(len(predictions) - predictions.sum())
            total_records += len(chunk)
            if label_column is not None:
                normalized_labels = chunk[label_column].astype("string").str.strip()
                valid_mask = normalized_labels.notna() & normalized_labels.ne("")
                valid_array = valid_mask.to_numpy(dtype=bool)
                unlabeled_records += int((~valid_array).sum())
                if valid_array.any():
                    actual = (
                        normalized_labels.loc[valid_mask]
                        .str.upper()
                        .ne("BENIGN")
                        .astype("int8")
                        .to_numpy()
                    )
                    actual_parts.append(actual)
                    prediction_parts.append(predictions[valid_array])
                    score_parts.append(attack_scores[valid_array])
                actual_class = np.full(len(chunk_results), "UNKNOWN", dtype=object)
                if valid_array.any():
                    actual_class[valid_array] = np.where(actual == 0, "BENIGN", "ATTACK")
                chunk_results["Actual_Class"] = actual_class
            chunk_results.to_csv(
                output_handle,
                index=False,
                header=(total_records == len(chunk)),
            )
            if preview_record_count < 100:
                remaining = 100 - preview_record_count
                preview_part = chunk_results.head(remaining)
                preview_parts.append(preview_part)
                preview_record_count += len(preview_part)
            status_placeholder.info(
                f"Processed {total_records:,} records across {chunk_number:,} chunks..."
            )
        output_handle.flush()
        output_handle.close()
        if total_records == 0:
            raise ValueError("The uploaded CSV contains no data rows.")
        metrics = None
        if label_column is not None:
            if actual_parts:
                metrics = evaluate_binary_arrays(
                    np.concatenate(actual_parts),
                    np.concatenate(prediction_parts),
                    np.concatenate(score_parts),
                    unlabeled_records,
                )
        preview = (
            pd.concat(preview_parts, ignore_index=True)
            if preview_parts else pd.DataFrame()
        )
        return {
            "output_path": output_path,
            "total_records": int(total_records),
            "attacks_detected": int(attacks_detected),
            "benign_records": int(benign_records),
            "metrics": metrics,
            "preview": preview,
            "chunk_size": INFERENCE_CHUNK_SIZE,
        }
    except Exception:
        output_handle.close()
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
def render_confusion_matrix(metrics: dict[str, Any]) -> None:
    matrix = pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        index=["Actual Benign", "Actual Attack"],
        columns=["Predicted Benign", "Predicted Attack"],
    )
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Confusion Matrix",
        labels={"x": "Predicted class", "y": "Actual class", "color": "Records"},
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)
def render_ground_truth_metrics(metrics: dict[str, Any] | None) -> None:
    st.divider()
    st.subheader("Ground-Truth Evaluation")
    if metrics is None:
        st.info(
            "No usable Label column was found. Predictions are shown, but "
            "accuracy, precision, recall and F1 cannot be calculated without "
            "ground-truth labels."
        )
        return
    st.caption(
        "Metrics are calculated only on rows with a non-empty Label value. "
        "Every non-BENIGN label is treated as an attack, matching the project's "
        "binary target definition."
    )
    if metrics["unlabeled_records"] > 0:
        st.warning(
            f"{metrics['unlabeled_records']:,} rows had missing or empty labels "
            "and were excluded from ground-truth metrics."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Attack Precision", f"{metrics['attack_precision']:.2%}")
    c3.metric("Attack Recall", f"{metrics['attack_recall']:.2%}")
    c4.metric("Attack F1", f"{metrics['attack_f1']:.4f}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
    c6.metric("False-Positive Rate", f"{metrics['false_positive_rate']:.2%}")
    c7.metric(
        "Balanced Accuracy",
        f"{metrics['balanced_accuracy']:.2%}"
        if metrics["balanced_accuracy"] is not None
        else "N/A",
    )
    c8.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.4f}"
        if metrics["roc_auc"] is not None
        else "N/A",
    )
    st.caption(
        f"Evaluated {metrics['labeled_records']:,} labeled rows: "
        f"{metrics['actual_benign']:,} benign and "
        f"{metrics['actual_attack']:,} attack."
    )
    render_confusion_matrix(metrics)
    confusion_data = pd.DataFrame(
        [
            {"Actual": "Benign", "Predicted": "Benign", "Count": metrics["tn"]},
            {"Actual": "Benign", "Predicted": "Attack", "Count": metrics["fp"]},
            {"Actual": "Attack", "Predicted": "Benign", "Count": metrics["fn"]},
            {"Actual": "Attack", "Predicted": "Attack", "Count": metrics["tp"]},
        ]
    )
    with st.expander("View confusion-matrix counts"):
        st.dataframe(confusion_data, use_container_width=True, hide_index=True)
# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("# 🛡️ Evolutionary IDS")
    st.caption("Machine Learning Based Intrusion Detection")
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Traffic Detection",
            "Model Comparison",
            "Feature Analysis",
            "Reports",
        ],
    )
    st.divider()
    st.markdown("### Model Information")
    st.caption("Baseline: Random Forest")
    st.caption(f"Baseline features: {len(baseline_features)}")
    st.caption(f"GA-selected features: {len(ga_features)}")
    st.caption(f"Baseline validation threshold: {baseline_threshold:.3f}")
    st.caption(f"GA validation threshold: {ga_threshold:.3f}")
    st.divider()
    st.caption("Dataset: CIC-IDS2017")
    st.caption("Task: Binary classification")
    st.caption("Input: Network-flow CSV files")
# ============================================================
# PAGE 1: DASHBOARD
# ============================================================
if page == "Dashboard":
    st.title("Intrusion Detection Dashboard")
    st.caption(
        "Network traffic classification using Random Forest and "
        "Genetic Algorithm feature selection."
    )
    st.divider()
    feature_reduction = (
        1.0 - len(ga_features) / len(baseline_features)
    ) * 100.0
    metric_file = load_report("final_calibrated_test_results.csv")
    baseline_test = None
    ga_test = None
    if isinstance(metric_file, pd.DataFrame) and not metric_file.empty:
        baseline_rows = metric_file[
            (metric_file["Model"] == "Baseline")
            & (metric_file["Subset"] == "Full")
        ]
        ga_rows = metric_file[
            (metric_file["Model"] == "GA")
            & (metric_file["Subset"] == "Full")
        ]
        if not baseline_rows.empty:
            baseline_test = baseline_rows.iloc[0]
        if not ga_rows.empty:
            ga_test = ga_rows.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Features", len(baseline_features))
    c2.metric("GA-Selected Features", len(ga_features))
    c3.metric("Feature Reduction", f"{feature_reduction:.2f}%")
    c4.metric("Dataset Records", "2.83M")
    st.divider()
    st.subheader("Calibrated Friday Test Performance")
    if baseline_test is not None and ga_test is not None:
        left, right = st.columns(2)
        with left:
            st.markdown("### Baseline Random Forest")
            st.metric("Macro F1", f"{baseline_test['Macro_F1']:.4f}")
            st.metric("Attack Recall", f"{baseline_test['Attack_Recall']:.2%}")
            st.metric("Attack Precision", f"{baseline_test['Attack_Precision']:.2%}")
            st.metric("False-Positive Rate", f"{baseline_test['False_Positive_Rate']:.2%}")
        with right:
            st.markdown("### GA-Selected Random Forest")
            st.metric("Macro F1", f"{ga_test['Macro_F1']:.4f}")
            st.metric("Attack Recall", f"{ga_test['Attack_Recall']:.2%}")
            st.metric("Attack Precision", f"{ga_test['Attack_Precision']:.2%}")
            st.metric("False-Positive Rate", f"{ga_test['False_Positive_Rate']:.2%}")
        comparison_chart = pd.DataFrame(
            [
                {
                    "Model": "Baseline",
                    "Macro F1": baseline_test["Macro_F1"],
                    "Attack Recall": baseline_test["Attack_Recall"],
                    "Attack Precision": baseline_test["Attack_Precision"],
                },
                {
                    "Model": "GA",
                    "Macro F1": ga_test["Macro_F1"],
                    "Attack Recall": ga_test["Attack_Recall"],
                    "Attack Precision": ga_test["Attack_Precision"],
                },
            ]
        )
        st.plotly_chart(
            px.bar(
                comparison_chart.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Baseline vs GA on Friday test data",
            ),
            use_container_width=True,
        )
    else:
        st.info("Final calibrated test metrics are not available yet.")
    st.divider()
    st.subheader("System Information")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Detection models**\n"
            "- Random Forest baseline\n"
            "- Random Forest using GA-selected features\n"
            "- Binary benign-versus-attack classification"
        )
    with right:
        st.markdown(
            "**Dataset and feature selection**\n"
            "- CIC-IDS2017\n"
            f"- {len(baseline_features)} model input features\n"
            f"- {len(ga_features)} GA-selected features\n"
            f"- {feature_reduction:.2f}% feature reduction"
        )
    st.info(
        "This application classifies uploaded network-flow CSV records. "
        "It does not capture live packets or monitor a live network interface. "
        "The displayed test metrics are dataset-experiment results, not a "
        "guarantee of real-world detection performance."
    )
# ============================================================
# PAGE 2: TRAFFIC DETECTION
# ============================================================
elif page == "Traffic Detection":
    st.title("Network Traffic Detection")
    st.caption(
        "Upload a CIC-IDS2017-compatible network-flow CSV for binary classification."
    )
    model_choice = st.selectbox(
        "Select detection model",
        ["Baseline Random Forest", "GA Optimized Random Forest"],
    )
    if model_choice == "Baseline Random Forest":
        active_model = baseline_model
        active_features = baseline_features
        recommended_threshold = baseline_threshold
    else:
        active_model = ga_model
        active_features = ga_features
        recommended_threshold = ga_threshold
    st.write(f"Required input features: **{len(active_features)}**")
    st.write(f"Validation-selected threshold: **{recommended_threshold:.3f}**")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.001,
        max_value=0.999,
        value=float(np.clip(recommended_threshold, 0.001, 0.999)),
        step=0.001,
        help=(
            "The default is selected using validation data. "
            "Changing it changes the attack/benign trade-off."
        ),
        format="%.3f",
    )
    uploaded_file = st.file_uploader(
        "Upload network-flow CSV",
        type=["csv", "gz"],
        help=(
            "Upload a .csv file or a gzip-compressed .csv.gz file. "
            "Compressed files reduce transfer time when network upload is the bottleneck. "
            f"All rows are processed in chunks of {INFERENCE_CHUNK_SIZE:,}."
        ),
    )
    if uploaded_file is not None:
        try:
# Read only a small preview first. The complete file is read later
# by the chunked inference function when the user clicks Analyze.
            uploaded_file.seek(0)
            preview_input = pd.read_csv(
                uploaded_file,
                nrows=10,
                encoding="utf-8-sig",
                skipinitialspace=True,
                engine="c",
                low_memory=True,
                compression=get_upload_compression(uploaded_file.name),
            )
            preview_input = normalize_columns(preview_input)
            uploaded_file.seek(0)
            missing_features = [
                feature for feature in active_features
                if feature not in preview_input.columns
            ]
            if missing_features:
                st.error(
                    "The uploaded CSV is missing required model features: "
                    + ", ".join(missing_features)
                )
            elif preview_input.empty:
                st.error("The uploaded CSV contains no data rows.")
            else:
                st.success(
                    f"File received: {uploaded_file.name} "
                    f"({uploaded_file.size / (1024 * 1024):.1f} MB)."
                )
                if uploaded_file.size >= 20 * 1024 * 1024:
                    st.info(
                        "For faster uploads on Streamlit Cloud, try uploading this "
                        "file as .csv.gz. Gzip compression reduces the amount of data "
                        "transferred; it does not change model predictions."
                    )

                with st.expander("Preview uploaded data"):
                    st.dataframe(preview_input, use_container_width=True)
                upload_signature = (
                    uploaded_file.name,
                    uploaded_file.size,
                    model_choice,
                    float(threshold),
                )
                if st.button(
                    "Analyze Network Traffic",
                    type="primary",
                    use_container_width=True,
                ):
                    status_placeholder = st.empty()
                    try:
                        with st.spinner("Processing the complete CSV in chunks..."):
                            inference_result = run_chunked_inference(
                                uploaded_file=uploaded_file,
                                model=active_model,
                                feature_names=active_features,
                                threshold=float(threshold),
                                status_placeholder=status_placeholder,
                            )
                        previous_result = st.session_state.get("latest_detection")
                        if previous_result:
                            previous_path = previous_result.get("output_path")
                            if previous_path and os.path.exists(previous_path):
                                try:
                                    os.remove(previous_path)
                                except OSError:
                                    pass
                        st.session_state["latest_detection"] = {
                            "signature": upload_signature,
                            "filename": uploaded_file.name,
                            "model": model_choice,
                            "threshold": float(threshold),
                            **inference_result,
                        }
                        status_placeholder.success(
                            f"Completed inference for {inference_result['total_records']:,} records."
                        )
                    except Exception as prediction_error:
                        status_placeholder.empty()
                        st.error(f"Traffic analysis failed: {prediction_error}")
                latest = st.session_state.get("latest_detection")
                if latest is not None and latest.get("signature") == upload_signature:
                    total_records = latest["total_records"]
                    attacks_detected = latest["attacks_detected"]
                    benign_records = latest["benign_records"]
                    st.divider()
                    st.subheader("Detection Results")
                    st.caption(
                        f"Processed the complete uploaded file in chunks of "
                        f"{latest['chunk_size']:,} rows."
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", f"{total_records:,}")
                    c2.metric("Predicted Attacks", f"{attacks_detected:,}")
                    c3.metric("Predicted Benign", f"{benign_records:,}")
                    prediction_counts = pd.DataFrame(
                        {
                            "Class": ["Attack", "Benign"],
                            "Count": [attacks_detected, benign_records],
                        }
                    )
                    pie = px.pie(
                        prediction_counts,
                        names="Class",
                        values="Count",
                        hole=0.55,
                        title="Predicted Traffic Distribution",
                        color="Class",
                        color_discrete_map={
                            "Attack": "#60a5fa",
                            "Benign": "#2563eb",
                        },
                    )
                    pie.update_layout(template="plotly_dark")
                    st.plotly_chart(pie, use_container_width=True)
                    render_ground_truth_metrics(latest["metrics"])
                    st.divider()
                    st.subheader("Individual Predictions")
                    st.caption(
                        "The table shows the first 100 predictions. The download "
                        "contains predictions for every processed row. Attack_Score "
                        "is the Random Forest output score and is not probability-calibrated."
                    )
                    preview_results = latest["preview"]
                    preferred_columns = [
                        "Predicted_Class",
                        "Attack_Score",
                        "Decision_Threshold",
                        "Actual_Class",
                    ]
                    visible_columns = [
                        column for column in preview_results.columns
                        if column in preferred_columns
                    ]
                    remaining_columns = [
                        column for column in preview_results.columns
                        if column not in visible_columns
                    ]
                    show_all_columns = st.checkbox(
                        "Show all uploaded and prediction columns",
                        value=False,
                    )
                    display_columns = (
                        preview_results.columns.tolist()
                        if show_all_columns
                        else remaining_columns[:10] + visible_columns
                    )
                    st.dataframe(
                        preview_results[display_columns],
                        use_container_width=True,
                        hide_index=True,
                    )
                    output_path = latest["output_path"]
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as prediction_file:
                            st.download_button(
                                "Download Full Prediction Results",
                                data=prediction_file,
                                file_name="intrusion_detection_results.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                    else:
                        st.error(
                            "The temporary prediction file is no longer available. "
                            "Please run the analysis again."
                        )
                elif latest is not None:
                    st.info(
                        "The previous prediction belongs to a different file, "
                        "model or threshold. Click Analyze Network Traffic to "
                        "run the current selection."
                    )
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV contains no readable data.")
        except Exception as upload_error:
            st.error(f"Unable to read the uploaded CSV: {upload_error}")
# ============================================================
# PAGE 3: MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    comparison = load_report("model_comparison.csv")
    calibrated_results = load_report("final_calibrated_test_results.csv")
    novel_results = load_report("novel_pattern_model_comparison.csv")
    novel_coverage = load_report("novel_pattern_coverage.csv")
    if isinstance(comparison, pd.DataFrame):
        st.subheader("Original 0.50-Threshold Experiment")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    if isinstance(calibrated_results, pd.DataFrame):
        st.subheader("Validation-Calibrated Friday Test Results")
        full_results = calibrated_results[
            calibrated_results["Subset"] == "Full"
        ].copy()
        st.dataframe(full_results, use_container_width=True, hide_index=True)
        available_metrics = [
            metric for metric in [
                "Macro_F1",
                "Attack_F1",
                "Attack_Recall",
                "Attack_Precision",
                "Accuracy",
                "False_Positive_Rate",
            ]
            if metric in full_results.columns
        ]
        if available_metrics:
            selected_metric = st.selectbox(
                "Metric to visualize",
                available_metrics,
            )
            fig = px.bar(
                full_results,
                x="Model",
                y=selected_metric,
                color="Model",
                title=f"Baseline vs GA: {selected_metric}",
                barmode="group",
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    if isinstance(novel_results, pd.DataFrame):
        st.subheader("Novel-Pattern Evaluation")
        st.caption(
            "Sensitivity analysis on records whose feature representation "
            "was not found in the reference partitions. This does not prove "
            "that the underlying network connections are unique."
        )
        st.dataframe(novel_results, use_container_width=True, hide_index=True)
    if isinstance(novel_coverage, pd.DataFrame):
        st.subheader("Novel-Pattern Coverage")
        st.dataframe(novel_coverage, use_container_width=True, hide_index=True)
# ============================================================
# PAGE 4: FEATURE ANALYSIS
# ============================================================
elif page == "Feature Analysis":
    st.title("Evolutionary Feature Analysis")
    importance = load_report("baseline_feature_importance.csv")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        st.subheader("Baseline Random Forest Feature Importance")
        top_features = importance.head(20).sort_values("Importance")
        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 20 Baseline Features",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Impurity-based Random Forest importance is model-specific and "
            "should not be interpreted as causal importance."
        )
    st.divider()
    st.subheader("Genetic Algorithm Selected Features")
    c1, c2 = st.columns(2)
    c1.metric("Original Features", len(baseline_features))
    c2.metric("Selected Features", len(ga_features))
    selected_features_df = pd.DataFrame(
        {"Selected Feature": ga_features}
    )
    st.dataframe(
        selected_features_df,
        use_container_width=True,
        hide_index=True,
    )
    fitness_history = load_report("ga_fitness_history.csv")
    if isinstance(fitness_history, pd.DataFrame) and not fitness_history.empty:
        st.divider()
        st.subheader("Genetic Algorithm Fitness Evolution")
        fig = go.Figure()
        if "Best_Fitness" in fitness_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=fitness_history["Generation"],
                    y=fitness_history["Best_Fitness"],
                    mode="lines+markers",
                    name="Best Fitness",
                )
            )
        if "Mean_Fitness" in fitness_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=fitness_history["Generation"],
                    y=fitness_history["Mean_Fitness"],
                    mode="lines+markers",
                    name="Mean Fitness",
                )
            )
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Generation",
            yaxis_title="Fitness",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("GA Experiment Metadata")
    st.json(ga_metadata)
# ============================================================
# PAGE 5: REPORTS
# ============================================================
elif page == "Reports":
    st.title("Experimental Reports")
    available_reports = [
        "baseline_metrics.json",
        "model_comparison.csv",
        "comparison_metadata.json",
        "final_calibrated_test_results.csv",
        "final_test_coverage.csv",
        "novel_pattern_model_comparison.csv",
        "novel_pattern_coverage.csv",
        "exact_cross_partition_audit.csv",
        "overlap_row_audit.csv",
        "ga_fitness_history.csv",
        "ga_selected_features.csv",
        "threshold_calibration.csv",
        "threshold_recommendations.json",
        "preprocessing_summary.csv",
        "eda_attack_distribution.csv",
        "eda_binary_distribution.csv",
        "data_quality_summary.csv",
    ]
    existing_reports = [
        filename for filename in available_reports
        if (REPORT_DIR / filename).exists()
    ]
    if not existing_reports:
        st.info("No generated reports were found in outputs/reports.")
    else:
        selected_report = st.selectbox(
            "Choose a report",
            existing_reports,
        )
        report = load_report(selected_report)
        st.subheader(selected_report)
        if isinstance(report, dict):
            st.json(report)
            download_content = json.dumps(report, indent=4).encode("utf-8")
            download_mime = "application/json"
        elif isinstance(report, pd.DataFrame):
            st.dataframe(report, use_container_width=True, hide_index=True)
            download_content = report.to_csv(index=False).encode("utf-8")
            download_mime = "text/csv"
        else:
            st.warning("The selected report could not be loaded.")
            download_content = None
            download_mime = "application/octet-stream"
        if download_content is not None:
            st.download_button(
                "Download Report",
                data=download_content,
                file_name=selected_report,
                mime=download_mime,
            )
# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Evolutionary IDS | CIC-IDS2017 | Random Forest + "
    "Genetic Algorithm Feature Selection"
)
