# Evolutionary IDS

> A Machine Learning-based Intrusion Detection System that uses Genetic Algorithm (GA) for feature selection and Random Forest classifiers to detect malicious network traffic.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)

---

## Overview

Evolutionary IDS is a machine learning-based Intrusion Detection System designed to identify malicious network traffic using the CIC-IDS2017 dataset.

The project investigates whether evolutionary feature selection can reduce the dimensionality of network traffic data while maintaining effective intrusion detection performance.

Instead of relying on a single classification model, the system implements two Random Forest approaches:

- **Baseline Random Forest:** Trained using all 69 available features.
- **GA-Optimized Random Forest:** Trained using 27 features selected through a Genetic Algorithm.

The two approaches are evaluated and compared through a Streamlit dashboard, providing insights into classification performance, feature selection, model behavior, and detection results.

### Core Components

- CIC-IDS2017 network traffic dataset
- Data preprocessing and cleaning
- Feature scaling using StandardScaler
- Genetic Algorithm-based feature selection
- Random Forest classification
- Baseline versus optimized model comparison
- Model evaluation and performance analysis
- Traffic detection through CSV uploads
- Chunk-based inference for large datasets
- Interactive Streamlit dashboard

---

## Key Highlights

| Component | Implementation |
|---|---|
| Dataset | CIC-IDS2017 |
| Baseline Features | 69 |
| GA-Selected Features | 27 |
| Feature Reduction | Approximately 60.9% |
| Baseline Classifier | Random Forest |
| Optimized Classifier | Random Forest |
| Feature Selection | Genetic Algorithm |
| Dashboard | Streamlit |
| Prediction Task | Binary Classification |

### Genetic Algorithm Feature Selection

The Genetic Algorithm searches for an effective subset of features by iteratively evaluating candidate feature combinations.

The optimization process involves:

1. Population initialization
2. Fitness evaluation
3. Selection
4. Crossover
5. Mutation
6. Best feature subset selection

The resulting 27-feature subset is used to train the optimized Random Forest classifier.

---

## System Architecture

<p align="center">
  <img src="assets/Evolutionary IDS Model Architecture.png" width="100%">
</p>

The system follows two parallel classification pipelines:

### Baseline Pipeline

```text
CIC-IDS2017 Dataset
        |
        v
   Preprocessing
        |
        v
  All 69 Features
        |
        v
 Random Forest
        |
        v
   Predictions
```

### GA-Optimized Pipeline

```text
CIC-IDS2017 Dataset
        |
        v
   Preprocessing
        |
        v
 Genetic Algorithm
        |
        v
  Selected 27 Features
        |
        v
 Random Forest
        |
        v
   Predictions
```

Both models are evaluated and compared to understand the impact of evolutionary feature selection on intrusion detection.

---

## Project Structure

## Project Structure

| **Folder / File** | **Purpose** |
|---|---|
| `app.py` | Main Streamlit application and dashboard |
| `models/` | Trained machine learning models |
| `models/baseline_random_forest.joblib` | Baseline Random Forest model (69 features) |
| `models/ga_random_forest.joblib` | GA-optimized Random Forest model (27 features) |
| `models/ga_selected_features.json` | Selected features from Genetic Algorithm |
| `data/processed/` | Processed data and feature metadata |
| `data/processed/feature_schema.json` | Expected input feature schema |
| `outputs/reports/` | Model evaluation results and analysis reports |
| `outputs/reports/model_comparison.csv` | Baseline versus GA model comparison |
| `outputs/reports/baseline_metrics.json` | Baseline model evaluation metrics |
| `outputs/reports/comparison_metadata.json` | Model comparison metadata |
| `outputs/reports/threshold_recommendations.json` | Threshold recommendations |
| `outputs/reports/ga_fitness_history.csv` | Genetic Algorithm fitness history |
| `outputs/reports/ga_selected_features.csv` | GA-selected feature details |
| `assets/` | Project images and visual assets |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignored files and directories |
| `README.md` | Project documentation |

*Note: The structure above highlights the primary application and model artifacts. Additional reports and project files may be present in the repository. Large raw and processed datasets are not included.*

---

## Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Dashboard | Streamlit |
| Machine Learning | Scikit-learn |
| Classification Algorithm | Random Forest |
| Feature Selection | Genetic Algorithm |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Visualization | Plotly |
| Model Serialization | Joblib |
| Data Format | CSV, JSON |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

---

## Execution Flow

```text
CIC-IDS2017 Dataset
        |
        v
 Data Preprocessing
        |
        v
 Feature Preparation
        |
        +-------------------------+
        |                         |
        v                         v
 Baseline Pipeline          GA Pipeline
        |                         |
        v                         v
  69 Features              Genetic Algorithm
        |                         |
        v                         v
 Random Forest              27 Features
        |                         |
        |                         v
        |                    Random Forest
        |                         |
        v                         v
 Baseline Results           GA Results
        |                         |
        +------------+------------+
                     |
                     v
             Model Comparison
                     |
                     v
             Performance Analysis
                     |
                     v
              Streamlit Dashboard
```

---

## Model Architecture

### 1. Baseline Random Forest

The baseline model uses all 69 available input features to classify network traffic.

It serves as the reference model for evaluating the effectiveness of feature selection.

### 2. GA-Optimized Random Forest

The optimized model uses a reduced subset of 27 features identified through Genetic Algorithm-based feature selection.

The objective is to reduce feature dimensionality while retaining useful predictive information.

### Model Comparison

The two models are compared using the following evaluation metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Proportion of predicted attacks that are actual attacks |
| Recall | Proportion of actual attacks correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish between classes |
| MCC | Balanced measure of classification quality |

The dashboard also provides model comparison and feature-related visualizations.

---

## Dashboard

The project includes an interactive Streamlit dashboard for exploring the trained models and analyzing network traffic.

### Dashboard Capabilities

- Model overview and performance comparison
- Baseline versus GA-optimized model analysis
- Feature selection insights
- Genetic Algorithm fitness analysis
- Model evaluation visualizations
- Traffic detection using uploaded CSV files
- Prediction summaries and result exploration
- Large-file inference using chunk-based processing

### Traffic Detection

Users can upload network traffic CSV files and perform predictions using the trained models.

The application supports:

- CSV uploads
- Gzip-compressed CSV uploads (`.csv.gz`)
- Feature schema validation
- Chunk-based processing
- Prediction summaries

Large raw datasets are processed in chunks to avoid loading the entire dataset into memory at once.

---

## Installation and Setup

### Prerequisites

- Python 3.11
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/us-utkarshsri07/Evolutionary-IDS.git

cd Evolutionary-IDS
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Select the `main` branch.
5. Set `app.py` as the application entry point.
6. Configure Python 3.11.
7. Deploy the application.

The deployment requires the trained model artifacts, feature schema, and report files referenced by the application.

---

## Key Design Decisions

### Why Random Forest?

Random Forest provides an ensemble-based classification approach that combines multiple decision trees to produce predictions.

It is used as the classification algorithm in both experimental pipelines, allowing the impact of feature selection to be studied separately.

### Why Genetic Algorithm?

Genetic Algorithms provide an evolutionary optimization approach for searching feature subsets.

The objective is to identify a smaller set of informative features rather than relying on all available input features.

### Why Compare Two Models?

Using the same classification algorithm with different feature sets provides a way to investigate the contribution of feature selection.

The comparison considers both predictive performance and feature dimensionality.

### Why Chunk-Based Inference?

Large network traffic CSV files can contain hundreds of thousands of records.

Chunk-based inference processes records in smaller batches, helping control memory consumption during prediction.

---

## Limitations

- Model performance depends on the quality and distribution of the training dataset.
- CIC-IDS2017 may not represent every real-world network environment.
- Feature selection results depend on the Genetic Algorithm configuration and fitness evaluation.
- Large CSV uploads can be limited by network speed and hosting infrastructure.
- The trained models are dependent on the expected feature schema and preprocessing pipeline.
- Detection performance on unseen traffic may differ from benchmark dataset performance.

---

## Future Improvements

- Real-time network traffic monitoring
- Integration with live packet capture
- Multi-class attack classification
- Evaluation on additional intrusion detection datasets
- Advanced evolutionary optimization techniques
- Model explainability using SHAP
- Improved inference performance
- Streaming detection and alert generation
- Automated model retraining
- Deployment through a scalable inference API

---

## Project Status

**Status:** Developed and deployed

Evolutionary IDS currently implements two Random Forest classification pipelines, compares their performance, and provides an interactive dashboard for network traffic analysis.

The central experimental objective is to investigate the effectiveness of Genetic Algorithm-based feature selection in reducing the number of input features from 69 to 27.

---

## License

This project is available under the MIT License.

---

## Author

**Utkarsh Srivastava**

B.Tech Computer Science and Engineering

GitHub: [@us-utkarshsri07](https://github.com/us-utkarshsri07)

---

<p align="center">
  <b>Evolutionary IDS</b><br>
  Exploring evolutionary feature selection for machine learning-based intrusion detection.
</p>
