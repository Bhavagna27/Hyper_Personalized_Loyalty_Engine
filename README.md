# 🏦 Hyper-Personalized Credit Card Loyalty Recommendation Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-oriented AI-powered loyalty recommendation engine** that transforms raw transaction and customer profile data into actionable business insights. The system handles the full ML lifecycle — from data ingestion and validation, through feature engineering and customer segmentation, to personalized recommendation generation and an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
  - [CLI Commands](#cli-commands)
  - [Python API](#python-api)
  - [Streamlit Dashboard](#streamlit-dashboard)
- [Pipeline Stages](#-pipeline-stages)
  - [1. Data Ingestion & Validation](#1-data-ingestion--validation)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Customer Segmentation](#3-customer-segmentation)
  - [4. Model Training](#4-model-training)
  - [5. Recommendation Engine](#5-recommendation-engine)
  - [6. New Customer Prediction](#6-new-customer-prediction)
  - [7. LLM Personalization](#7-llm-personalization)
- [Dashboard](#-dashboard)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This engine processes **Excel workbooks** containing three core sheets:

| Sheet | Description |
|---|---|
| `Transaction_History` | Row-level transaction logs (purchases, rewards, engagement) |
| `Customer_Loyalty_Profile` | Demographic and behavioral customer attributes |
| `AI_Recommendations` | Pre-computed AI recommendations (optional) |

The pipeline **ingests → validates → engineers features → segments customers → trains models → generates recommendations**, all exposed through an interactive **Streamlit dashboard** with 5 pages.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   📥 DATA INPUT                              │
│            Excel Workbook (.xlsx)                             │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              1️⃣  INGESTION & VALIDATION                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Schema Check │→ │ Quality Rules│→│ Cleaned CSVs +     │  │
│  │              │  │              │  │ Validation Report  │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              2️⃣  FEATURE ENGINEERING                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │   RFM    │ │  Reward  │ │Engagement│ │    Shopping    │  │
│  │ Features │ │ Features │ │ Features │ │   Behavior    │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
│  ┌──────────────┐ ┌──────────────┐                          │
│  │Customer Value│ │   Derived    │ → customer_features.csv  │
│  └──────────────┘ └──────────────┘   feature_pipeline.*     │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│             3️⃣  CUSTOMER SEGMENTATION                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ K-Evaluation │→│KMeans Training│→│  Persona Assignment  │ │
│  │  (K=2..10)   │ │              │ │(Premium Traveler,    │ │
│  └──────────────┘ └──────────────┘ │ Digital Explorer…)   │ │
│                                    └──────────────────────┘ │
│  Outputs: cluster_profiles.csv, cluster_statistics.csv,      │
│           kmeans_model.joblib, Visualization PNGs            │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│         4️⃣  TRAINING & RECOMMENDATIONS                       │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │  Model Training  │    │    Recommendation Engine      │   │
│  │ (Random Forest)  │    │  (Hybrid Scoring + Top-3)    │   │
│  └──────────────────┘    └──────────────────────────────┘   │
│  Outputs: model_bundle.joblib, recommendations.csv,          │
│           model_evaluation.json, reward_catalog.csv          │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│            5️⃣  NEW CUSTOMER PREDICTION                       │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐   │
│  │    Feature   │→│    Cluster   │→│  Similarity Score  │   │
│  │  Engineering │ │  Assignment  │ │  + Recommendations │   │
│  └──────────────┘ └──────────────┘ └────────────────────┘   │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│            6️⃣  LLM PERSONALIZATION (Optional)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ OpenAI Integration → Natural Language Insights       │   │
│  │ Fallback: Deterministic Templates + Response Caching │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│               📊 INTERACTIVE DASHBOARD                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ Overview │ │Customer  │ │Recommend-│ │  New Customer  │ │
│  │          │ │Segment.  │ │  ations  │ │  Prediction    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
│  ┌──────────────────┐                                       │
│  │Business Analytics │  ← 5-page Streamlit App              │
│  └──────────────────┘                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Data Pipeline
- **Multi-sheet Excel ingestion** with auto-discovery and fallback handling
- **Structural & data-quality validation** — schema checks, missing value detection, duplicate detection, type coercion
- **Intelligent cleaning** — trimming, deduplication, median-fill imputation, categorical mode-fill
- **6 feature groups** — 40+ engineered features covering RFM, Rewards, Engagement, Shopping Behavior, Customer Value, and Derived metrics

### Customer Segmentation
- **Automatic K evaluation** — Elbow method + Silhouette score from K=2 to K=10
- **Business interpretability scoring** — ranks segmentation solutions by persona separation
- **Dynamic persona assignment** — data-driven naming (e.g., "Premium Traveler", "Digital Explorer") with textual rationales
- **5 visualization outputs** — Elbow curve, Silhouette scores, PCA 2D scatter, centroid heatmap, pairwise comparison

### Machine Learning
- **Supervised models** — Random Forest classifiers for Churn Risk & Customer Health prediction
- **Model evaluation** — Accuracy, Precision, Recall, F1, confusion matrices
- **Reproducible training** — fixed random_state=42, configurable test split

### Recommendation Engine
- **Hybrid scoring** — combines persona affinity with reward category fit
- **Top-3 ranking** — per-customer ranked recommendations with scores
- **Business insight generation** — contextual explanations for each recommendation

### New Customer Prediction
- **Single & batch prediction** — predict cluster, persona, similarity score for unseen customers
- **Distance-based similarity** — normalized centroid distance → 0–100 confidence score
- **Confidence levels** — "Excellent Match" through "Outlier" based on similarity thresholds

### LLM Personalization (Optional)
- **OpenAI integration** — generates natural-language customer messages, retention strategies, and upsell opportunities
- **Deterministic fallback** — template-based responses when no API key is configured
- **Response caching** — caches LLM results to avoid redundant API calls

### Interactive Dashboard
- **5-page Streamlit dashboard** with custom theme (navy/blue gradient)
- **Real-time KPIs** — total customers, clusters, recommendations, similarity scores
- **Interactive filtering** — filter recommendations by persona, score threshold, segment
- **Export capabilities** — download any data table as CSV
- **Animated visualizations** — animated bar charts, PCA scatter with play/pause controls

---

## 🛠️ Tech Stack

| Category | Library | Purpose |
|---|---|---|
| **Data Processing** | pandas, numpy | Data loading, cleaning, feature engineering |
| **Machine Learning** | scikit-learn | KMeans, Random Forest, preprocessing, evaluation |
| **Model Persistence** | joblib | Pipeline and model artifact serialization |
| **Excel Handling** | openpyxl | Excel workbook reading/writing |
| **Visualization** | plotly, matplotlib, seaborn | Dashboard charts & EDA plots |
| **Dashboard** | streamlit | Interactive web UI |
| **LLM (Optional)** | openai | Natural language personalization |
| **CLI** | argparse | Command-line interface |

---

## 📦 Installation

### Prerequisites
- Python **3.11+**
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Bhavagna27/Hyper_Personalized_Loyalty_Engine.git
cd Hyper_Personalized_Loyalty_Engine

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the full pipeline
python scripts/run_pipeline.py

# 6. Launch the dashboard
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
Hyper-Personalized-Loyalty-Engine/
├── data/
│   ├── Consultant_Loyalty_Dataset_200_Customers.xlsx    # Source workbook
│   ├── Consultant_Loyalty_Dataset_200_Customers_2.xlsx   # Secondary workbook
│   └── processed/                                        # Cleaned CSVs
│       ├── Transaction_History.csv
│       ├── Customer_Loyalty_Profile.csv
│       ├── AI_Recommendations.csv
│       └── customer_features.csv                         # Engineered features
├── artifacts/                                            # ML artifacts
│   ├── feature_pipeline.joblib                           # sklearn pipeline
│   ├── feature_metadata.json                             # Feature definitions
│   ├── kmeans_model.joblib                               # KMeans model
│   ├── model_bundle.joblib                               # Supervised models
│   ├── cluster_summary.json                              # Segmentation summary
│   └── scored_customers.csv                              # Scored output
├── outputs/
│   ├── clustering/                                       # Segmentation outputs
│   │   ├── cluster_profiles.csv
│   │   ├── cluster_statistics.csv
│   │   ├── cluster_summary.json
│   │   ├── kmeans_model.joblib
│   │   ├── *.png                                         # Visualization plots
│   │   └── cluster_distance_stats.json
│   ├── models/
│   │   └── model_evaluation.json                         # Model metrics
│   └── recommendations/
│       ├── recommendations.csv                           # Per-customer recs
│       └── reward_catalog.csv                            # Reward library
├── reports/
│   ├── validation_report.json                            # Data quality report
│   └── logs/                                             # Pipeline logs
├── src/loyalty_engine/                                   # Core package
│   ├── cli.py                                            # CLI entry point
│   ├── config.py                                         # Paths & config
│   ├── io/                                               # Data I/O
│   │   ├── excel_loader.py                               # Excel discovery
│   │   └── ingestion.py                                  # Ingestion pipeline
│   ├── validation/                                       # Data quality
│   │   ├── schema.py                                     # Column schemas
│   │   ├── rules.py                                      # Validation rules
│   │   └── report.py                                     # Report generation
│   ├── preprocessing/
│   │   └── cleaning.py                                   # Data cleaning
│   ├── features/
│   │   └── engineering.py                                # Feature engineering
│   ├── models/
│   │   ├── segmentation.py                               # KMeans clustering
│   │   ├── trainers.py                                   # Supervised models
│   │   └── artifacts.py                                  # Save/load artifacts
│   ├── recommendations/
│   │   └── engine.py                                     # Recommendation engine
│   ├── pipeline/
│   │   ├── train.py                                      # Training pipeline
│   │   ├── infer.py                                      # Scoring pipeline
│   │   ├── segmentation_pipeline.py                      # Segmentation pipeline
│   │   └── new_customer_prediction.py                    # New customer preds
│   ├── llm/
│   │   └── personalization.py                            # LLM personalization
│   ├── visualization/
│   │   ├── eda.py                                        # EDA plots
│   │   └── clustering_viz.py                             # Cluster viz
│   └── __init__.py
├── dashboard/                                            # Streamlit dashboard
│   ├── app.py                                            # Entry point
│   ├── core.py                                           # Page renderers
│   ├── components/
│   │   ├── charts.py                                     # Plotly charts
│   │   ├── data.py                                       # Data loaders
│   │   ├── metrics.py                                    # KPI cards
│   │   ├── sidebar.py                                    # Navigation
│   │   ├── tables.py                                     # Data tables
│   │   └── theme.py                                      # Styling & theme
│   └── pages/
│       ├── 1_Overview.py                                 # Overview page
│       ├── 2_Customer_Segmentation.py                    # Segmentation page
│       ├── 3_Recommendations.py                          # Recommendations page
│       ├── 4_New_Customer.py                             # New customer page
│       └── 5_Business_Analytics.py                       # Analytics page
├── scripts/
│   ├── run_pipeline.py                                   # Full pipeline runner
│   ├── explore.py                                        # EDA script
│   ├── train.py                                          # Training script
│   └── predict.py                                        # Prediction script
├── tests/
│   ├── test_recommendations.py                           # Recommendation tests
│   └── test_training.py                                  # Training tests
├── requirements.txt                                      # Dependencies
├── pyproject.toml                                        # Project metadata
└── README.md                                             # This file
```

---

## 🚀 Usage

### CLI Commands

The project exposes a unified CLI via `python -m loyalty_engine.cli`:

```bash
# 1️⃣ Ingest, validate, and clean data
python -m loyalty_engine.cli ingest \
  --input data/Consultant_Loyalty_Dataset_200_Customers.xlsx

# 2️⃣ Engineer features (generates 40+ features)
python -m loyalty_engine.cli features \
  --input data/Consultant_Loyalty_Dataset_200_Customers.xlsx

# 3️⃣ Customer segmentation (KMeans with auto K-selection)
python -m loyalty_engine.cli segment \
  --input data/Consultant_Loyalty_Dataset_200_Customers.xlsx

# 4️⃣ Train supervised models (Churn & Health classifiers)
python -m loyalty_engine.cli train \
  --input data/Consultant_Loyalty_Dataset_200_Customers.xlsx

# 5️⃣ Generate recommendations
python -m loyalty_engine.cli score \
  --input data/Consultant_Loyalty_Dataset_200_Customers.xlsx

# 6️⃣ EDA visualizations
python -m loyalty_engine.cli eda \
  --input data/Consultant_Loyalty_Dataset_200_Customers.xlsx
```

**One-command pipeline:**
```bash
python scripts/run_pipeline.py
```

### Python API

```python
from pathlib import Path
from loyalty_engine.io import IngestionPipeline
from loyalty_engine.features import FeatureEngineeringPipeline
from loyalty_engine.pipeline import (
    SegmentationPipeline,
    NewCustomerPredictor,
    run_training_pipeline,
)

# 1. Ingest data
ingest = IngestionPipeline(Path("data/workbook.xlsx")).run()

# 2. Engineer features
fe = FeatureEngineeringPipeline().run(
    profile=ingest.cleaned_frames["Customer_Loyalty_Profile"],
    transactions=ingest.cleaned_frames["Transaction_History"],
    save=True,
)

# 3. Segment customers
seg = SegmentationPipeline(
    input_path=Path("data/workbook.xlsx")
).run()

print(f"Optimal clusters: {seg.bundle.optimal_k}")
print(f"Personas: {seg.bundle.personas}")

# 4. Predict for a new customer
predictor = NewCustomerPredictor()
result = predictor.predict_customer({
    "Customer_ID": "NEW001",
    "Total_Spend_6M": 5000.0,
    "Total_Transactions_6M": 12,
    "Days_Since_Last_Purchase": 5,
    "Membership_Tier": "Gold",
    # ... other profile fields
})
print(f"Predicted cluster: {result['predicted_cluster']}")
print(f"Persona: {result['persona']}")
print(f"Similarity score: {result['similarity_score']}")
```

### Streamlit Dashboard

```bash
# Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard opens at **http://localhost:8501** with 5 pages:

| Page | Route | Purpose |
|---|---|---|
| **Overview** | `/` | Executive summary, KPIs, architecture diagram, pipeline flow |
| **Customer Segmentation** | `/Customer_Segmentation` | Cluster pies, PCA scatter, centroid radar, persona cards |
| **Recommendations** | `/Recommendations` | Filtered recommendations, score distribution, reward catalog |
| **New Customer** | `/New_Customer` | Cluster prediction, similarity gauge, business insights |
| **Business Analytics** | `/Business_Analytics` | Validation health, model metrics, operational artifacts |

---

## 🔬 Pipeline Stages

### 1. Data Ingestion & Validation

The ingestion layer runs four stages in sequence:

| Stage | Module | What Happens |
|---|---|---|
| **Load** | `io/excel_loader.py` | Auto-discovers worksheets; falls back to three standard sheets |
| **Validate** | `validation/schema.py`, `rules.py`, `report.py` | Structural + data-quality checks; produces `ValidationReport` |
| **Clean** | `preprocessing/cleaning.py` | Trim whitespace · drop duplicates · coerce types · median-fill |
| **Save** | `io/ingestion.py` | Writes `data/processed/<Sheet>.csv` + `reports/validation_report.json` |

### 2. Feature Engineering

Engineers **40+ features** across 6 groups:

| Group | Features |
|---|---|
| **RFM** | recency, frequency, monetary, avg order value, total spend, purchase frequency, days since last purchase |
| **Reward** | points earned, points redeemed, utilization %, coupon usage %, redemption rate, avg reward/transaction |
| **Engagement** | website visits, session duration, wishlist added, cart abandoned, email click rate, push response, app usage score, engagement score |
| **Shopping Behavior** | favorite category, favorite brand, top merchant, online/offline/weekend ratio, avg monthly spend, seasonality |
| **Customer Value** | CLV, estimated annual spend, profitability, health score, upgrade readiness, churn risk, loyalty score |
| **Derived** | spending diversity, category diversity, brand diversity, avg days between purchases, HVC/dormancy/premium flags |

Also saves a reusable `sklearn` preprocessing pipeline (`feature_pipeline.joblib`) and feature metadata (`feature_metadata.json`).

### 3. Customer Segmentation

Evaluates K from **2 to 10** using:

| Metric | Purpose |
|---|---|
| **Elbow Method** | Inertia vs. K — identifies diminishing returns |
| **Silhouette Score** | Cluster cohesion & separation |
| **Business Interpretability** | Custom metric ranking persona separation |

The best K is selected by combining these three signals, then **KMeans** (n_init=20, random_state=42) is trained and personified with data-driven names:

- **Premium Traveler** — High spend, engagement, and CLV with diverse purchase patterns
- **Luxury Lifestyle** — Top-tier spending, premium membership, low churn
- **Digital Explorer** — Digital-first, strong app engagement, online-heavy
- **Value Shopper** — Reward-driven, frequent purchases, value-conscious
- **Dormant Customer** — Low recent activity, needs reactivation
- **Loyal Cashback User** — Sustained frequency, high reward utilization

### 4. Model Training

Trains **Random Forest classifiers** for two target variables:

| Target | Purpose |
|---|---|
| **Churn_Risk** | Predict churn probability (High/Medium/Low) |
| **Customer_Health** | Predict health status (Healthy/Stable/Needs Attention/Critical) |

Evaluation metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.

### 5. Recommendation Engine

- **Score-based ranking** — scores rewards by persona affinity + category fit
- **Top-3 per customer** — highest-scoring rewards with normalized scores
- **Business insights** — contextual explanations for each recommendation

### 6. New Customer Prediction

For brand-new customer profiles:

1. **Feature engineering** — constructs feature vector from minimal input
2. **Cluster assignment** — finds nearest centroid via KMeans
3. **Similarity scoring** — normalizes centroid distance to 0–100 scale
4. **Confidence level** — maps similarity to "Excellent Match" → "Outlier"
5. **Recommendation** — generates top-3 rewards and business insights

### 7. LLM Personalization (Optional)

When an `OPENAI_API_KEY` is configured, the LLM module:

- Generates **customer-facing messages** in natural language
- Produces **retention strategies** and **upsell opportunities**
- Falls back to **deterministic templates** when no API key is available
- **Caches results** to minimize API costs

---

## 📊 Dashboard

The Streamlit dashboard provides a production-grade UI with:

### Theme
- Custom navy/blue gradient theme with `Segoe UI` font
- Glassmorphism cards, rounded corners, soft shadows
- Responsive layout with collapsible sidebar navigation

### Pages

**1. Overview** — Executive summary with:
- KPI row: Total Customers, Clusters, Recommendations, Avg Similarity, Avg Score
- Pipeline status indicators (Ready/Partial)
- Executive/Governance/Commercial tabs
- Architecture diagram, pipeline flow, validation snapshot
- Downloadable CSVs for all data tables

**2. Customer Segmentation** — Deep cluster analysis:
- Interactive pie chart with pull-to-focus
- Animated bar chart with play/pause controls
- Animated PCA scatter plot
- Radar centroid comparison with fill highlighting
- Per-cluster detail cards (spend, frequency, engagement, persona)

**3. Recommendations** — Filtered recommendation explorer:
- Filter by persona (multi-select) and minimum score threshold
- Sortable table with top-3 rewards per customer
- Score distribution histogram
- Reward catalog browser
- CSV export

**4. New Customer** — Prediction interface:
- Cluster distribution bar chart
- Similarity gauge indicator
- Per-customer inspection with business insights
- Confidence levels and expected impact

**5. Business Analytics** — Governance & operations:
- Validation health by sheet with drill-down
- Model performance metrics with confusion matrix
- Reward catalog by category
- Operational artifact inventory

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_recommendations.py -v
pytest tests/test_training.py -v
```

Tests cover:
- Recommendation engine scoring and ranking logic
- Model training and evaluation pipelines
- Data validation rules
- Feature engineering correctness

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow existing code style (ruff, mypy conventions)
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <strong>Built with ❤️ for hyper-personalized banking loyalty</strong><br/>
  <sub>© 2026 Hyper-Personalized Loyalty Engine</sub>
</div>
