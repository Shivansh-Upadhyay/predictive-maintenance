<div align="center">

# ⚙️ PREDICTMAINT

### Industrial Equipment Failure Prediction System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-0B3D91?style=for-the-badge)](https://ti.arc.nasa.gov)

> **Predict industrial equipment failure before it happens —
> giving maintenance teams up to 9 cycles of advance warning
> to prevent costly unplanned downtime.**

</div>

---

## 🔴 The Problem

Unplanned equipment failure in oil & gas operations costs companies
**lakhs of rupees per hour** in lost production. Traditional maintenance
approaches fail in two ways:

```
❌ Fixed schedule maintenance  →  wasteful, replaces healthy components
❌ Reactive maintenance        →  dangerous, production halts unexpectedly
```

**PredictMaint solves this with data** — reading sensor signals to
predict exactly when a machine needs attention, before it fails.

---

## ✅ The Solution

A full end-to-end ML pipeline that:

- Ingests raw sensor data from industrial equipment
- Engineers 67+ predictive features from time-series readings
- Trains and compares Random Forest vs XGBoost models
- Deploys as an interactive real-time monitoring dashboard
- Monitors entire fleets of equipment simultaneously

---

## 📊 Model Performance

Trained and evaluated across **all 4 NASA CMAPSS fault scenarios**:

| Dataset | Fault Modes | Op Conditions | Best Model    | RMSE            | R²         |
| ------- | ----------- | ------------- | ------------- | --------------- | ---------- |
| FD001   | 1           | 1 (fixed)     | XGBoost       | **7.08 cycles** | **0.9704** |
| FD002   | 1           | 6 (variable)  | Random Forest | **8.93 cycles** | **0.9538** |
| FD003   | 2           | 1 (fixed)     | XGBoost       | **7.14 cycles** | **0.9687** |
| FD004   | 2           | 6 (variable)  | Random Forest | **8.65 cycles** | **0.9548** |

> **All 4 scenarios achieve R² above 0.95** — model explains
> 95%+ of all variance in remaining equipment life.

**Key Finding:**

> XGBoost outperformed Random Forest on single-condition data
> (FD001, FD003), while Random Forest proved more stable under
> variable operating conditions (FD002, FD004). This demonstrates
> why model selection must always be data-driven, not assumption-based.

---

## 🏗️ Project Architecture

```
raw sensor data (NASA CMAPSS)
         │
         ▼
┌─────────────────┐
│  fetch_data.py  │  →  loads txt files, assigns RUL labels,
└─────────────────┘      saves train_processed_FD00X.csv
         │
         ▼
┌──────────────────────────┐
│ feature_engineering.py   │  →  rolling stats, cycle features,
└──────────────────────────┘     condition clustering (FD002/FD004)
         │
         ▼
┌────────────────┐
│   train.py     │  →  RF vs XGBoost comparison,
└────────────────┘     saves best_model_FD00X.pkl
         │
         ▼
┌────────────────┐
│    app.py      │  →  Streamlit dashboard,
└────────────────┘     real-time RUL prediction
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Mac/Linux terminal
- NASA CMAPSS dataset (instructions below)

### 1 — Clone The Repository

```bash
git clone https://github.com/YOURUSERNAME/predictive-maintenance.git
cd predictive-maintenance
```

### 2 — Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Download Dataset

1. Go to 👉 [NASA CMAPSS on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
2. Download and unzip
3. Move all `.txt` files into the `data/` folder

```
data/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

### 5 — Run The Full Pipeline

```bash
# Process all 4 datasets
python fetch_data.py FD001
python fetch_data.py FD002
python fetch_data.py FD003
python fetch_data.py FD004

# Feature engineering
python feature_engineering.py FD001
python feature_engineering.py FD002
python feature_engineering.py FD003
python feature_engineering.py FD004

# Train models
python train.py FD001
python train.py FD002
python train.py FD003
python train.py FD004

# Launch dashboard
streamlit run app.py
```

### 6 — Open Dashboard

```
http://localhost:8501
```

---

## 📁 Project Structure

```
predictive-maintenance/
├── data/
│   ├── train_FD00X.txt          ← raw NASA sensor data
│   ├── train_processed_FD00X.csv ← RUL labelled data
│   └── train_featured_FD00X.csv  ← engineered features
│
├── models/
│   ├── best_model_FD00X.pkl      ← trained models
│   ├── feature_cols_FD00X.pkl    ← feature lists
│   ├── kmeans_FD002.pkl          ← condition clustering
│   ├── kmeans_FD004.pkl          ← condition clustering
│   └── model_comparison_FD00X.csv ← RF vs XGBoost results
│
├── fetch_data.py                 ← data loading + RUL labelling
├── feature_engineering.py       ← feature engineering pipeline
├── train.py                     ← model training + comparison
├── app.py                       ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🔬 Feature Engineering

Raw sensor values alone are weak predictors. We engineer
**67–68 features** from 26 original columns:

| Feature Type             | Count | What It Captures          |
| ------------------------ | ----- | ------------------------- |
| Rolling mean (10 cycles) | 14    | Sensor degradation trend  |
| Rolling std (10 cycles)  | 14    | Sensor instability        |
| Sensor diff              | 14    | Rate of change per cycle  |
| Cycle normalized         | 1     | Engine age (0.0 → 1.0)    |
| Op cluster (FD002/FD004) | 1     | Operating condition group |
| Raw sensors              | 14    | Baseline readings         |

**Key insight:** `cycle_normalized` contributed 89–95% of
feature importance across all datasets — confirming that
engine age is the dominant predictor of remaining useful life.
Sensor features capture the critical margin cases where two
engines of the same age have very different failure trajectories.

---

## 🖥️ Dashboard Features

```
⚙️  Single Engine Monitor   →  RUL gauge + live status
📈  RUL Timeline            →  full degradation curve cycle 1→400
🔬  Sensor Trends           →  4 key sensors plotted over lifetime
🏆  Model Comparison        →  RF vs XGBoost across RMSE/MAE/R²
📊  Cross-Dataset Chart     →  all 4 scenarios side by side
🏭  Fleet Monitor           →  10 engines simultaneously
```

---

## 🌍 Real-World Applicability

This system was built at **RGIPT (Rajiv Gandhi Institute of
Petroleum Technology)** — India's premier petroleum university.

While trained on NASA CMAPSS turbofan data, the sensor
degradation patterns are directly analogous to equipment
used in Indian oil & gas operations:

```
NASA turbofan sensors    ≈    ONGC compressor sensors
Temperature rise pattern ≈    Pump bearing degradation
Pressure drop signature  ≈    Valve wear pattern
```

**Future integration path:**

> Replace `fetch_data.py` with a live ONGC sensor feed —
> all downstream models, features, and dashboard remain
> identical. The architecture was designed for this swap
> from day one.

---

## 🔭 Future Scope

- [ ] Integrate real ONGC/BPCL sensor data via API
- [ ] Live streaming sensor feed (Apache Kafka)
- [ ] SMS/email alerts when RUL drops below threshold
- [ ] Multi-plant fleet monitoring across locations
- [ ] Anomaly detection for sudden sensor spikes
- [ ] FD003/FD004 fault type classification (which fault?)
- [ ] Mobile-responsive dashboard

---

## 🛠️ Tech Stack

| Layer               | Technology                            |
| ------------------- | ------------------------------------- |
| Language            | Python 3.12                           |
| ML Models           | XGBoost, Random Forest (scikit-learn) |
| Feature Engineering | Pandas, NumPy                         |
| Clustering          | KMeans (scikit-learn)                 |
| Visualisation       | Plotly, Streamlit                     |
| Model Persistence   | Joblib                                |
| Dataset             | NASA CMAPSS (160,359 rows)            |

---

## 📦 Requirements

Create `requirements.txt` with:

```
pandas
numpy
scikit-learn
xgboost
streamlit
plotly
joblib
```

---

## 👤 About

Built by a **Computer Science student at RGIPT
(Rajiv Gandhi Institute of Petroleum Technology)**
— bridging software engineering with the energy
domain to solve real industrial problems.

> _"The architecture is intentionally modular —
> swap the data source, keep everything else.
> Built for NASA benchmarks today,
> ready for ONGC data tomorrow."_

---

<div align="center">

**⚙️ PredictMaint** &nbsp;|&nbsp;
CS @ RGIPT &nbsp;|&nbsp;
NASA CMAPSS &nbsp;|&nbsp;
XGBoost + Streamlit

</div>
