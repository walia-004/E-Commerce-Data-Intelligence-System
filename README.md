# 🛒 E-Commerce Data Intelligence System

> Predicting customer purchase behavior through end-to-end data mining — from raw behavioral logs to production-ready ML models.

---

## 📌 Project Overview

**ShopSmart Analytics** is a full-cycle data science project built on a real-world-style e-commerce dataset of **16,000 user sessions**. The goal is to uncover hidden patterns in shopper behavior and build accurate models that predict whether a user will make a purchase — enabling smarter marketing, personalization, and business decisions.

This project was developed as part of **INFO911: Data Mining and Knowledge Discovery**.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-green?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 📂 Repository Structure

```
shopsmart-analytics/
│
├── INFO911_project.ipynb            # Main Jupyter Notebook (all sections)
├── final_ecommerce_dataset_16000.csv # Dataset (16,000 user records)
└── README.md
```

---

## 📊 Dataset

| Property        | Details                          |
|----------------|----------------------------------|
| Records         | 16,000 user sessions             |
| Features        | 14 columns                       |
| Target Variable | `purchase` (binary: 0/1)         |
| Source          | Synthetic e-commerce behavior data |

### Key Features

| Column | Description |
|--------|-------------|
| `age` | User age (18–59) |
| `gender` | Male / Female |
| `device_type` | Mobile / Desktop / Tablet |
| `time_on_site` | Total minutes spent on site |
| `pages_viewed` | Pages viewed per session |
| `previous_purchases` | Past purchase count |
| `cart_items` | Items added to cart |
| `discount_seen` | Whether a discount was displayed (0/1) |
| `ad_clicked` | Whether an ad was clicked (0/1) |
| `returning_user` | First-time vs returning visitor (0/1) |
| `avg_session_time` | Average session duration (minutes) |
| `bounce_rate` | % of sessions with no interaction |
| `purchase` | **Target**: Did user purchase? (0/1) |

---

## 🔬 Project Pipeline

### Section 1 — Setup & Imports
- Core libraries: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- ML libraries: `Scikit-learn` (preprocessing, clustering, classification, regression)

### Section 2 — Data Understanding & Preprocessing
- Missing value analysis and imputation (median for numeric, mode for categorical)
- Outlier detection using the **IQR method** with boxplot visualization
- **Feature engineering:**
  - `engagement_score` — weighted composite of time, pages, and bounce rate
  - `age_group` — binned age categories (18–25, 26–35, 36–45, 46–59)
  - `high_value_user` — flag for top-quartile purchasers
  - `interaction_score` — combined ad click + discount seen signal
- Label encoding (gender) and one-hot encoding (device type)
- Feature scaling with `StandardScaler` and `MinMaxScaler`

### Section 3 — Exploratory Data Analysis (EDA)
- Distribution plots for all key features
- Purchase rate breakdowns by gender, device type, and age group
- Correlation heatmap and cross-feature analysis
- Behavioral pattern discovery across returning vs. new users

### Section 4 — Clustering Analysis
Three unsupervised algorithms applied to uncover natural customer segments:

| Algorithm | Purpose |
|-----------|---------|
| **K-Means** | Partition-based clustering with Elbow method for optimal K |
| **Agglomerative Clustering** | Hierarchical clustering with dendrogram visualization |
| **DBSCAN** | Density-based clustering for irregular/noise-aware segments |

Evaluated using **Silhouette Score**.

### Section 5 — Classification & Prediction Models
Four supervised models trained to predict `purchase`:

| Model | Type |
|-------|------|
| **Decision Tree** | Interpretable, rule-based classifier |
| **Random Forest** | Ensemble of decision trees (bagging) |
| **Support Vector Machine (SVM)** | Margin-maximizing classifier |
| **MLP Neural Network** | Multi-layer perceptron (deep learning lite) |

### Section 6 — Model Evaluation & Overfitting Analysis
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices for all models
- ROC curve comparison across classifiers
- Cross-validation with `StratifiedKFold`
- Overfitting analysis: train vs. test performance gap

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Run the Notebook
```bash
git clone https://github.com/walia-004/E-Commerce-Data-Intelligence-System.git
cd shopsmart-analytics
jupyter notebook Data_mining.ipynb
```

> Make sure `final_ecommerce_dataset_16000.csv` is in the same directory as the notebook, or update the file path in **Section 2, Step 1**.

---

## 📈 Key Results

### 🏆 Comprehensive Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree | 0.8703 | 0.9601 | 0.8727 | 0.9143 | 0.8925 |
| SVM (RBF) | 0.8775 | 0.9597 | 0.8825 | 0.9195 | 0.8925 |
| Logistic Regression | 0.8519 | **0.9615** | 0.8471 | 0.9007 | **0.8993** |

### 🥇 Best Performers per Metric

| Metric | Best Model | Score |
|--------|-----------|-------|
| 🎯 Accuracy | **SVM (RBF)** | 0.8775 |
| 🔍 Precision | **Logistic Regression** | 0.9615 |
| 📡 Recall | **SVM (RBF)** | 0.8825 |
| ⚖️ F1-Score | **SVM (RBF)** | 0.9195 |
| 📈 ROC-AUC | **Logistic Regression** | 0.8993 |

> **Overall winner: SVM (RBF)** — leads in Accuracy, Recall, and F1-Score, making it the most balanced classifier for purchase prediction.

### Other Highlights
- Discovered **customer segments** using K-Means, Agglomerative, and DBSCAN clustering
- Engineered **4 new behavioral features** improving model signal
- Identified `cart_items`, `previous_purchases`, and `engagement_score` as top purchase predictors

---
