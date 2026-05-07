# Adult Income Prediction Project (UCI Census Income)

## Project Overview
This project aims to build a predictive model to determine whether an individual's annual income exceeds $50,000 based on census data. This is a binary classification task using the well-known **UCI Census Income (Adult)** dataset.

**Project Status:** 
- Work Plans 1-3: ✅ **COMPLETED** (EDA & Preprocessing)
- Work Plan 4: ✅ **COMPLETED** (Baseline & Symbolic Models)
- Work Plan 5: ✅ **COMPLETED** (Cross-Validation)
- Work Plan 6: ✅ **COMPLETED** (Advanced Models, Ensemble & Fairness)

## 📊 Live Results Dashboard (WP 4 & 6)

| Model | Accuracy | F1-Score (Weighted) | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 76.06% | 0.6572 | 0.5000 |
| **Logistic Regression** | 84.49% | 0.8394 | 0.8986 |
| **Decision Tree (depth=5)** | 84.96% | 0.8421 | 0.8863 |
| **SVM (RBF kernel)** | 85.58% | 0.8494 | 0.8953 |
| **Random Forest (n=200)** | 84.76% | 0.8439 | 0.8936 |
| **XGBoost (n=300)** | **86.69%** | **0.8645** | **0.9230** |
| **Soft Voting Ensemble** | 86.49% | 0.8611 | 0.9200 |

> **Note**: Primary metric is **F1-Score (weighted)** due to class imbalance (76/24). **XGBoost** is the best performing model.

## ⚖️ Fairness Analysis Summary (WP 6)
Analyzed using the **Soft Voting Ensemble** across protected attributes:

| Protected Attribute | Group | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Sex** | Male | 83.08% | 0.8278 |
| **Sex** | Female | 93.24% | 0.9262 |
| **Race** | White | 85.76% | 0.8538 |
| **Race** | Black | 92.49% | 0.9186 |

*For full fairness details across all categories (including native-country), see `results/fairness_analysis.csv`.*

## 📁 Project Structure
```
DATA_MINING_PROJECT/
├── data/
│   ├── raw/                # Original dataset (census_income_full.csv)
│   └── processed/          # Preprocessed train/val/test CSV splits
├── notebooks/              # Jupyter Notebooks (Ordered 01-03)
│   ├── 01_import_dataset.ipynb
│   ├── 02_EDA_Preprocessing.ipynb
│   └── 03_Modelling.ipynb
├── scripts/                # Utility scripts
│   └── run_pipeline.py     # COMPLETE End-to-End Execution Script
├── results/                # Visualizations, rules, and model summaries
├── docs/                   # Project planning (Work Plan PDF)
├── .gitignore              # Git configuration
├── LICENSE.txt             # Project license
└── README.md               # Updated Project Documentation
```

## 🛠️ Implementation Highlights

### 1. Data Exploration & Preprocessing (WP 1-3)
- **Cleaning**: Mode imputation for missing values in `workclass`, `occupation`, and `native-country`.
- **Feature Engineering**: Merged capital gain/loss into `capital_net`. Dropped `fnlwgt` and `education` (redundant).
- **Pipeline**: Applied `StandardScaler` for numeric and `TargetEncoder` for high-cardinality `native-country`.

### 2. Modeling & Evaluation (WP 4 & 5)
- **Baseline**: Majority-class dummy classifier as a benchmark.
- **Symbolic**: Logistic Regression coefficients analyzed and Decision Tree (max_depth=5) visualized.
- **Stability**: Performed **5-Fold Stratified Cross-Validation** on XGBoost (Mean F1: 0.864).

### 3. Advanced & Interpretation (WP 6)
- **Advanced Classifiers**: Comparative analysis of SVM, Random Forest, and Gradient Boosting.
- **Interpretation**: Extracted human-readable decision rules and analyzed feature importance (XGBoost Gain vs. RF Gini).
- **Fairness**: Conducted audit across Sex, Race, and Native-Country.

## 🚀 How to Run
1. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn category_encoders xgboost`
2. Run the end-to-end pipeline: `python scripts/run_pipeline.py`
3. Results are automatically saved to the `results/` folder.

---
**Team 04 - Data Mining Project - IE500**
