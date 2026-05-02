# Adult Income Prediction Project (UCI Census Income)

## Project Overview
This project aims to build a predictive model to determine whether an individual's annual income exceeds $50,000 based on census data. This is a binary classification task using the well-known **UCI Census Income (Adult)** dataset.

**Project Status:** 
- Work Plans 1, 2, 3: Completed (EDA & Preprocessing)
- Work Plan 4: Completed (Baseline & Symbolic Models)
- Work Plan 6: Completed (Advanced Models & Ensemble & Interpretation)

## Project Structure
The project is organized into the following directory structure for better maintainability and clarity:

```
DATA_MINING_PROJECT/
├── data/
│   ├── raw/                # Original dataset (census_income_full.csv)
│   └── processed/          # Preprocessed train/val/test splits
├── notebooks/              # Jupyter Notebooks for different stages
│   ├── 01_import_dataset.ipynb
│   ├── 02_EDA_Preprocessing.ipynb
│   └── 03_Modelling.ipynb
├── scripts/                # Utility scripts and complete pipeline
│   └── run_pipeline.py     # End-to-end execution script
├── results/                # Visualizations, rules, and model summaries
├── docs/                   # Project planning and requirements
├── .gitignore              # Files to be ignored by Git
├── LICENSE.txt             # Project license
└── README.md               # Project documentation
```

## Implementation Summary

### 1. Data Exploration & Preprocessing (WP 1-3)
- **Data Acquisition**: Dataset fetched from UCI ML Repository.
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of demographics, income distributions, and feature correlations.
- **Data Cleaning**: Mode imputation for missing values and removal of duplicate rows.
- **Feature Engineering**: 
  - Created `capital_net` (Capital Gain - Capital Loss).
  - Categorized Education into tiers.
  - Dropped redundant features like `fnlwgt` and `education`.
- **Transformation**: Applied One-Hot Encoding for categorical features and Target Encoding for high-cardinality features (`native-country`). Standardized numeric features.

### 2. Baseline & Symbolic Models (WP 4)
- **Baseline**: Established a Majority-Class Baseline (76.1% accuracy).
- **Logistic Regression**: Analyzed feature coefficients to understand predictors for income levels.
- **Decision Tree**: Trained a interpretable tree with `max_depth=5` for rule-based prediction.

### 3. Advanced Models & Ensemble (WP 6)
- **Advanced Classifiers**: Trained and evaluated **SVM (RBF kernel)**, **Random Forest**, and **XGBoost**.
- **Ensemble Learning**: Built a **Soft Voting Ensemble** (Logistic Regression + Random Forest + XGBoost) for robust predictions.
- **Interpretation**: Extracted human-readable decision rules and analyzed feature importance (Gini/Gain).
- **Fairness Analysis**: Evaluated the best-performing model across protected attributes (Race and Sex) to ensure ethical and unbiased predictions.

## Key Results
The **XGBoost** model and the **Ensemble** model achieved the highest performance:
- **XGBoost F1-Score (Weighted)**: 0.8645
- **Ensemble ROC-AUC**: 0.9200

Detailed results and plots can be found in the `results/` directory.

## How to Run
1. Ensure all dependencies are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn category_encoders xgboost
   ```
2. To run the entire pipeline end-to-end:
   ```bash
   python scripts/run_pipeline.py
   ```
3. Or explore the Jupyter notebooks in the `notebooks/` folder sequentially.

---
**Team 04 - Data Mining Project**
