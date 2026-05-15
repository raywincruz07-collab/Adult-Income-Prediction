# Adult Income Prediction Project (UCI Census Income)

## Project Overview
This project aims to build a predictive model to determine whether an individual's annual income exceeds $50,000 based on census data. This is a binary classification task using the well known **UCI Census Income (Adult)** dataset.

## 📊 Live Results Dashboard (Tuned Models)

| Model | Accuracy | F1-Score (Weighted) | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 76.06% | 0.6572 | 0.5000 |
| **Logistic Regression (Tuned)** | 79.75% | 0.8096 | 0.8978 |
| **Decision Tree (Tuned)** | 82.06% | 0.8250 | 0.8700 |
| **Random Forest (Tuned)** | 84.19% | 0.8462 | 0.9108 |
| **XGBoost (Tuned)** | 84.07% | 0.8471 | 0.9238 |
| **SVM (RBF kernel)** | 80.58% | 0.8174 | 0.9004 |
| **Tuned Ensemble (Soft Voting)** | 84.16% | 0.8482 | 0.9215 |

> **Note**: Primary metric is **F1-Score (weighted)** due to class imbalance (76/24). The models have been systematically optimized using **GridSearchCV**. The **Tuned Ensemble** provides strong balanced performance.

## ⚖️ Fairness Analysis Summary (Tuned Ensemble)
Analyzed using the **Tuned Ensemble** across protected attributes:

| Protected Attribute | Group | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Sex** | Male | 79.75% | 0.8043 |
| **Sex** | Female | 92.89% | 0.9292 |
| **Race** | White | 83.35% | 0.8401 |
| **Race** | Black | 91.59% | 0.9180 |

*For full fairness details across all categories (including native-country), see `results_final/fairness_analysis.csv`.*

## 🛠️ Implementation Highlights

### 1. Data Exploration & Preprocessing
- **Cleaning**: Mode imputation for missing values in `workclass`, `occupation`, and `native-country`.
- **Feature Engineering**: Merged capital gain/loss into `capital_net`. Dropped `fnlwgt` and `education` (redundant).
- **Pipeline**: Applied `StandardScaler` for numeric and `TargetEncoder` for high-cardinality `native-country`.

### 2. Modeling & Evaluation
- **Baseline**: Majority-class dummy classifier as a benchmark.
- **Symbolic**: Logistic Regression coefficients analyzed and Decision Tree (max_depth=5) visualized.
- **Stability**: Performed **5-Fold Stratified Cross-Validation** on models, achieving strong performance metrics (e.g., Mean F1: ~0.855 for the Ensemble).

### 3. Advanced, Interpretation & Tuning
- **Advanced Classifiers**: Comparative analysis of SVM, Random Forest, and XGBoost.
- **Hyperparameter Tuning**: Systematically optimized models using `GridSearchCV` with class weighting (`balanced`) to handle class imbalance.
- **Interpretation**: Extracted human-readable decision rules and analyzed feature importance (XGBoost Gain vs. RF Gini).
- **Fairness**: Conducted audit across Sex and Race using the Tuned Ensemble model.

## 🚀 How to Run
1. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn category_encoders xgboost`
2. Run the end-to-end pipeline: `python scripts/run_pipeline.py`
3. Results are automatically saved to the `results/` folder (Baseline outputs) and `results_final/` folder (Tuned model outputs).

---
**Team 04 - Data Mining Project - IE500**
