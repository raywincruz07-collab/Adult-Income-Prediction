# Adult Income Prediction

> An end-to-end binary-classification study predicting whether annual income exceeds $50,000 using the UCI Adult dataset, with model comparison, hyperparameter tuning, interpretation and fairness analysis.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Verified Results](#verified-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [My Contributions](#my-contributions)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [License](#license)
- [Project Origin](#project-origin)

## Project Overview

**Problem Statement:** Income inequality and bias in predictive models are critical concerns in machine learning. This project analyzes the UCI Adult dataset to predict income brackets while rigorously evaluating models for predictive performance and fairness.

**Solution:** A complete machine-learning pipeline comparing baseline classifiers against tuned algorithms (Logistic Regression, Decision Trees, SVM, Random Forest, XGBoost, and a Soft-voting Ensemble), including cross-validation and fairness auditing.

## Architecture

```mermaid
flowchart LR
    A[UCI Adult data] --> B[Cleaning and feature engineering]
    B --> C[Encoding and scaling]
    C --> D[Baseline and candidate models]
    D --> E[GridSearchCV]
    E --> F[Performance evaluation]
    F --> G[Interpretation and fairness audit]
```

## Methodology

- **Preprocessing:** Handling missing values, engineering features, and target encoding.
- **Model Comparison:** Evaluated multiple models including SVM, Random Forest, and XGBoost.
- **Hyperparameter Tuning:** Systematically optimized using GridSearchCV.
- **Fairness Analysis:** Conducted audits across sensitive attributes (Sex, Race, Native-Country). Group-level performance differences require careful interpretation and do not automatically prove the model is completely unbiased.

## Verified Results

Extracted from `results_final/model_comparison.csv`.

| Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 84.19% | 0.8462 | 0.9108 |
| XGBoost | 84.07% | 0.8471 | 0.9238 |
| Soft Voting Ensemble | 84.16% | 0.8482 | 0.9215 |

## Repository Structure

- `data/`: Raw and processed datasets.
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, and modeling.
- `scripts/`: Python scripts for running the end-to-end pipeline.
- `results_final/`: Final authoritative visualizations, CSVs, and classification reports.

## Getting Started

To reproduce the pipeline:

1. Ensure Python 3.x is installed.
2. Install the verified dependencies:
   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn category_encoders xgboost
   ```
3. Run the end-to-end workflow, which reads from `data/raw/census_income_full.csv` and outputs to `results_final/`:
   ```bash
   python scripts/run_pipeline.py
   ```
4. Alternatively, open and execute the notebooks in the `notebooks/` directory sequentially.

## My Contributions

My contribution to this project included:
- Setting up the professional project structure and end-to-end modeling pipeline.
- Configuring 5-fold cross-validation and generating the native-country fairness analysis.
- Enhancing result visibility by producing full classification reports.

## Limitations & Future Improvements
- Further analysis on intersectional fairness metrics.
- Deployment of the final ensemble model via a REST API.

## License

See the [LICENSE.txt](LICENSE.txt) file for licensing information.

## Project Origin

This repository is maintained as a personal fork of the original project.

- [Original repository](https://github.com/hazarcs/IE500-DataMining-Project)
- [Raywin Cruz – GitHub](https://github.com/raywincruz07-collab)
