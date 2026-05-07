"""
=============================================================================
Adult Income Prediction — Complete Pipeline
Work Plans 4 & 6: Baseline, Symbolic, Advanced Models & Interpretation
=============================================================================
This script:
1. Loads raw data and preprocesses it (matching EDA_Preprocessing.ipynb)
2. Saves preprocessed train/val/test splits to data/
3. Trains & evaluates all models required by Work Plans 4 & 6
4. Saves all results and plots to results/
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, cross_validate
from xgboost import XGBClassifier
import category_encoders as ce

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12, "axes.labelsize": 10})

# Create output directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Initialize classification reports file
with open("results/classification_reports.txt", "w") as f:
    f.write("ADULT INCOME PREDICTION - DETAILED CLASSIFICATION REPORTS\n")
    f.write("========================================================\n")

all_results = []
plot_counter = [0]

def save_fig(name):
    plot_counter[0] += 1
    path = f"results/{plot_counter[0]:02d}_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")
    auc  = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}  << PRIMARY")
    print(f"  ROC-AUC  : {auc:.4f}")
    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    print(f"\nClassification Report:")
    print(report)

    # Save detailed report to file
    with open("results/classification_reports.txt", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"MODEL: {model_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc:.4f}\n")
        f.write(f"\nDetailed Classification Report:\n")
        f.write(report)
        f.write("\n\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["<=50K", ">50K"],
        cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{model_name} - Confusion Matrix")
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f"{model_name} - ROC Curve")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    save_fig(safe_name)

    result = {"Model": model_name, "Accuracy": acc, "Precision": prec,
              "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}
    all_results.append(result)
    return result

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING (reproducing EDA_Preprocessing.ipynb)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  STEP 1: PREPROCESSING")
print("=" * 70)

df = pd.read_csv("data/raw/census_income_full.csv")
print(f"Loaded data: {df.shape}")

# Clean target and missing values
df["income"] = df["income"].str.strip().str.rstrip(".")
df.replace("?", np.nan, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)
print(f"After dedup: {df.shape}")

# Drop fnlwgt
df.drop(columns=["fnlwgt"], inplace=True)

# Mode imputation
for col in ["workclass", "occupation", "native-country"]:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"  Filled '{col}' with mode = '{mode_val}'")

# Encode target to numeric
df["income"] = (df["income"] == ">50K").astype(int)

# Feature engineering: capital_net
df["capital_net"] = df["capital-gain"] - df["capital-loss"]
df.drop(columns=["capital-gain", "capital-loss"], inplace=True)

# Drop education (keep education-num)
df.drop(columns=["education"], inplace=True)

# Define feature groups
numeric_features = ["age", "education-num", "hours-per-week", "capital_net"]
ohe_features = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]
target_enc_features = ["native-country"]
target = "income"

# Train/Val/Test split
X = df.drop(columns=[target])
y = df[target]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.333, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval)

print(f"\nSplit sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

# StandardScaler
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val[numeric_features] = scaler.transform(X_val[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Target Encoding
te = ce.TargetEncoder(cols=target_enc_features)
X_train[target_enc_features] = te.fit_transform(X_train[target_enc_features], y_train)
X_val[target_enc_features] = te.transform(X_val[target_enc_features])
X_test[target_enc_features] = te.transform(X_test[target_enc_features])

# One-Hot Encoding
X_train = pd.get_dummies(X_train, columns=ohe_features, drop_first=True)
X_val = pd.get_dummies(X_val, columns=ohe_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=ohe_features, drop_first=True)
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"Final feature matrix: {X_train.shape}")

# Save preprocessed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_val.to_csv("data/processed/X_val.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False, header=True)
y_val.to_csv("data/processed/y_val.csv", index=False, header=True)
y_test.to_csv("data/processed/y_test.csv", index=False, header=True)
print("Preprocessed data saved to data/processed/")

# Convert to numpy for sklearn
y_train = y_train.values
y_val = y_val.values
y_test = y_test.values

# ═══════════════════════════════════════════════════════════════════════════
# WORK PLAN 4: Baseline & Symbolic Models
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WORK PLAN 4: BASELINE & SYMBOLIC MODELS")
print("=" * 70)

# ── Majority-Class Baseline ──
print("\n--- Majority-Class Baseline ---")
baseline = DummyClassifier(strategy="most_frequent", random_state=42)
baseline.fit(X_train, y_train)
evaluate_model(baseline, X_test, y_test, "Majority-Class Baseline")

# ── Logistic Regression ──
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

# Coefficient Analysis
coeffs = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lr_model.coef_[0]
}).sort_values("Coefficient", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#DD8452" if c > 0 else "#4C72B0" for c in coeffs["Coefficient"]]
ax.barh(coeffs["Feature"], coeffs["Coefficient"], color=colors, edgecolor="white")
ax.set_title("Logistic Regression - Feature Coefficients", fontsize=14, fontweight="bold")
ax.set_xlabel("Coefficient Value")
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
save_fig("LR_coefficients")

print("\nTop 5 features pushing towards >50K:")
print(coeffs.head(5).to_string(index=False))
print("\nTop 5 features pushing towards <=50K:")
print(coeffs.tail(5).to_string(index=False))

# ── Decision Tree ──
print("\n--- Decision Tree (max_depth=5) ---")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
evaluate_model(dt_model, X_test, y_test, "Decision Tree (depth=5)")

# Tree Visualization
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(dt_model, feature_names=X_train.columns.tolist(),
          class_names=["<=50K", ">50K"], filled=True, rounded=True,
          fontsize=8, ax=ax)
ax.set_title("Decision Tree Visualization (max_depth=5)", fontsize=16, fontweight="bold")
plt.tight_layout()
save_fig("DT_visualization")

# WP4 Summary
wp4_df = pd.DataFrame(all_results).set_index("Model")
print("\n" + "=" * 70)
print("  WORK PLAN 4 SUMMARY")
print("=" * 70)
print(wp4_df.round(4).to_string())

# ═══════════════════════════════════════════════════════════════════════════
# WORK PLAN 6: Advanced Models, Ensemble & Interpretation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WORK PLAN 6: ADVANCED MODELS, ENSEMBLE & INTERPRETATION")
print("=" * 70)

# ── SVM ──
print("\n--- SVM (RBF kernel) --- [This may take a few minutes]")
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, X_test, y_test, "SVM (RBF kernel)")

# ── Random Forest ──
print("\n--- Random Forest (n=200) ---")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test, "Random Forest (n=200)")

# ── XGBoost ──
print("\n--- XGBoost (n=300) ---")
xgb_model = XGBClassifier(
    n_estimators=300, random_state=42,
    use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
xgb_model.fit(X_train, y_train)
evaluate_model(xgb_model, X_test, y_test, "XGBoost (n=300)")

# ── Soft Voting Ensemble ──
print("\n--- Soft Voting Ensemble (LR+RF+XGB) ---")
ensemble_model = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=300, random_state=42,
                              use_label_encoder=False, eval_metric="logloss", n_jobs=-1))
    ],
    voting="soft"
)
ensemble_model.fit(X_train, y_train)
evaluate_model(ensemble_model, X_test, y_test, "Soft Voting Ensemble")

# ── Cross-Validation (Item 5 in Work Plan) ──
print("\n--- 5-Fold Stratified Cross-Validation (XGBoost) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring="f1_weighted")
print(f"  CV F1-Scores (Weighted): {cv_scores}")
print(f"  Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
with open("results/classification_reports.txt", "a") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"5-FOLD STRATIFIED CROSS-VALIDATION (XGBoost)\n")
    f.write(f"{'='*60}\n")
    f.write(f"CV F1-Scores: {cv_scores}\n")
    f.write(f"Mean F1: {cv_scores.mean():.4f}\n\n")

# ── Feature Importance ──
print("\n--- Feature Importance Analysis ---")
rf_imp = pd.DataFrame({
    "Feature": X_train.columns, "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

xgb_imp = pd.DataFrame({
    "Feature": X_train.columns, "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes[0].barh(rf_imp["Feature"], rf_imp["Importance"], color="#4C72B0", edgecolor="white")
axes[0].set_title("Random Forest - Gini Importance", fontsize=13, fontweight="bold")
axes[0].invert_yaxis()
axes[1].barh(xgb_imp["Feature"], xgb_imp["Importance"], color="#DD8452", edgecolor="white")
axes[1].set_title("XGBoost - Gain Importance", fontsize=13, fontweight="bold")
axes[1].invert_yaxis()
plt.tight_layout()
save_fig("feature_importance")

print("\nTop 5 Features (Random Forest):")
print(rf_imp.head(5).to_string(index=False))
print("\nTop 5 Features (XGBoost):")
print(xgb_imp.head(5).to_string(index=False))

# ── Decision Tree Rules ──
print("\n--- Decision Tree Rule Extraction ---")
tree_rules = export_text(dt_model, feature_names=X_train.columns.tolist())
print(tree_rules[:2000])  # Print first 2000 chars
with open("results/decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)
print("  Full rules saved to results/decision_tree_rules.txt")

# ── Fairness Analysis ──
print("\n--- Fairness Analysis ---")
fair_cols = [c for c in X_test.columns if c.startswith("sex_") or c.startswith("race_") or c == "native-country"]
print(f"Protected attribute columns: {fair_cols}")

best_model = ensemble_model
y_pred = best_model.predict(X_test)

X_test_df = pd.DataFrame(X_test, columns=X_train.columns)
fairness_rows = []
for col in fair_cols:
    for val in [True, False]:
        mask = X_test_df[col] == val
        if mask.sum() < 10:
            continue
        g_f1 = f1_score(y_test[mask], y_pred[mask], average="weighted", zero_division=0)
        g_acc = accuracy_score(y_test[mask], y_pred[mask])
        fairness_rows.append({
            "Group": f"{col}={val}", "Size": int(mask.sum()),
            "Accuracy": round(g_acc, 4), "F1-Score": round(g_f1, 4)
        })

fairness_df = pd.DataFrame(fairness_rows)
print("\nFairness Results (Soft Voting Ensemble):")
print(fairness_df.to_string(index=False))
fairness_df.to_csv("results/fairness_analysis.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# FINAL COMBINED COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL MODEL COMPARISON (Work Plans 4 + 6)")
print("=" * 70)

final_df = pd.DataFrame(all_results).set_index("Model")
print(final_df.round(4).to_string())
final_df.to_csv("results/model_comparison.csv")

fig, ax = plt.subplots(figsize=(12, 6))
final_df[["F1-Score", "ROC-AUC"]].plot(kind="bar", ax=ax, edgecolor="white")
ax.set_title("Model Comparison: F1-Score & ROC-AUC", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
save_fig("final_comparison")

print("\n" + "=" * 70)
print("  STEP 1-6 DONE! Check results/ directory for plots and CSVs.")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# NEW SECTION: HPM TUNING & MODELLING (reproducing 04_HPM_Tuning_Modelling)
# All outputs go to results_final/
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  HPM TUNING & MODELLING — STARTING")
print("=" * 70)

# ── 3.1 Setup ─────────────────────────────────────────────────────────────
os.makedirs("results_final", exist_ok=True)

with open("results_final/classification_reports.txt", "w") as f:
    f.write("ADULT INCOME PREDICTION - HPM TUNING CLASSIFICATION REPORTS\n")
    f.write("=" * 60 + "\n")

all_results_final = []
plot_counter_final = [0]


def save_fig_final(name):
    plot_counter_final[0] += 1
    path = f"results_final/{plot_counter_final[0]:02d}_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def evaluate_model_final(model, X_test_data, y_test_data, model_name="Model"):
    y_pred = model.predict(X_test_data)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_data)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test_data)
    else:
        y_proba = y_pred

    acc  = accuracy_score(y_test_data, y_pred)
    prec = precision_score(y_test_data, y_pred, average="weighted")
    rec  = recall_score(y_test_data, y_pred, average="weighted")
    f1   = f1_score(y_test_data, y_pred, average="weighted")
    auc  = roc_auc_score(y_test_data, y_proba)

    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}  << PRIMARY")
    print(f"  ROC-AUC  : {auc:.4f}")
    report = classification_report(y_test_data, y_pred, target_names=["<=50K", ">50K"])
    print(f"\nClassification Report:")
    print(report)

    with open("results_final/classification_reports.txt", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"MODEL: {model_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1-Score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc:.4f}\n")
        f.write(f"\nDetailed Classification Report:\n")
        f.write(report)
        f.write("\n\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test_data, y_pred, display_labels=["<=50K", ">50K"],
        cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{model_name} - Confusion Matrix")
    RocCurveDisplay.from_predictions(y_test_data, y_proba, ax=axes[1])
    axes[1].set_title(f"{model_name} - ROC Curve")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    save_fig_final(safe_name)

    result = {"Model": model_name, "Accuracy": acc, "Precision": prec,
              "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}
    all_results_final.append(result)
    return result


# ── 3.2 5-Fold Cross-Validation ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()


def run_cv(model, X, y, model_name):
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=[
            "f1_weighted",
            "roc_auc",
            "accuracy",
            "precision_weighted",
            "recall_weighted"
        ],
        n_jobs=-1
    )
    print(f"\n{'='*90}")
    print(f"  CV Results: {model_name}")
    print(f"{'='*90}")
    print(f"  Accuracy  : {scores['test_accuracy'].mean():.4f}  +/- {scores['test_accuracy'].std():.4f}")
    print(f"  Precision : {scores['test_precision_weighted'].mean():.4f}  +/- {scores['test_precision_weighted'].std():.4f}")
    print(f"  Recall    : {scores['test_recall_weighted'].mean():.4f}  +/- {scores['test_recall_weighted'].std():.4f}")
    print(f"  F1-Score  : {scores['test_f1_weighted'].mean():.4f}  +/- {scores['test_f1_weighted'].std():.4f}  << PRIMARY")
    print(f"  ROC-AUC   : {scores['test_roc_auc'].mean():.4f}  +/- {scores['test_roc_auc'].std():.4f}")

    return {
        "Model": model_name,
        "CV Accuracy": round(scores["test_accuracy"].mean(), 4),
        "CV Precision": round(scores["test_precision_weighted"].mean(), 4),
        "CV Recall": round(scores["test_recall_weighted"].mean(), 4),
        "CV F1-Score": round(scores["test_f1_weighted"].mean(), 4),
        "CV ROC-AUC":  round(scores["test_roc_auc"].mean(), 4),
    }


cv_results = []

cv_results.append(run_cv(
    LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced"),
    X_train, y_train, "Logistic Regression"
))

cv_results.append(run_cv(
    DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
    X_train, y_train, "Decision Tree (depth=5)"
))

cv_results.append(run_cv(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    X_train, y_train, "Random Forest (n=200)"
))

cv_results.append(run_cv(
    XGBClassifier(n_estimators=300, random_state=42,
                  eval_metric="logloss", n_jobs=-1, scale_pos_weight=neg/pos),
    X_train, y_train, "XGBoost (n=300)"
))

cv_results.append(run_cv(
    VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
            ("xgb", XGBClassifier(n_estimators=300, random_state=42,
                                  eval_metric="logloss", n_jobs=-1, scale_pos_weight=neg/pos))
        ],
        voting="soft"
    ),
    X_train, y_train, "Soft Voting Ensemble"
))

cv_df = pd.DataFrame(cv_results).set_index("Model")
print("\n" + "="*90)
print("  CV COMPARISON SUMMARY")
print("="*90)
print(cv_df.to_string())

with open("results_final/classification_reports.txt", "a") as f:
    f.write("\n" + "="*90 + "\n")
    f.write("5-FOLD CV COMPARISON SUMMARY\n")
    f.write("="*90 + "\n")
    f.write(cv_df.to_string())
    f.write("\n\n")

cv_df.to_csv("results_final/cv_comparison.csv")


# ── 3.3 GridSearchCV Hyperparameter Tuning ────────────────────────────────
print("\n" + "=" * 70)
print("  HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 70)

lr_param_grid = {
    "C":       [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver":  ["liblinear"],
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    lr_param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
)
lr_grid.fit(X_train, y_train)
print("LR        -> best params:", lr_grid.best_params_)
print("LR        -> best CV F1 :", round(lr_grid.best_score_, 4))

dt_param_grid = {
    "max_depth":        [3, 5, 7, 10, None],
    "min_samples_split":[2, 5, 10],
    "criterion":        ["gini", "entropy"],
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    dt_param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
)
dt_grid.fit(X_train, y_train)
print("DT        -> best params:", dt_grid.best_params_)
print("DT        -> best CV F1 :", round(dt_grid.best_score_, 4))

rf_param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
    rf_param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)
print("RF        -> best params:", rf_grid.best_params_)
print("RF        -> best CV F1 :", round(rf_grid.best_score_, 4))

xgb_param_grid = {
    "n_estimators":  [200, 300, 400, 500],
    "max_depth":     [3, 5, 7, 10],
    "learning_rate": [0.05, 0.1, 0.2, 0.3],
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss",
                  n_jobs=-1, scale_pos_weight=neg/pos),
    xgb_param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)
print("XGBoost   -> best params:", xgb_grid.best_params_)
print("XGBoost   -> best CV F1 :", round(xgb_grid.best_score_, 4))

with open("results_final/classification_reports.txt", "a") as f:
    f.write("\n" + "="*90 + "\n")
    f.write("GRIDSEARCHCV TUNING RESULTS\n")
    f.write("="*90 + "\n")
    f.write(f"LR        -> best params: {lr_grid.best_params_}\n")
    f.write(f"LR        -> best CV F1 : {round(lr_grid.best_score_, 4)}\n")
    f.write(f"DT        -> best params: {dt_grid.best_params_}\n")
    f.write(f"DT        -> best CV F1 : {round(dt_grid.best_score_, 4)}\n")
    f.write(f"RF        -> best params: {rf_grid.best_params_}\n")
    f.write(f"RF        -> best CV F1 : {round(rf_grid.best_score_, 4)}\n")
    f.write(f"XGBoost   -> best params: {xgb_grid.best_params_}\n")
    f.write(f"XGBoost   -> best CV F1 : {round(xgb_grid.best_score_, 4)}\n")
    f.write("\n\n")


# ── 3.4 Tuned Model Evaluation on Test Set ───────────────────────────────
print("\n" + "=" * 70)
print("  TUNED MODEL EVALUATION ON TEST SET")
print("=" * 70)

baseline_final = DummyClassifier(strategy="most_frequent", random_state=42)
baseline_final.fit(X_train, y_train)
evaluate_model_final(baseline_final, X_test, y_test, "Majority-Class Baseline")

evaluate_model_final(lr_grid.best_estimator_, X_test, y_test, "Logistic Regression (Tuned)")
evaluate_model_final(dt_grid.best_estimator_, X_test, y_test, "Decision Tree (Tuned)")
evaluate_model_final(rf_grid.best_estimator_, X_test, y_test, "Random Forest (Tuned)")
evaluate_model_final(xgb_grid.best_estimator_, X_test, y_test, "XGBoost (Tuned)")

# SVM evaluation on test set
print("\n" + "=" * 70)
print("  SVM EVALUATION ON TEST SET")
print("=" * 70)
svm_model = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
svm_model.fit(X_train, y_train)
evaluate_model_final(svm_model, X_test, y_test, "SVM (RBF kernel)")

# Tuned Soft Voting Ensemble (LR + RF + XGB)
print("\n" + "=" * 70)
print("  TUNED ENSEMBLE EVALUATION ON TEST SET")
print("=" * 70)
tuned_ensemble = VotingClassifier(
    estimators=[
        ("lr", lr_grid.best_estimator_),
        ("rf", rf_grid.best_estimator_),
        ("xgb", xgb_grid.best_estimator_)
    ],
    voting="soft"
)
tuned_ensemble.fit(X_train, y_train)
evaluate_model_final(tuned_ensemble, X_test, y_test, "Tuned Ensemble")


# ── 3.5 Tuned Feature Importance ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  TUNED FEATURE IMPORTANCE (RF + XGBoost)")
print("=" * 70)

feature_names = X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]

rf_importances = rf_grid.best_estimator_.feature_importances_
rf_idx = np.argsort(rf_importances)[::-1][:10]

xgb_importances = xgb_grid.best_estimator_.feature_importances_
xgb_idx = np.argsort(xgb_importances)[::-1][:10]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(range(10), rf_importances[rf_idx][::-1])
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([feature_names[i] for i in rf_idx][::-1])
axes[0].set_title("Random Forest (Tuned) - Top 10 Features")
axes[0].set_xlabel("Gini Importance")

axes[1].barh(range(10), xgb_importances[xgb_idx][::-1])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([feature_names[i] for i in xgb_idx][::-1])
axes[1].set_title("XGBoost (Tuned) - Top 10 Features")
axes[1].set_xlabel("Gain Importance")

plt.tight_layout()
save_fig_final("tuned_feature_importance")

print("\nTop 5 RF features:")
for i in rf_idx[:5]:
    print(f"  {feature_names[i]}: {rf_importances[i]:.4f}")

print("\nTop 5 XGBoost features:")
for i in xgb_idx[:5]:
    print(f"  {feature_names[i]}: {xgb_importances[i]:.4f}")


# ── 3.6 Tuned Decision Tree Rules ────────────────────────────────────────
print("\n" + "=" * 70)
print("  TUNED DECISION TREE RULES")
print("=" * 70)

dt_rules = export_text(dt_grid.best_estimator_,
                       feature_names=list(feature_names))
print(dt_rules[:2000])
with open("results_final/decision_tree_rules.txt", "w") as f:
    f.write(dt_rules)
print("  Saved: results_final/decision_tree_rules.txt")


# ── 3.7 Tuned Fairness Analysis ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  TUNED FAIRNESS ANALYSIS (Ensemble)")
print("=" * 70)

fairness_cols = [c for c in X_test.columns if c.startswith("race_") or c.startswith("sex_")]
best_model_final = tuned_ensemble
y_pred_final = best_model_final.predict(X_test)

fairness_rows_final = []
for col in fairness_cols:
    for val in [0, 1]:
        mask = X_test[col] == val
        if mask.sum() == 0:
            continue
        g_f1 = f1_score(y_test[mask], y_pred_final[mask], average="weighted")
        g_acc = accuracy_score(y_test[mask], y_pred_final[mask])
        fairness_rows_final.append({
            "Group": f"{col}={val}", "Size": int(mask.sum()),
            "Accuracy": round(g_acc, 4), "F1-Score": round(g_f1, 4)
        })
        print(f"Fairness: {col}={val}  n={int(mask.sum())}  Acc={g_acc:.4f}  F1={g_f1:.4f}")

fairness_df_final = pd.DataFrame(fairness_rows_final)
print("\nFairness Results (Tuned Ensemble):")
print(fairness_df_final.to_string(index=False))
fairness_df_final.to_csv("results_final/fairness_analysis.csv", index=False)


# ── 3.8 Final Tuned Model Comparison ─────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL TUNED MODEL COMPARISON")
print("=" * 70)

final_df_tuned = pd.DataFrame(all_results_final).set_index("Model")
print(final_df_tuned.round(4).to_string())
final_df_tuned.to_csv("results_final/model_comparison.csv")

fig, ax = plt.subplots(figsize=(12, 6))
final_df_tuned[["F1-Score", "ROC-AUC"]].plot(kind="bar", ax=ax, edgecolor="white")
ax.set_title("Tuned Model Comparison: F1-Score & ROC-AUC", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
save_fig_final("final_tuned_comparison")

print("\n" + "=" * 70)
print("  ALL DONE! Check results/ and results_final/ directories.")
print("=" * 70)
