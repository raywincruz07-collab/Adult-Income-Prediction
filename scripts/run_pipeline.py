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
from xgboost import XGBClassifier
import category_encoders as ce

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12, "axes.labelsize": 10})

# Create output directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

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
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

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
fair_cols = [c for c in X_test.columns if c.startswith("sex_") or c.startswith("race_")]
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
print("  ALL DONE! Check results/ directory for plots and CSVs.")
print("=" * 70)
