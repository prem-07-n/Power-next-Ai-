"""
Validity Prediction Pipeline
=============================
Goal: Predict VALID / INVALID test records, compare 4 classifiers,
evaluate with Precision/Recall/F1/Confusion Matrix, investigate the
root causes behind "Invalid" records, and score the held-out test file.

Run: python validity_pipeline.py
Outputs (written to ./outputs/):
    - model_comparison.csv          (metrics for all 4 models)
    - confusion_matrices.png
    - feature_importance.png
    - invalid_pattern_analysis.png
    - test_predictions.csv          (final predictions for Test_Data_CLEANED.csv)
    - summary_report.txt
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_LABEL = "Invalid"  # the minority / "interesting" class we care about detecting

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
train_raw = pd.read_excel(PROJECT_DIR / "Training_Data_CLEANED.xlsx")
test_raw = pd.read_excel(PROJECT_DIR / "Test_Data_CLEANED.xlsx")

print(f"Training rows: {len(train_raw)} | Test rows: {len(test_raw)}")
print(train_raw["Validity_Label"].value_counts())

# Feature columns: everything common to train & test, excluding ID.
# Reference_Parameter and the label columns only exist in training, so they
# are NOT used as model features (would not be available at inference time).
FEATURES = [c for c in test_raw.columns if c != "Test_ID"]
print(f"\nUsing {len(FEATURES)} features:\n{FEATURES}")

X = train_raw[FEATURES].copy()
y = train_raw["Validity_Label_Bin"]  # 1 = Valid, 0 = Invalid (from the source file)

X_test_final = test_raw[FEATURES].copy()

# ---------------------------------------------------------------------------
# 2. TRAIN / VALIDATION SPLIT (stratified, since test file is unlabeled we
#    need a held-out labeled split to actually compute Precision/Recall/F1)
# ---------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scaled versions for Logistic Regression (distance/gradient based model);
# tree ensembles are scale-invariant so they use the raw features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# ---------------------------------------------------------------------------
# 3. MODELS
#    class imbalance (12.4% Invalid) is handled via class_weight='balanced'
#    (or per-sample weights for GradientBoosting, which has no class_weight arg)
# ---------------------------------------------------------------------------
sample_weight_train = compute_sample_weight("balanced", y_train)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        X_train_scaled, X_val_scaled, X_test_scaled, False
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
        X_train, X_val, X_test_final, False
    ),
    "Extra Trees": (
        ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
        X_train, X_val, X_test_final, False
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        X_train, X_val, X_test_final, True  # needs sample_weight instead of class_weight
    ),
}

results = []
fitted_models = {}
val_predictions = {}

for name, (model, xtr, xval, xtest, use_sample_weight) in models.items():
    if use_sample_weight:
        model.fit(xtr, y_train, sample_weight=sample_weight_train)
    else:
        model.fit(xtr, y_train)

    preds = model.predict(xval)
    val_predictions[name] = preds
    fitted_models[name] = model

    # NOTE: label 0 = Invalid, 1 = Valid. We report metrics for the Invalid
    # class (pos_label=0) since that's the class of real operational interest,
    # plus macro-F1 for an overall balanced view.
    precision_invalid = precision_score(y_val, preds, pos_label=0, zero_division=0)
    recall_invalid = recall_score(y_val, preds, pos_label=0, zero_division=0)
    f1_invalid = f1_score(y_val, preds, pos_label=0, zero_division=0)
    f1_macro = f1_score(y_val, preds, average="macro")

    results.append({
        "Model": name,
        "Precision (Invalid)": round(precision_invalid, 4),
        "Recall (Invalid)": round(recall_invalid, 4),
        "F1 (Invalid)": round(f1_invalid, 4),
        "F1 (macro)": round(f1_macro, 4),
    })

results_df = pd.DataFrame(results).sort_values("F1 (Invalid)", ascending=False)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
print("\n=== Model Comparison (validation split) ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
best_preds = val_predictions[best_model_name]
best_f1_invalid = results_df.iloc[0]["F1 (Invalid)"]

print(f"\nBest model: {best_model_name} (F1 on Invalid class = {best_f1_invalid})")
print("\nFull classification report for best model:")
print(classification_report(y_val, best_preds, target_names=["Invalid", "Valid"]))

# ---------------------------------------------------------------------------
# 4. CONFUSION MATRICES (all 4 models, for the deliverable)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, (name, preds) in zip(axes.ravel(), val_predictions.items()):
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nF1(Invalid)={f1_score(y_val, preds, pos_label=0, zero_division=0):.3f}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Invalid", "Valid"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Invalid", "Valid"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (from the best model)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
elif hasattr(best_model, "coef_"):
    importances = pd.Series(np.abs(best_model.coef_[0]), index=FEATURES).sort_values(ascending=False)
else:
    importances = pd.Series(dtype=float)

top_importances = importances.head(12)
top_importances.iloc[::-1].plot(kind="barh")
plt.title(f"Top Feature Importances — {best_model_name}")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close()

print("\n=== Top 10 Important Features ===")
print(importances.head(10).to_string())

# ---------------------------------------------------------------------------
# 6. WHY ARE RECORDS INVALID? (root-cause investigation, not just black-box)
# ---------------------------------------------------------------------------
invalid_mask = train_raw["Validity_Label"] == "Invalid"
valid_mask = train_raw["Validity_Label"] == "Valid"

flag_cols = [c for c in FEATURES if "flag" in c.lower() or "missing" in c.lower()]
flag_cols += ["Total_Missing_Sensors"] if "Total_Missing_Sensors" not in flag_cols else []
flag_cols = list(dict.fromkeys(flag_cols))  # de-dup, keep order

flag_rate_compare = pd.DataFrame({
    "Invalid_rate": train_raw.loc[invalid_mask, flag_cols].mean(),
    "Valid_rate": train_raw.loc[valid_mask, flag_cols].mean(),
})
flag_rate_compare["lift"] = (flag_rate_compare["Invalid_rate"] /
                              flag_rate_compare["Valid_rate"].replace(0, np.nan))
flag_rate_compare = flag_rate_compare.sort_values("Invalid_rate", ascending=False)
flag_rate_compare.to_csv(f"{OUT_DIR}/invalid_flag_analysis.csv")

print("\n=== Flag prevalence: Invalid vs Valid records ===")
print(flag_rate_compare.to_string())

# Continuous-feature distribution shift between Invalid & Valid
numeric_cols = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
                "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
                "Power_kW"]
dist_compare = pd.DataFrame({
    "Invalid_mean": train_raw.loc[invalid_mask, numeric_cols].mean(),
    "Valid_mean": train_raw.loc[valid_mask, numeric_cols].mean(),
    "Invalid_std": train_raw.loc[invalid_mask, numeric_cols].std(),
    "Valid_std": train_raw.loc[valid_mask, numeric_cols].std(),
})
dist_compare.to_csv(f"{OUT_DIR}/invalid_numeric_distribution.csv")
print("\n=== Numeric feature shift: Invalid vs Valid ===")
print(dist_compare.to_string())

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
plot_cols = ["Sensor_Fault_Flag", "Total_Missing_Sensors", "Sensor_S2_outlier_flag",
             "Sensor_S1", "Sensor_S2", "Power_kW"]
for ax, col in zip(axes.ravel(), plot_cols):
    train_raw.boxplot(column=col, by="Validity_Label", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
plt.suptitle("Invalid vs Valid — abnormal pattern check")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/invalid_pattern_analysis.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. RETRAIN BEST MODEL ON FULL TRAINING DATA, PREDICT ON TEST FILE
# ---------------------------------------------------------------------------
if best_model_name == "Logistic Regression":
    full_scaler = StandardScaler().fit(X)
    X_full_scaled = full_scaler.transform(X)
    X_test_scaled_final = full_scaler.transform(X_test_final)
    final_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    final_model.fit(X_full_scaled, y)
    test_preds = final_model.predict(X_test_scaled_final)
elif best_model_name == "Gradient Boosting":
    sw_full = compute_sample_weight("balanced", y)
    final_model = GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
    final_model.fit(X, y, sample_weight=sw_full)
    test_preds = final_model.predict(X_test_final)
else:
    ModelClass = RandomForestClassifier if best_model_name == "Random Forest" else ExtraTreesClassifier
    final_model = ModelClass(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)
    final_model.fit(X, y)
    test_preds = final_model.predict(X_test_final)

test_output = test_raw[["Test_ID"]].copy()
test_output["Predicted_Validity"] = np.where(test_preds == 1, "Valid", "Invalid")
test_output.to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)

print(f"\nPredicted distribution on test file:\n{test_output['Predicted_Validity'].value_counts()}")

# ---------------------------------------------------------------------------
# 8. WRITE SUMMARY REPORT
# ---------------------------------------------------------------------------
with open(f"{OUT_DIR}/summary_report.txt", "w") as f:
    f.write("VALIDITY PREDICTION — SUMMARY REPORT\n")
    f.write("=" * 45 + "\n\n")
    f.write("Best Model: " + best_model_name + "\n")
    f.write(f"F1 Score (Invalid class): {best_f1_invalid}\n\n")
    f.write("Model comparison (validation split):\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    f.write("Top Important Features:\n")
    f.write(importances.head(10).to_string() + "\n\n")
    f.write("Flag prevalence — Invalid vs Valid:\n")
    f.write(flag_rate_compare.to_string() + "\n\n")
    f.write("Test-set prediction distribution:\n")
    f.write(test_output["Predicted_Validity"].value_counts().to_string() + "\n")

print("\nDone. See ./outputs/ for all artifacts.")
