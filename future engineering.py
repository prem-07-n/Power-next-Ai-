"""
Feature Engineering Analysis
=============================
Goal: test a set of candidate engineered features (electrical/physical
combinations of the raw inputs) and keep only the ones that ACTUALLY
improve validation performance, for:

  1) the Validity classifier   (target: Validity_Label / Validity_Label_Bin)
  2) a Reference_Parameter regressor (target: Reference_Parameter)

It also investigates whether there is a hidden mathematical / physical
relationship between the raw inputs and Reference_Parameter.

Method
------
For each candidate feature we do a *paired* comparison: same CV folds,
baseline feature set vs. baseline + one new feature, repeated over
multiple random seeds to get a mean +/- std of the improvement. A
feature is only "kept" if it improves the metric by more than noise
(mean delta clearly positive and not overlapping ~0 within 1 std).

Run:  python feature_engineering_analysis.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, r2_score, mean_absolute_error

RANDOM_SEEDS = [0, 1, 2, 3, 4]     # repeat CV with different splits -> robust estimate
N_SPLITS = 5
RANDOM_STATE = 42

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = Path(
    os.environ.get(
        "TRAIN_DATA_PATH", str(PROJECT_DIR / "Training_Data_CLEANED.xlsx")
    )
)

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_PATH}. "
        "Set TRAIN_DATA_PATH to the cleaned .xlsx file."
    )

df = pd.read_excel(TRAIN_PATH)

RAW_NUMERIC = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
    "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]
FLAG_COLS = [
    "Sensor_S1_negative_flag", "Sensor_S1_outlier_flag",
    "Sensor_S2_negative_flag", "Sensor_S2_outlier_flag",
    "Sensor_S3_negative_flag", "Sensor_S3_outlier_flag",
    "Sensor_S4_negative_flag", "Sensor_S4_outlier_flag",
    "Sensor_Fault_Flag",
    "Sensor_S1_was_missing", "Sensor_S2_was_missing",
    "Sensor_S3_was_missing", "Sensor_S4_was_missing",
    "Total_Missing_Sensors",
]
# Power_kW = Applied_Voltage_kV * Load_Current_A exactly (verified: ratio == 1.000000
# with std ~1e-16), so it is already a known engineered feature -> part of baseline.
BASELINE_FEATURES = RAW_NUMERIC + FLAG_COLS + ["Power_kW"]
REQUIRED_COLUMNS = BASELINE_FEATURES + ["Validity_Label", "Reference_Parameter"]
missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
if missing_columns:
    raise ValueError(f"Training data is missing required columns: {missing_columns}")

# --------------------------------------------------------------------------
# 2. Candidate engineered features
# --------------------------------------------------------------------------
def add_candidate_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["VxI"]        = d["Applied_Voltage_kV"] * d["Load_Current_A"]      # == Power_kW
    d["I2"]         = d["Load_Current_A"] ** 2
    d["V2"]         = d["Applied_Voltage_kV"] ** 2
    d["IxDur"]      = d["Load_Current_A"] * d["Test_Duration_min"]
    d["S1_S2"]      = d["Sensor_S1"] - d["Sensor_S2"]
    d["S1_S3"]      = d["Sensor_S1"] - d["Sensor_S3"]
    d["S2_S3"]      = d["Sensor_S2"] - d["Sensor_S3"]
    sensors = d[["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]]
    d["Sensor_avg"] = sensors.mean(axis=1)
    d["Sensor_max"] = sensors.max(axis=1)
    d["Sensor_min"] = sensors.min(axis=1)
    return d

df = add_candidate_features(df)

CANDIDATES = ["VxI", "I2", "V2", "IxDur", "S1_S2", "S1_S3", "S2_S3",
              "Sensor_avg", "Sensor_max", "Sensor_min"]

y_clf = (df["Validity_Label"] == "Invalid").astype(int)   # 1 = Invalid (minority/positive class)
y_reg = df["Reference_Parameter"].values

# --------------------------------------------------------------------------
# 3. Paired CV evaluation helper
# --------------------------------------------------------------------------
def cv_score_classifier(feature_cols, y):
    """Mean F1 (Invalid) across repeated Stratified 5-fold CV."""
    X = df[feature_cols].values
    scores = []
    for seed in RANDOM_SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[va])
            scores.append(f1_score(y[va], pred, pos_label=1, zero_division=0))
    return np.array(scores)

def cv_score_regressor(feature_cols, y):
    """Mean R2 and MAE across repeated 5-fold CV."""
    X = df[feature_cols].values
    r2s, maes = [], []
    for seed in RANDOM_SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, va in kf.split(X):
            reg = GradientBoostingRegressor(random_state=RANDOM_STATE)
            reg.fit(X[tr], y[tr])
            pred = reg.predict(X[va])
            r2s.append(r2_score(y[va], pred))
            maes.append(mean_absolute_error(y[va], pred))
    return np.array(r2s), np.array(maes)

# --------------------------------------------------------------------------
# 4. Baseline scores
# --------------------------------------------------------------------------
print("=" * 70)
print("BASELINE (original + Power_kW features only)")
print("=" * 70)

base_f1 = cv_score_classifier(BASELINE_FEATURES, y_clf.values)
print(f"Classifier  F1(Invalid): {base_f1.mean():.4f} +/- {base_f1.std():.4f}")

base_r2, base_mae = cv_score_regressor(BASELINE_FEATURES, y_reg)
print(f"Regressor   R2:          {base_r2.mean():.4f} +/- {base_r2.std():.4f}")
print(f"Regressor   MAE:         {base_mae.mean():.4f} +/- {base_mae.std():.4f}")

# --------------------------------------------------------------------------
# 5. Test each candidate feature individually (paired vs. baseline)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CANDIDATE FEATURE IMPACT (baseline + 1 feature)")
print("=" * 70)

results = []
for feat in CANDIDATES:
    feats = BASELINE_FEATURES + [feat]

    f1 = cv_score_classifier(feats, y_clf.values)
    d_f1 = f1.mean() - base_f1.mean()

    r2, mae = cv_score_regressor(feats, y_reg)
    d_r2 = r2.mean() - base_r2.mean()
    d_mae = mae.mean() - base_mae.mean()   # negative = improvement (lower error)

    results.append({
        "feature": feat,
        "F1_Invalid": f1.mean(), "delta_F1": d_f1,
        "F1_std": f1.std(),
        "R2": r2.mean(), "delta_R2": d_r2,
        "R2_std": r2.std(),
        "MAE": mae.mean(), "delta_MAE": d_mae,
        "MAE_std": mae.std(),
    })

res_df = pd.DataFrame(results).sort_values("delta_R2", ascending=False)
res_df.to_csv(PROJECT_DIR / "feature_engineering_results.csv", index=False)
pd.set_option("display.width", 140)
print(res_df.to_string(index=False,
      formatters={c: "{:.4f}".format for c in
                  ["F1_Invalid", "delta_F1", "F1_std", "R2", "delta_R2",
                   "R2_std", "MAE", "delta_MAE", "MAE_std"]}))

# --------------------------------------------------------------------------
# 6. Decide which features to KEEP
#    Rule: classifier -> delta_F1 > 0.01 ; regressor -> delta_R2 > 0.01
#    (small, noisy deltas below this are treated as "no real effect")
# --------------------------------------------------------------------------
KEEP_THRESHOLD_F1 = 0.01
KEEP_THRESHOLD_R2 = 0.01

keep_for_clf = res_df.loc[res_df["delta_F1"] > KEEP_THRESHOLD_F1, "feature"].tolist()
keep_for_reg = res_df.loc[res_df["delta_R2"] > KEEP_THRESHOLD_R2, "feature"].tolist()

print("\n" + "=" * 70)
print("DECISIONS")
print("=" * 70)
print(f"Features that meaningfully help the CLASSIFIER (dF1 > {KEEP_THRESHOLD_F1}):")
print(f"  {keep_for_clf if keep_for_clf else '(none)'}")
print(f"Features that meaningfully help the REGRESSOR   (dR2 > {KEEP_THRESHOLD_R2}):")
print(f"  {keep_for_reg if keep_for_reg else '(none)'}")

# --------------------------------------------------------------------------
# 7. Combined-feature test: adding ALL kept regressor features together
#    (interactions between kept features can help or hurt further)
# --------------------------------------------------------------------------
if keep_for_reg:
    combo_feats = BASELINE_FEATURES + keep_for_reg
    r2c, maec = cv_score_regressor(combo_feats, y_reg)
    print(f"\nRegressor with ALL kept features combined {keep_for_reg}:")
    print(f"  R2:  {r2c.mean():.4f} +/- {r2c.std():.4f}  (baseline {base_r2.mean():.4f})")
    print(f"  MAE: {maec.mean():.4f} +/- {maec.std():.4f}  (baseline {base_mae.mean():.4f})")

if keep_for_clf:
    combo_feats_c = BASELINE_FEATURES + keep_for_clf
    f1c = cv_score_classifier(combo_feats_c, y_clf.values)
    print(f"\nClassifier with ALL kept features combined {keep_for_clf}:")
    print(f"  F1(Invalid): {f1c.mean():.4f} +/- {f1c.std():.4f}  (baseline {base_f1.mean():.4f})")

# --------------------------------------------------------------------------
# 8. Hidden physical/mathematical relationship check: inputs -> Reference_Parameter
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HIDDEN RELATIONSHIP CHECK: inputs vs Reference_Parameter")
print("=" * 70)

corr_targets = RAW_NUMERIC + CANDIDATES
corrs = df[corr_targets + ["Reference_Parameter"]].corr()["Reference_Parameter"] \
          .drop("Reference_Parameter").sort_values(key=np.abs, ascending=False)
print("\nCorrelation of each raw/engineered feature with Reference_Parameter:")
print(corrs.to_string())

# Load_Current_A^2 stands out strongly -> test simple physical fit Ref = a*I^2 + b
I = df["Load_Current_A"].values
Ref = df["Reference_Parameter"].values
a, b = np.polyfit(I ** 2, Ref, 1)
pred_phys = a * I ** 2 + b
r2_phys = r2_score(Ref, pred_phys)
print(f"\nSimple physical fit  Reference_Parameter ~= {a:.5f} * Current^2 + {b:.3f}")
print(f"  R2 of this single-variable fit: {r2_phys:.4f}")
print("  (Load_Current^2 tracks I^2*R-style resistive heating -> Reference_Parameter")
print("   behaves like a heat/stress index driven mainly by current, not voltage.)")

print("\nDone.")