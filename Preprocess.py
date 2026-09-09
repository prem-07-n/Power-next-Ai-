"""
CPRI Hackathon - Data Preprocessing Pipeline
=============================================
Cleans Training_Data and Test_Data sheets from
CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx

Steps performed:
1. Load raw data
2. Remove exact-duplicate measurement pairs (same Voltage/Current/Temp/
   Duration/Sensor readings under two different Test_IDs) from training data
3. Flag physically-impossible sensor readings (negative / near-zero values)
4. Flag statistical outliers per sensor (IQR method) as an engineered
   "fault flag" feature -- these strongly correlate with Validity_Label
   and are informative, so they are flagged rather than deleted
5. Impute missing sensor values (Sensor_S1-S4) using the median from the
   TRAINING set only (fit on train, applied to both train & test -> no
   data leakage)
6. Encode Validity_Label to a binary target (Valid=1 / Invalid=0)
7. Add a couple of simple engineered features (Power, missing-count)
8. Save cleaned Training and Test sets to CSV
"""

import pandas as pd
import numpy as np 

RAW_PATH  = r"C:\Users\nadip\OneDrive\Documents\Power-next-Ai-\CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
OUT_TRAIN = r"C:\Users\nadip\OneDrive\Documents\Power-next-Ai-\Training_Data_CLEANED.xlsx"
OUT_TEST  = r"C:\Users\nadip\OneDrive\Documents\Power-next-Ai-\Test_Data_CLEANED.xlsx"
INPUT_COLS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
]
SENSOR_COLS = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
FEATURE_COLS = INPUT_COLS + SENSOR_COLS


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_excel(RAW_PATH, sheet_name="Training_Data")
    test = pd.read_excel(RAW_PATH, sheet_name="Test_Data")
    print(f"Loaded Training_Data: {train.shape}, Test_Data: {test.shape}")
    return train, test


# ---------------------------------------------------------------------------
# 2. REMOVE DUPLICATE MEASUREMENT PAIRS (train only - these are labelled
#    rows so we can safely drop the redundant copy)
# ---------------------------------------------------------------------------
def drop_duplicate_pairs(df):
    before = len(df)
    dup_mask = df.duplicated(subset=FEATURE_COLS, keep="first")
    print(f"Duplicate measurement pairs found: {dup_mask.sum()}")
    df = df.loc[~dup_mask].reset_index(drop=True)
    print(f"Rows dropped: {before - len(df)} -> remaining: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# 3 & 4. FLAG IMPOSSIBLE / OUTLIER SENSOR READINGS
#    (kept as features, not removed -- they are diagnostic of faulty tests)
# ---------------------------------------------------------------------------
def flag_sensor_faults(df, bounds=None):
    """
    bounds: dict of {col: (lower, upper)} IQR bounds. If None, they are
    computed from this dataframe (use TRAIN bounds when calling for test).
    Returns (df_with_flags, bounds_used)
    """
    df = df.copy()
    computed_bounds = {}
    fault_flag = pd.Series(0, index=df.index)

    for col in SENSOR_COLS:
        # physically impossible: negative sensor readings
        neg_mask = df[col] < 0
        df[f"{col}_negative_flag"] = neg_mask.astype(int)

        # statistical outliers via IQR (computed on TRAIN if bounds is None)
        if bounds is None:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            lo, hi = bounds[col]
        computed_bounds[col] = (lo, hi)

        out_mask = (df[col] < lo) | (df[col] > hi)
        df[f"{col}_outlier_flag"] = out_mask.astype(int)

        fault_flag = fault_flag | neg_mask.astype(int) | out_mask.astype(int)

    df["Sensor_Fault_Flag"] = fault_flag
    print(f"Rows flagged with at least one sensor fault: {fault_flag.sum()}")
    return df, computed_bounds


# ---------------------------------------------------------------------------
# 5. IMPUTE MISSING SENSOR VALUES (median from TRAIN, applied to both)
# ---------------------------------------------------------------------------
def impute_missing(df, medians=None):
    df = df.copy()
    if medians is None:
        medians = df[SENSOR_COLS].median()

    # track what was missing, before filling (useful predictive feature)
    for col in SENSOR_COLS:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    print("Missing values before imputation:")
    print(df[SENSOR_COLS].isna().sum())

    df[SENSOR_COLS] = df[SENSOR_COLS].fillna(medians)

    print("Missing values after imputation:")
    print(df[SENSOR_COLS].isna().sum())
    return df, medians


# ---------------------------------------------------------------------------
# 6. ENCODE LABEL (train only)
# ---------------------------------------------------------------------------
def encode_label(df):
    df = df.copy()
    df["Validity_Label_Bin"] = df["Validity_Label"].map({"Valid": 1, "Invalid": 0})
    return df


# ---------------------------------------------------------------------------
# 7. SIMPLE FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["Power_kW"] = df["Applied_Voltage_kV"] * df["Load_Current_A"]  # V*I
    df["Total_Missing_Sensors"] = df[[c for c in df.columns if c.endswith("_was_missing")]].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    train, test = load_data()

    # --- Training set ---
    train = drop_duplicate_pairs(train)
    train, sensor_bounds = flag_sensor_faults(train, bounds=None)
    train, sensor_medians = impute_missing(train, medians=None)
    train = encode_label(train)
    train = add_features(train)

    # --- Test set (reuse TRAIN's bounds/medians -> no leakage) ---
    test, _ = flag_sensor_faults(test, bounds=sensor_bounds)
    test, _ = impute_missing(test, medians=sensor_medians)
    test = add_features(test)

    # --- Save ---
    train.to_excel(OUT_TRAIN, index=False)
    test.to_excel(OUT_TEST, index=False)
    print(f"\nSaved cleaned training data -> {OUT_TRAIN}  {train.shape}")
    print(f"Saved cleaned test data     -> {OUT_TEST}  {test.shape}")

    return train, test


if __name__ == "__main__":
    main()
