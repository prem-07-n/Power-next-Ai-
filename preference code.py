import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent

# 1. Load Cleaned Data
train_df = pd.read_excel(PROJECT_DIR / 'Training_Data_CLEANED.xlsx')
test_df = pd.read_excel(PROJECT_DIR / 'Test_Data_CLEANED.xlsx')

# 2. Prepare Features and Target
target_col = 'Reference_Parameter'
cols_to_drop = ['Test_ID', 'Reference_Parameter', 'Validity_Label', 'Validity_Label_Bin']

X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 3. Define the candidate models
# Wrapping in Pipelines with a SimpleImputer to prevent CV crashes if any NaNs remain
models = {
    'Linear Regression': Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', LinearRegression())]),
    'Random Forest': Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', RandomForestRegressor(random_state=42))]),
    'Extra Trees': Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', ExtraTreesRegressor(random_state=42))]),
    'Gradient Boosting': Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', GradientBoostingRegressor(random_state=42))])
}

# 4. Compare models using MAE, RMSE, and Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

best_model_name = ""
best_mae = float('inf')
best_rmse = float('inf')

print("Evaluating models (this may take a moment)...\n")
for name, model in models.items():
    cv_results = cross_validate(
        model, X_train, y_train, cv=kf,
        scoring={'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}
    )
    
    mae = -cv_results['test_mae'].mean()
    rmse = -cv_results['test_rmse'].mean()
    results[name] = {'MAE': mae, 'RMSE': rmse}
    
    # Track the best model
    if mae < best_mae:
        best_mae = mae
        best_rmse = rmse
        best_model_name = name

# 5. Extract Important Features for the Best Model
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Using permutation importance works for all models (including HistGradientBoosting which lacks .feature_importances_)
perm_importance = permutation_importance(best_model, X_train, y_train, n_repeats=5, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
top_features = [f"{X_train.columns[i]} (score: {perm_importance.importances_mean[i]:.4f})" for i in sorted_idx[:5]]

# 6. Generate the exact required output for the team
print("=" * 50)
print("OUTPUT TO TELL THE TEAM:")
print("=" * 50)
print(f"Best Model: {best_model_name}")
print(f"Validation MAE: {best_mae:.4f}")
print(f"Validation RMSE: {best_rmse:.4f}")
print("Important Features:")
for feature in top_features:
    print(f"  - {feature}")
print("=" * 50)

# 7. Final Predictions
test_df['Predicted_Reference_Parameter'] = best_model.predict(X_test)
submission = test_df[['Test_ID', 'Predicted_Reference_Parameter']]
output_path = PROJECT_DIR / 'Reference_Predictions.csv'
submission.to_csv(output_path, index=False)
print(f"\nFinal predictions using {best_model_name} saved to {output_path}")