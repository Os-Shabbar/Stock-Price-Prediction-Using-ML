"""
Stock Price Prediction - XGBoost Model Building and Training (Part 4B)
This script builds and trains the XGBoost model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("XGBOOST MODEL - BUILDING AND TRAINING")
print("="*70)

# ============================================
# STEP 1: Load Preprocessed Data
# ============================================
print("\nStep 1: Loading Preprocessed XGBoost Data...")
print("-" * 70)

# Load XGBoost data (already flattened)
X_train = np.load('X_train_xgb.npy')
y_train = np.load('y_train_xgb.npy')
X_val = np.load('X_val_xgb.npy')
y_val = np.load('y_val_xgb.npy')
X_test = np.load('X_test_xgb.npy')
y_test = np.load('y_test_xgb.npy')

print(f"✓ Data loaded successfully!")
print(f"\nData shapes:")
print(f"  X_train: {X_train.shape} (samples, flattened_features)")
print(f"  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_test:  {y_test.shape}")

n_samples, n_features = X_train.shape
print(f"\nXGBoost configuration:")
print(f"  • Total features: {n_features}")
print(f"  • Training samples: {n_samples}")

# ============================================
# STEP 2: Create DMatrix for XGBoost
# ============================================
print("\n" + "="*70)
print("Step 2: Creating XGBoost DMatrix")
print("-" * 70)

# DMatrix is XGBoost's optimized data structure
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

print("✓ DMatrix created for train, validation, and test sets")

# ============================================
# STEP 3: Configure XGBoost Parameters
# ============================================
print("\n" + "="*70)
print("Step 3: Configuring XGBoost Parameters")
print("-" * 70)

params = {
    # Objective function
    'objective': 'reg:squarederror',  # Regression with squared error
    
    # Tree parameters
    'max_depth': 6,                    # Maximum tree depth (prevent overfitting)
    'min_child_weight': 3,             # Minimum sum of instance weight in child
    'eta': 0.05,                       # Learning rate (lower = slower but better)
    'gamma': 0.1,                      # Minimum loss reduction for split
    
    # Regularization
    'subsample': 0.8,                  # Sample 80% of training data
    'colsample_bytree': 0.8,           # Sample 80% of features
    'alpha': 0.1,                      # L1 regularization
    'lambda': 1.0,                     # L2 regularization
    
    # Training
    'seed': 42,                        # Reproducibility
    'eval_metric': 'rmse',             # Evaluation metric
    'verbosity': 1                     # Show progress
}

print("✓ Parameters configured:")
for key, value in params.items():
    print(f"  • {key}: {value}")

# ============================================
# STEP 4: Train XGBoost Model
# ============================================
print("\n" + "="*70)
print("Step 4: Training XGBoost Model")
print("-" * 70)

print("\nTraining with early stopping...")
print("(Stops if validation error doesn't improve for 50 rounds)")
print("-" * 70)

# Evaluation list for monitoring
evallist = [(dtrain, 'train'), (dval, 'validation')]

# Train the model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,           # Maximum iterations
    evals=evallist,
    early_stopping_rounds=50,       # Stop if no improvement
    verbose_eval=50                 # Print every 50 rounds
)

print("\n" + "="*70)
print("✓ Training completed!")
print("="*70)

# Get best iteration
best_iteration = model.best_iteration
print(f"\nBest iteration: {best_iteration}")

# ============================================
# STEP 5: Make Predictions
# ============================================
print("\n" + "="*70)
print("Step 5: Making Predictions")
print("-" * 70)

# Predictions (scaled)
y_train_pred = model.predict(dtrain)
y_val_pred = model.predict(dval)
y_test_pred = model.predict(dtest)

print("✓ Predictions generated on all datasets")

# Load scaler to inverse transform
target_scaler = joblib.load('target_scaler.pkl')

# Inverse transform to get actual prices
y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1))
y_train_pred_actual = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1))

y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1))
y_val_pred_actual = target_scaler.inverse_transform(y_val_pred.reshape(-1, 1))

y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_test_pred_actual = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1))

print("✓ Predictions inverse transformed to actual prices")

# ============================================
# STEP 6: Calculate Performance Metrics
# ============================================
print("\n" + "="*70)
print("Step 6: Evaluating XGBoost Performance")
print("-" * 70)

# Training metrics
train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
train_r2 = r2_score(y_train_actual, y_train_pred_actual)

# Validation metrics
val_mae = mean_absolute_error(y_val_actual, y_val_pred_actual)
val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred_actual))
val_r2 = r2_score(y_val_actual, y_val_pred_actual)

# Test metrics
test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
test_r2 = r2_score(y_test_actual, y_test_pred_actual)

print("\n" + "="*70)
print("📊 XGBOOST MODEL PERFORMANCE")
print("="*70)

print(f"\nTraining Set:")
print(f"  • MAE:  ${train_mae:.2f}")
print(f"  • RMSE: ${train_rmse:.2f}")
print(f"  • R² Score: {train_r2:.4f}")

print(f"\nValidation Set:")
print(f"  • MAE:  ${val_mae:.2f}")
print(f"  • RMSE: ${val_rmse:.2f}")
print(f"  • R² Score: {val_r2:.4f}")

print(f"\nTest Set:")
print(f"  • MAE:  ${test_mae:.2f}")
print(f"  • RMSE: ${test_rmse:.2f}")
print(f"  • R² Score: {test_r2:.4f}")

# ============================================
# STEP 7: Compare with LSTM
# ============================================
print("\n" + "="*70)
print("📈 COMPARISON: XGBoost vs LSTM")
print("="*70)

print("\nImproved LSTM Test Results:")
print("  • MAE:  $31.83")
print("  • RMSE: $35.08")
print("  • R² Score: -1.3240")

print(f"\nXGBoost Test Results:")
print(f"  • MAE:  ${test_mae:.2f}")
print(f"  • RMSE: ${test_rmse:.2f}")
print(f"  • R² Score: {test_r2:.4f}")

if test_mae < 31.83:
    improvement = ((31.83 - test_mae) / 31.83) * 100
    print(f"\n✅ XGBoost is {improvement:.1f}% better than LSTM!")
elif test_mae < 55.77:
    print(f"\n✅ XGBoost outperforms original LSTM but needs improvement vs LSTM-v2")
else:
    print(f"\n⚠ XGBoost underperforms LSTM")

# Overfitting check
print("\n" + "="*70)
print("🔍 OVERFITTING CHECK")
print("="*70)

train_val_gap = abs(train_mae - val_mae)
val_test_gap = abs(val_mae - test_mae)

print(f"\nMAE Gap Analysis:")
print(f"  • Train → Val gap: ${train_val_gap:.2f}")
print(f"  • Val → Test gap: ${val_test_gap:.2f}")

if train_val_gap < 5 and val_test_gap < 10:
    print("  ✅ Excellent generalization!")
elif train_val_gap < 10 and val_test_gap < 20:
    print("  ✅ Good generalization")
elif train_val_gap < 20:
    print("  ⚠ Moderate overfitting")
else:
    print("  ❌ Severe overfitting")

# ============================================
# STEP 8: Feature Importance Analysis
# ============================================
print("\n" + "="*70)
print("Step 8: Feature Importance Analysis")
print("-" * 70)

# Get feature importance
importance_dict = model.get_score(importance_type='gain')

# Sort by importance
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 20 Most Important Features:")
print("(Based on information gain)")
print("-" * 70)

for i, (feature, importance) in enumerate(sorted_importance[:20], 1):
    print(f"  {i:2d}. Feature {feature}: {importance:.2f}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(12, 8))

# Get top 30 features
top_features = sorted_importance[:30]
features = [f"F{f[0]}" for f in top_features]
importances = [f[1] for f in top_features]

ax.barh(features, importances, color='steelblue', alpha=0.7)
ax.set_xlabel('Importance (Gain)', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('XGBoost: Top 30 Feature Importances', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance plot saved as 'xgboost_feature_importance.png'")
plt.show()

# ============================================
# STEP 9: Visualize Predictions
# ============================================
print("\n" + "="*70)
print("Step 9: Visualizing Predictions")
print("-" * 70)

# Load test dates
test_dates = np.load('test_dates.npy', allow_pickle=True)

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Test predictions
axes[0].plot(test_dates, y_test_actual, label='Actual Price', 
             linewidth=2, alpha=0.7, color='blue')
axes[0].plot(test_dates, y_test_pred_actual, label='XGBoost Prediction', 
             linewidth=2, alpha=0.7, color='red')
axes[0].set_title('XGBoost: Test Set Predictions vs Actual', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Stock Price ($)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Prediction errors
errors = y_test_actual.flatten() - y_test_pred_actual.flatten()
axes[1].plot(test_dates, errors, label='Prediction Error', 
             color='red', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('XGBoost: Prediction Errors on Test Set', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Error ($)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Plot 3: Error distribution
axes[2].hist(errors, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[2].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[2].axvline(x=np.mean(errors), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: ${np.mean(errors):.2f}')
axes[2].set_title('XGBoost: Distribution of Prediction Errors', 
                  fontsize=14, fontweight='bold')
axes[2].set_xlabel('Error ($)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('xgboost_predictions_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Predictions visualization saved")
plt.show()

# ============================================
# STEP 10: Save Model and Results
# ============================================
print("\n" + "="*70)
print("Step 10: Saving XGBoost Model and Results")
print("-" * 70)

# Save the model
model.save_model('xgboost_model.json')
print("✓ Model saved as 'xgboost_model.json'")

# Save predictions
np.save('xgboost_train_predictions.npy', y_train_pred_actual)
np.save('xgboost_val_predictions.npy', y_val_pred_actual)
np.save('xgboost_test_predictions.npy', y_test_pred_actual)
print("✓ Predictions saved")

# Save metrics
metrics = {
    'train_mae': train_mae,
    'train_rmse': train_rmse,
    'train_r2': train_r2,
    'val_mae': val_mae,
    'val_rmse': val_rmse,
    'val_r2': val_r2,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'test_r2': test_r2,
    'best_iteration': best_iteration
}

pd.DataFrame([metrics]).to_csv('xgboost_metrics.csv', index=False)
print("✓ Metrics saved as 'xgboost_metrics.csv'")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("XGBOOST MODEL TRAINING COMPLETE!")
print("="*70)

print(f"\n📊 Final XGBoost Results:")
print(f"  Test Set Performance:")
print(f"    • MAE:  ${test_mae:.2f}")
print(f"    • RMSE: ${test_rmse:.2f}")
print(f"    • R² Score: {test_r2:.4f}")

print(f"\n📁 Files saved:")
print(f"    • xgboost_model.json (trained model)")
print(f"    • xgboost_metrics.csv (performance metrics)")
print(f"    • xgboost_train/val/test_predictions.npy")
print(f"    • xgboost_feature_importance.png")
print(f"    • xgboost_predictions_visualization.png")

print(f"\n🎯 Next Steps:")
print(f"    1. Compare LSTM vs XGBoost results")
print(f"    2. Try ensemble (combine both models)")
print(f"    3. Investigate why test performance is poor")
print(f"    4. Consider using shorter prediction horizon")
print(f"    5. Try rolling window prediction")

print("\n" + "="*70)
