"""
Stock Price Prediction - LSTM Model Building and Training (Part 4A)
This script builds and trains the LSTM model Usning Stock Return
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMPROVED APPROACH: PERCENTAGE-BASED PREDICTION")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================
LOOKBACK_WINDOW = 20  # Shorter window (20 days instead of 60)
TRAIN_SPLIT = 0.8     # Use more recent data (80% train, 10% val, 10% test)
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

print("\n🔧 Key Improvements:")
print("  1. Predict % CHANGE instead of absolute price")
print("  2. Shorter lookback window (20 days vs 60)")
print("  3. More recent training data (80/10/10 split)")
print("  4. Use returns (naturally scaled)")

# ============================================
# STEP 1: Load and Prepare Data
# ============================================
print("\n" + "="*70)
print("Step 1: Loading and Transforming Data")
print("-" * 70)

df = pd.read_csv(r"C:\Users\hp\Downloads\AAPL_complete_data_with_economics.csv", 
                 index_col=0, parse_dates=True)

print(f"✓ Original data loaded: {df.shape}")

# ============================================
# STEP 2: Calculate Returns (Percentage Changes)
# ============================================
print("\n" + "="*70)
print("Step 2: Converting to Percentage Returns")
print("-" * 70)

# Calculate returns for all numeric columns
returns_df = df.copy()

# For price columns, calculate percentage change
price_cols = ['Close', 'Open', 'High', 'Low']
for col in price_cols:
    if col in returns_df.columns:
        returns_df[f'{col}_Return'] = returns_df[col].pct_change()

# Keep technical indicators as-is (they're already normalized)
# Keep economic indicators as-is

# Target: Next day's return
returns_df['Target_Return'] = returns_df['Close'].pct_change().shift(-1)

# Drop NaN values
returns_df = returns_df.dropna()

print(f"✓ Returns calculated")
print(f"  Data shape after transformation: {returns_df.shape}")
print(f"\nTarget Statistics (Daily Returns):")
print(f"  Mean: {returns_df['Target_Return'].mean()*100:.4f}%")
print(f"  Std:  {returns_df['Target_Return'].std()*100:.4f}%")
print(f"  Min:  {returns_df['Target_Return'].min()*100:.2f}%")
print(f"  Max:  {returns_df['Target_Return'].max()*100:.2f}%")

# ============================================
# STEP 3: Feature Selection
# ============================================
print("\n" + "="*70)
print("Step 3: Feature Selection")
print("-" * 70)

# Select features (exclude original prices, keep returns and indicators)
feature_cols = [col for col in returns_df.columns 
                if col not in ['Close', 'Open', 'High', 'Low', 'Target_Return']]

print(f"✓ Selected {len(feature_cols)} features")

# ============================================
# STEP 4: Split Data (More Recent Training)
# ============================================
print("\n" + "="*70)
print("Step 4: Splitting Data (80/10/10)")
print("-" * 70)

total_samples = len(returns_df)
train_size = int(total_samples * TRAIN_SPLIT)
val_size = int(total_samples * VAL_SPLIT)

train_data = returns_df.iloc[:train_size]
val_data = returns_df.iloc[train_size:train_size + val_size]
test_data = returns_df.iloc[train_size + val_size:]

print(f"\nData Split:")
print(f"  Training:   {len(train_data):4d} samples | {train_data.index.min()} to {train_data.index.max()}")
print(f"  Validation: {len(val_data):4d} samples | {val_data.index.min()} to {val_data.index.max()}")
print(f"  Test:       {len(test_data):4d} samples | {test_data.index.min()} to {test_data.index.max()}")

# ============================================
# STEP 5: Standardize Features
# ============================================
print("\n" + "="*70)
print("Step 5: Standardizing Features")
print("-" * 70)

scaler = StandardScaler()
scaler.fit(train_data[feature_cols])

train_features = scaler.transform(train_data[feature_cols])
val_features = scaler.transform(val_data[feature_cols])
test_features = scaler.transform(test_data[feature_cols])

train_target = train_data['Target_Return'].values
val_target = val_data['Target_Return'].values
test_target = test_data['Target_Return'].values

print("✓ Features standardized (mean=0, std=1)")

# ============================================
# STEP 6: Create Sequences
# ============================================
print("\n" + "="*70)
print("Step 6: Creating Sequences")
print("-" * 70)

def create_sequences(features, target, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

X_train_seq, y_train_seq = create_sequences(train_features, train_target, LOOKBACK_WINDOW)
X_val_seq, y_val_seq = create_sequences(val_features, val_target, LOOKBACK_WINDOW)
X_test_seq, y_test_seq = create_sequences(test_features, test_target, LOOKBACK_WINDOW)

print(f"✓ Sequences created with lookback={LOOKBACK_WINDOW}")
print(f"  X_train: {X_train_seq.shape}")
print(f"  X_val:   {X_val_seq.shape}")
print(f"  X_test:  {X_test_seq.shape}")

# ============================================
# STEP 7: Build Simple LSTM Model
# ============================================
print("\n" + "="*70)
print("Step 7: Building LSTM Model (Returns-Based)")
print("-" * 70)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK_WINDOW, len(feature_cols))),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # Predicts return (can be positive or negative)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("✓ LSTM model built")
model.summary()

# ============================================
# STEP 8: Train LSTM
# ============================================
print("\n" + "="*70)
print("Step 8: Training LSTM Model")
print("-" * 70)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n✓ LSTM training completed!")

# ============================================
# STEP 9: Make Predictions and Convert Back to Prices
# ============================================
print("\n" + "="*70)
print("Step 9: Making Predictions and Converting to Prices")
print("-" * 70)

# Predict returns
y_train_pred_return = model.predict(X_train_seq, verbose=0).flatten()
y_val_pred_return = model.predict(X_val_seq, verbose=0).flatten()
y_test_pred_return = model.predict(X_test_seq, verbose=0).flatten()

print("✓ Return predictions generated")

# Convert returns back to actual prices
def returns_to_prices(returns_pred, initial_prices):
    """Convert predicted returns to actual prices"""
    prices = [initial_prices[0]]
    for i, ret in enumerate(returns_pred):
        next_price = prices[-1] * (1 + ret)
        prices.append(next_price)
    return np.array(prices[1:])  # Remove initial price

# Get initial prices for each sequence
train_initial_prices = train_data['Close'].iloc[LOOKBACK_WINDOW-1:-1].values
val_initial_prices = val_data['Close'].iloc[LOOKBACK_WINDOW-1:-1].values
test_initial_prices = test_data['Close'].iloc[LOOKBACK_WINDOW-1:-1].values

# Convert predicted returns to prices
y_train_pred_prices = train_initial_prices * (1 + y_train_pred_return)
y_val_pred_prices = val_initial_prices * (1 + y_val_pred_return)
y_test_pred_prices = test_initial_prices * (1 + y_test_pred_return)

# Get actual prices
y_train_actual_prices = train_data['Close'].iloc[LOOKBACK_WINDOW:].values
y_val_actual_prices = val_data['Close'].iloc[LOOKBACK_WINDOW:].values
y_test_actual_prices = test_data['Close'].iloc[LOOKBACK_WINDOW:].values

print("✓ Converted returns back to prices")

# ============================================
# STEP 10: Evaluate Performance
# ============================================
print("\n" + "="*70)
print("📊 IMPROVED MODEL PERFORMANCE (Returns-Based)")
print("="*70)

# Metrics on returns
train_mae_ret = mean_absolute_error(y_train_seq, y_train_pred_return)
val_mae_ret = mean_absolute_error(y_val_seq, y_val_pred_return)
test_mae_ret = mean_absolute_error(y_test_seq, y_test_pred_return)

print("\nPerformance on Returns:")
print(f"  Training MAE:   {train_mae_ret*100:.4f}%")
print(f"  Validation MAE: {val_mae_ret*100:.4f}%")
print(f"  Test MAE:       {test_mae_ret*100:.4f}%")

# Metrics on prices
train_mae_price = mean_absolute_error(y_train_actual_prices, y_train_pred_prices)
train_rmse_price = np.sqrt(mean_squared_error(y_train_actual_prices, y_train_pred_prices))
train_r2_price = r2_score(y_train_actual_prices, y_train_pred_prices)

val_mae_price = mean_absolute_error(y_val_actual_prices, y_val_pred_prices)
val_rmse_price = np.sqrt(mean_squared_error(y_val_actual_prices, y_val_pred_prices))
val_r2_price = r2_score(y_val_actual_prices, y_val_pred_prices)

test_mae_price = mean_absolute_error(y_test_actual_prices, y_test_pred_prices)
test_rmse_price = np.sqrt(mean_squared_error(y_test_actual_prices, y_test_pred_prices))
test_r2_price = r2_score(y_test_actual_prices, y_test_pred_prices)

print("\nPerformance on Prices:")
print("-" * 70)
print(f"Training Set:")
print(f"  • MAE:  ${train_mae_price:.2f}")
print(f"  • RMSE: ${train_rmse_price:.2f}")
print(f"  • R² Score: {train_r2_price:.4f}")

print(f"\nValidation Set:")
print(f"  • MAE:  ${val_mae_price:.2f}")
print(f"  • RMSE: ${val_rmse_price:.2f}")
print(f"  • R² Score: {val_r2_price:.4f}")

print(f"\nTest Set:")
print(f"  • MAE:  ${test_mae_price:.2f}")
print(f"  • RMSE: ${test_rmse_price:.2f}")
print(f"  • R² Score: {test_r2_price:.4f}")

# ============================================
# STEP 11: Compare with Previous Approach
# ============================================
print("\n" + "="*70)
print("📈 COMPARISON WITH PREVIOUS MODELS")
print("="*70)

print("\nPrevious Results (Absolute Price Prediction):")
print("  • LSTM v2:  MAE=$31.83, R²=-1.32")
print("  • XGBoost:  MAE=$89.22, R²=-14.92")

print(f"\nNew Results (Returns-Based Prediction):")
print(f"  • Returns LSTM: MAE=${test_mae_price:.2f}, R²={test_r2_price:.4f}")

if test_r2_price > -1.32:
    print("\n✅ SIGNIFICANT IMPROVEMENT!")
    if test_r2_price > 0:
        print("   R² is now POSITIVE - model has predictive power!")
else:
    print("\n⚠ Still needs work, but addressing the right problem")

# ============================================
# STEP 12: Visualize Results
# ============================================
print("\n" + "="*70)
print("Step 12: Creating Visualizations")
print("-" * 70)

test_dates = test_data.index[LOOKBACK_WINDOW:]

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Predictions vs Actual
axes[0].plot(test_dates, y_test_actual_prices, label='Actual Price', 
            linewidth=2, color='black', alpha=0.8)
axes[0].plot(test_dates, y_test_pred_prices, label='Returns-Based LSTM Prediction', 
            linewidth=2, alpha=0.7, color='green')
axes[0].set_title('Returns-Based Prediction: Test Set', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Prediction errors
errors = y_test_actual_prices - y_test_pred_prices
axes[1].plot(test_dates, errors, linewidth=2, color='red', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Prediction Errors', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Error ($)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Return predictions
axes[2].scatter(y_test_seq*100, y_test_pred_return*100, alpha=0.5)
axes[2].plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Perfect Prediction')
axes[2].set_title('Predicted vs Actual Returns', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Actual Return (%)')
axes[2].set_ylabel('Predicted Return (%)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('returns_based_prediction.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'returns_based_prediction.png'")
plt.show()

# ============================================
# STEP 13: Save Model and Components
# ============================================
print("\n" + "="*70)
print("Step 13: Saving Model and Components")
print("-" * 70)

# Save the trained model
model.save('returns_based_lstm_model.keras')  # Use .keras format (modern)
print("✓ Model saved as 'returns_based_lstm_model.keras'")

# Also save just the weights (more portable)
model.save_weights('returns_based_lstm.weights.h5')
print("✓ Model weights saved as 'returns_based_lstm.weights.h5'")

# Save model architecture as JSON
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)
print("✓ Model architecture saved as 'model_architecture.json'")

# Save the scaler
import joblib
joblib.dump(scaler, 'returns_scaler.pkl')
print("✓ Scaler saved as 'returns_scaler.pkl'")

# Save feature column names
import json
with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)
print("✓ Feature columns saved as 'feature_columns.json'")

# Save configuration
config = {
    'lookback_window': LOOKBACK_WINDOW,
    'train_split': TRAIN_SPLIT,
    'val_split': VAL_SPLIT,
    'test_split': TEST_SPLIT
}
with open('model_config.json', 'w') as f:
    json.dump(config, f)
print("✓ Configuration saved as 'model_config.json'")

print("\n📁 All components saved for deployment!")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("🎯 KEY TAKEAWAYS")
print("="*70)

print("\n✅ What We Fixed:")
print("  1. Changed from absolute prices → percentage returns")
print("  2. Reduced lookback window (60 → 20 days)")
print("  3. Used more recent training data (80/10/10 split)")
print("  4. Eliminated extrapolation problem")

print(f"\n📊 Results:")
print(f"  Test R² Score: {test_r2_price:.4f}")
if test_r2_price > 0:
    print("  ✅ Model has predictive power!")
elif test_r2_price > -1:
    print("  ⚠ Improved but still challenging")
else:
    print("  ❌ Still struggling - stock prediction is inherently difficult")

print("\n💡 Reality Check:")
print("  Stock prices are extremely hard to predict, even with ML.")
print("  A positive R² score means the model is better than random guessing.")
print("  Professional quant funds consider 55% accuracy a success!")

print("\n" + "="*70)
