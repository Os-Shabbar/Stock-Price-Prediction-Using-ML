"""
Stock Price Prediction - Data Preprocessing & Preparation (Part 3)
This script prepares the data for LSTM and XGBoost models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
DATA_FILE = r"C:\Users\hp\Downloads\AAPL_complete_data_with_economics.csv"
LOOKBACK_WINDOW = 60  # Number of days to look back for prediction
TRAIN_SPLIT = 0.7     # 70% training
VAL_SPLIT = 0.15      # 15% validation
TEST_SPLIT = 0.15     # 15% testing

print("="*70)
print("STOCK PREDICTION - DATA PREPROCESSING")
print("="*70)



df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)


# Target variable (what we want to predict)
TARGET = 'Close'

# Features to use for prediction (all columns except target)
feature_columns = [col for col in df.columns if col != TARGET]

for i, feature in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {feature}")


missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    print("⚠ Missing values detected:")
    print(missing_values[missing_values > 0])
    print("\nFilling missing values with forward fill...")
    df = df.ffill().bfill()
    print("✓ Missing values handled")

# Check for infinite values
inf_values = np.isinf(df.select_dtypes(include=[np.number])).sum()
if inf_values.sum() == 0:
    print("✓ No infinite values found!")
else:
    print("⚠ Infinite values detected:")
    print(inf_values[inf_values > 0])
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    print("✓ Infinite values handled")


# Calculate split indices
total_samples = len(df)
train_size = int(total_samples * TRAIN_SPLIT)
val_size = int(total_samples * VAL_SPLIT)
test_size = total_samples - train_size - val_size

# Split the data chronologically (important for time series!)
train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size + val_size]
test_data = df.iloc[train_size + val_size:]




# Initialize scalers
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit scalers on training data only (to prevent data leakage!)
print("\n✓ Fitting scalers on training data...")
feature_scaler.fit(train_data[feature_columns])
target_scaler.fit(train_data[[TARGET]])

# Transform all datasets
train_features_scaled = feature_scaler.transform(train_data[feature_columns])
train_target_scaled = target_scaler.transform(train_data[[TARGET]])

val_features_scaled = feature_scaler.transform(val_data[feature_columns])
val_target_scaled = target_scaler.transform(val_data[[TARGET]])

test_features_scaled = feature_scaler.transform(test_data[feature_columns])
test_target_scaled = target_scaler.transform(test_data[[TARGET]])




def create_sequences(features, target, lookback):
    """
    Create sequences for LSTM
    Uses 'lookback' days to predict the next day
    """
    X, y = [], []
    
    for i in range(lookback, len(features)):
        # Take 'lookback' days of features
        X.append(features[i-lookback:i])
        # Predict the next day's target
        y.append(target[i])
    
    return np.array(X), np.array(y)


# Create sequences for training set
X_train_seq, y_train_seq = create_sequences(
    train_features_scaled, 
    train_target_scaled, 
    LOOKBACK_WINDOW
)

# Create sequences for validation set
X_val_seq, y_val_seq = create_sequences(
    val_features_scaled, 
    val_target_scaled, 
    LOOKBACK_WINDOW
)

# Create sequences for test set
X_test_seq, y_test_seq = create_sequences(
    test_features_scaled, 
    test_target_scaled, 
    LOOKBACK_WINDOW
)




def create_xgboost_features(features, target, lookback):
    """
    Create flattened features for XGBoost
    XGBoost needs 2D array (samples, features) instead of 3D
    """
    X, y = [], []
    
    for i in range(lookback, len(features)):
        # Flatten the lookback window into a single row
        flattened_features = features[i-lookback:i].flatten()
        X.append(flattened_features)
        y.append(target[i])
    
    return np.array(X), np.array(y)



# Create XGBoost datasets
X_train_xgb, y_train_xgb = create_xgboost_features(
    train_features_scaled, 
    train_target_scaled.flatten(), 
    LOOKBACK_WINDOW
)

X_val_xgb, y_val_xgb = create_xgboost_features(
    val_features_scaled, 
    val_target_scaled.flatten(), 
    LOOKBACK_WINDOW
)

X_test_xgb, y_test_xgb = create_xgboost_features(
    test_features_scaled, 
    test_target_scaled.flatten(), 
    LOOKBACK_WINDOW
)



# Save LSTM sequences
np.save('X_train_lstm.npy', X_train_seq)
np.save('y_train_lstm.npy', y_train_seq)
np.save('X_val_lstm.npy', X_val_seq)
np.save('y_val_lstm.npy', y_val_seq)
np.save('X_test_lstm.npy', X_test_seq)
np.save('y_test_lstm.npy', y_test_seq)
print("✓ LSTM sequences saved")

# Save XGBoost data
np.save('X_train_xgb.npy', X_train_xgb)
np.save('y_train_xgb.npy', y_train_xgb)
np.save('X_val_xgb.npy', X_val_xgb)
np.save('y_val_xgb.npy', y_val_xgb)
np.save('X_test_xgb.npy', X_test_xgb)
np.save('y_test_xgb.npy', y_test_xgb)
print("✓ XGBoost data saved")

# Save scalers for later use (important for inverse transformation!)
import joblib
joblib.dump(feature_scaler, 'feature_scaler.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl')
print("✓ Scalers saved")

# Save test dates for plotting later
test_dates = test_data.index[LOOKBACK_WINDOW:]
np.save('test_dates.npy', test_dates)
print("✓ Test dates saved")



# Plot target variable distribution across splits
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Close price over time
axes[0, 0].plot(train_data.index, train_data[TARGET], label='Train', alpha=0.7)
axes[0, 0].plot(val_data.index, val_data[TARGET], label='Validation', alpha=0.7)
axes[0, 0].plot(test_data.index, test_data[TARGET], label='Test', alpha=0.7)
axes[0, 0].set_title('Stock Close Price - Train/Val/Test Split')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Close Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training data close price
axes[0, 1].plot(train_data.index, train_data[TARGET])
axes[0, 1].set_title('Training Data - Close Price')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Close Price ($)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of scaled values
axes[1, 0].hist(train_target_scaled, bins=50, alpha=0.7, label='Scaled')
axes[1, 0].set_title('Distribution of Scaled Target Values')
axes[1, 0].set_xlabel('Scaled Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Sample size comparison
split_names = ['Train', 'Validation', 'Test']
split_sizes = [len(X_train_seq), len(X_val_seq), len(X_test_seq)]
axes[1, 1].bar(split_names, split_sizes, color=['blue', 'orange', 'green'], alpha=0.7)
axes[1, 1].set_title('Dataset Split Sizes (After Sequencing)')
axes[1, 1].set_ylabel('Number of Samples')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data_preprocessing_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'data_preprocessing_visualization.png'")
plt.show()

print("\n" + "="*70)
print("All preprocessing steps completed successfully!")
print("="*70)
