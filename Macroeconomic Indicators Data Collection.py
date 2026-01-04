"""
Stock Price Prediction - Macroeconomic Data Collection (Part 2)
This script collects macroeconomic indicators from FRED and merges with stock data
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
FRED_API_KEY = 'df5f94ffe53670ac1c1b814124f262e2'
STOCK_CSV_FILE = r"C:\Users\hp\Downloads\AAPL_stock_data_with_indicators.csv" # From Part 1

# Initialize FRED API
fred = Fred(api_key=FRED_API_KEY)

print("="*60)
print("MACROECONOMIC DATA COLLECTION FROM FRED")
print("="*60)

# ============================================
# STEP 1: Load Stock Data from Part 1
# ============================================
print("\nStep 1: Loading stock data from Part 1...")
try:
    df_stock = pd.read_csv(STOCK_CSV_FILE, index_col=0, parse_dates=True)
    print(f"✓ Loaded stock data: {df_stock.shape}")
    print(f"  Date range: {df_stock.index.min()} to {df_stock.index.max()}")
except FileNotFoundError:
    print(f"✗ Error: File '{STOCK_CSV_FILE}' not found!")
    print("  Please run Part 1 script first to generate stock data.")
    exit()

# ============================================
# STEP 2: Define FRED Series IDs for Economic Indicators
# ============================================
print("\n" + "="*60)
print("Step 2: Defining FRED Economic Indicators")
print("="*60)

# FRED Series IDs for macroeconomic indicators
FRED_SERIES = {
    # Interest Rates
    'FEDFUNDS': 'Federal Funds Rate',  # Monthly
    'DGS10': '10-Year Treasury Rate',  # Daily
    
    # Inflation
    'CPIAUCSL': 'Consumer Price Index (CPI)',  # Monthly
    'CPILFESL': 'Core CPI (excluding food & energy)',  # Monthly
    
    # GDP
    'GDP': 'Gross Domestic Product',  # Quarterly
    'GDPC1': 'Real GDP',  # Quarterly
    
    # Unemployment
    'UNRATE': 'Unemployment Rate',  # Monthly
    'CIVPART': 'Labor Force Participation Rate',  # Monthly
    
    # Exchange Rates
    'DEXUSEU': 'USD to Euro Exchange Rate',  # Daily
    'DEXCHUS': 'China to USD Exchange Rate',  # Daily
    'DEXJPUS': 'Japan to USD Exchange Rate',  # Daily
}

print("\nEconomic indicators to collect:")
for series_id, description in FRED_SERIES.items():
    print(f"  • {series_id}: {description}")

# ============================================
# STEP 3: Download Economic Data from FRED
# ============================================
print("\n" + "="*60)
print("Step 3: Downloading Economic Data from FRED")
print("="*60)

# Get date range from stock data
start_date = df_stock.index.min()
end_date = df_stock.index.max()

# Dictionary to store economic data
economic_data = {}

for series_id, description in FRED_SERIES.items():
    try:
        print(f"\nDownloading {series_id} ({description})...")
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        economic_data[series_id] = data
        print(f"  ✓ Downloaded {len(data)} observations")
        print(f"    Frequency: {data.index.to_series().diff().mode()[0]}")
        print(f"    Date range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"  ✗ Error downloading {series_id}: {e}")
        economic_data[series_id] = None

# ============================================
# STEP 4: Create Economic DataFrame
# ============================================
print("\n" + "="*60)
print("Step 4: Creating Economic DataFrame")
print("="*60)

# Combine all economic series into one DataFrame
df_economic = pd.DataFrame()

for series_id, data in economic_data.items():
    if data is not None:
        df_economic[series_id] = data

print(f"\n✓ Economic DataFrame created: {df_economic.shape}")
print(f"\nEconomic data columns: {df_economic.columns.tolist()}")

# Display sample
print("\nSample of economic data:")
print(df_economic.head())

# ============================================
# STEP 5: Ensure DatetimeIndex for Economic Data
# ============================================
print("\n" + "="*60)
print("Step 5: Setting DatetimeIndex")
print("="*60)

# Make sure the index is a DatetimeIndex
if not isinstance(df_economic.index, pd.DatetimeIndex):
    print("⚠ Converting index to DatetimeIndex...")
    df_economic.index = pd.to_datetime(df_economic.index)
    print("✓ Index converted to DatetimeIndex")
else:
    print("✓ Index is already DatetimeIndex")

print(f"Index type: {type(df_economic.index)}")

# ============================================
# STEP 6: Handle Missing Values in Economic Data
# ============================================
print("\n" + "="*60)
print("Step 6: Handling Missing Values")
print("="*60)

print("\nMissing values in economic data:")
print(df_economic.isnull().sum())

# Forward fill missing values (common for economic data)
df_economic_filled = df_economic.ffill()
print("\n✓ Forward-filled missing values")

# ============================================
# STEP 7: Resample Economic Data to Daily Frequency
# ============================================
print("\n" + "="*60)
print("Step 7: Resampling to Daily Frequency")
print("="*60)

print("\nResampling economic data to match daily stock data...")
print("(Monthly/Quarterly data will be forward-filled to daily)")

# Resample to daily and forward fill
df_economic_daily = df_economic_filled.resample('D').ffill()

print(f"✓ Resampled to daily: {df_economic_daily.shape}")
print(f"  Date range: {df_economic_daily.index.min()} to {df_economic_daily.index.max()}")

# ============================================
# STEP 8: Calculate Economic Derived Features
# ============================================
print("\n" + "="*60)
print("Step 8: Calculating Economic Derived Features")
print("="*60)

# Calculate inflation rate (month-over-month change in CPI)
if 'CPIAUCSL' in df_economic_daily.columns:
    df_economic_daily['Inflation_Rate_MoM'] = df_economic_daily['CPIAUCSL'].pct_change(periods=30) * 100
    print("✓ Calculated Month-over-Month Inflation Rate")

# Calculate GDP growth rate (quarter-over-quarter)
if 'GDP' in df_economic_daily.columns:
    df_economic_daily['GDP_Growth_QoQ'] = df_economic_daily['GDP'].pct_change(periods=90) * 100
    print("✓ Calculated Quarter-over-Quarter GDP Growth")

# Calculate interest rate spread (10Y - Fed Funds)
if 'DGS10' in df_economic_daily.columns and 'FEDFUNDS' in df_economic_daily.columns:
    df_economic_daily['Interest_Rate_Spread'] = df_economic_daily['DGS10'] - df_economic_daily['FEDFUNDS']
    print("✓ Calculated Interest Rate Spread")

print(f"\nFinal economic features: {len(df_economic_daily.columns)}")

# ============================================
# STEP 9: Merge Stock Data with Economic Data
# ============================================
print("\n" + "="*60)
print("Step 9: Merging Stock and Economic Data")
print("="*60)

print("\nMerging datasets...")
print(f"  Stock data shape: {df_stock.shape}")
print(f"  Economic data shape: {df_economic_daily.shape}")

# Merge on date index (left join to keep all stock dates)
df_merged = df_stock.join(df_economic_daily, how='left')

print(f"\n✓ Merged dataset shape: {df_merged.shape}")

# Forward fill any remaining NaN values from the merge
df_merged = df_merged.ffill().bfill()

print(f"✓ Filled remaining missing values")

# ============================================
# STEP 10: Final Data Validation
# ============================================
print("\n" + "="*60)
print("Step 10: Final Data Validation")
print("="*60)

print(f"\nFinal Dataset Information:")
print(f"  Shape: {df_merged.shape}")
print(f"  Features: {len(df_merged.columns)}")
print(f"  Date range: {df_merged.index.min()} to {df_merged.index.max()}")
print(f"  Total rows: {len(df_merged)}")

print(f"\nMissing values check:")
missing_summary = df_merged.isnull().sum()
if missing_summary.sum() == 0:
    print("  ✓ No missing values!")
else:
    print(f"  ⚠ Missing values found:")
    print(missing_summary[missing_summary > 0])

print(f"\nAll feature columns ({len(df_merged.columns)}):")
for i, col in enumerate(df_merged.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================
# STEP 11: Display Sample Data
# ============================================
print("\n" + "="*60)
print("Step 11: Sample Data Preview")
print("="*60)

print("\nFirst 5 rows:")
print(df_merged.head())

print("\nLast 5 rows:")
print(df_merged.tail())

# Display statistics for key columns
key_columns = ['Close', 'RSI', 'MACD', 'FEDFUNDS', 'UNRATE', 'GDP']
available_key_cols = [col for col in key_columns if col in df_merged.columns]

if available_key_cols:
    print(f"\nStatistics for key features:")
    print(df_merged[available_key_cols].describe())

# ============================================
# STEP 12: Save Final Dataset
# ============================================
print("\n" + "="*60)
print("Step 12: Saving Final Dataset")
print("="*60)

output_filename = 'AAPL_complete_data_with_economics.csv'
df_merged.to_csv(output_filename)
print(f"✓ Complete dataset saved to '{output_filename}'")

# Also save feature names for later use
feature_names = df_merged.columns.tolist()
feature_df = pd.DataFrame({'Feature': feature_names})
feature_df.to_csv('feature_names.csv', index=False)
print(f"✓ Feature names saved to 'feature_names.csv'")

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE!")
print("="*60)

print(f"\n📊 Summary:")
print(f"  • Total features: {len(df_merged.columns)}")
print(f"  • Stock technical indicators: ~30")
print(f"  • Economic indicators: {len(FRED_SERIES)}")
print(f"  • Derived economic features: 3")
print(f"  • Total data points: {len(df_merged):,}")

print(f"\n✅ Your dataset is ready for ML modeling!")
print(f"\n📁 Output files:")
print(f"  • {output_filename}")
print(f"  • feature_names.csv")

print(f"\n🚀 Next steps:")
print(f"  1. Normalize/scale the features")
print(f"  2. Create sequences for LSTM (time windows)")
print(f"  3. Split into train/validation/test sets")
print(f"  4. Build and train LSTM and XGBoost models")
