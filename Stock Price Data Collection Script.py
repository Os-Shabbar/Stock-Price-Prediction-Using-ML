"""
Stock Price Prediction - Data Collection Script (Part 1)
This script collects stock data and calculates technical indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
STOCK_SYMBOL = 'AAPL'  # Change this to your desired stock
START_DATE = '2014-01-01'  # 10 years of data
END_DATE = datetime.now().strftime('%Y-%m-%d')

print(f"Collecting data for {STOCK_SYMBOL} from {START_DATE} to {END_DATE}...")

# ============================================
# STEP 1: Download Stock Data (OHLCV)
# ============================================
def download_stock_data(symbol, start_date, end_date):
    """Download historical stock data from Yahoo Finance"""
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        print(f"✓ Downloaded {len(stock_data)} days of stock data")
        return stock_data
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return None

# Download the data
df = download_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)

# Display basic info
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names: {df.columns.tolist()}")

# ============================================
# STEP 2: Calculate Technical Indicators
# ============================================

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_atr(data, period=14):
    """Calculate Average True Range (volatility indicator)"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_obv(data):
    """Calculate On-Balance Volume"""
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

print("\n" + "="*50)
print("Calculating Technical Indicators...")
print("="*50)

# Calculate all technical indicators
df['SMA_5'] = calculate_sma(df, 5)
df['SMA_10'] = calculate_sma(df, 10)
df['SMA_20'] = calculate_sma(df, 20)
df['SMA_50'] = calculate_sma(df, 50)
df['SMA_200'] = calculate_sma(df, 200)
print("✓ SMAs calculated (5, 10, 20, 50, 200)")

df['EMA_12'] = calculate_ema(df, 12)
df['EMA_26'] = calculate_ema(df, 26)
print("✓ EMAs calculated (12, 26)")

df['RSI'] = calculate_rsi(df, 14)
print("✓ RSI calculated")

df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df)
print("✓ MACD calculated")

df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df)
print("✓ Bollinger Bands calculated")

df['ATR'] = calculate_atr(df)
print("✓ ATR calculated")

df['OBV'] = calculate_obv(df)
print("✓ OBV calculated")

# ============================================
# STEP 3: Add Price Change Features
# ============================================
print("\nCalculating price change features...")
df['Daily_Return'] = df['Close'].pct_change()
df['Price_Change'] = df['Close'].diff()
df['High_Low_Range'] = df['High'] - df['Low']
print("✓ Price change features calculated")

# ============================================
# STEP 4: Clean the Data
# ============================================
print("\n" + "="*50)
print("Cleaning Data...")
print("="*50)

# Check for missing values
print(f"\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop rows with NaN values (mainly from initial calculations)
df_cleaned = df.dropna()
print(f"\n✓ Dropped {len(df) - len(df_cleaned)} rows with missing values")
print(f"Final dataset shape: {df_cleaned.shape}")

# ============================================
# STEP 5: Display Final Dataset
# ============================================
print("\n" + "="*50)
print("Final Dataset Preview")
print("="*50)
print(f"\nTotal features: {len(df_cleaned.columns)}")
print(f"\nFeature names:")
for i, col in enumerate(df_cleaned.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\nFirst 5 rows of processed data:")
print(df_cleaned.head())

print(f"\nLast 5 rows of processed data:")
print(df_cleaned.tail())

print(f"\nBasic Statistics:")
print(df_cleaned[['Close', 'RSI', 'MACD', 'Volume']].describe())

# ============================================
# STEP 6: Save to CSV (Optional)
# ============================================
output_filename = rf"C:\Users\hp\Downloads\{STOCK_SYMBOL}_stock_data_with_indicators.csv"

df_cleaned.to_csv(output_filename)
print(f"\n✓ Data saved to '{output_filename}'")

print("\n" + "="*50)
print("Data Collection Complete!")
print("="*50)
print(f"\nNext steps:")
print("1. Add macroeconomic indicators (Interest rates, Inflation, GDP, etc.)")
print("2. Normalize/scale the data")
print("3. Create sequences for LSTM")
print("4. Train the models")
