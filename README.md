# Stock-Price-Prediction-Using-ML
Stock Price Prediction Using LSTM &amp; XGBoost

---

A comprehensive machine learning project for predicting Apple (AAPL) stock prices using deep learning (LSTM) and gradient boosting (XGBoost) models. This project demonstrates the transition from absolute price prediction to percentage-based return prediction, achieving significantly improved results.

## 📊 Project Overview

This project implements and compares multiple approaches to stock price prediction:

- **LSTM Neural Networks**: Deep learning model for sequential time series prediction
- **XGBoost**: Gradient boosting model for tabular feature analysis
- **Returns-Based Prediction**: Improved approach predicting percentage changes instead of absolute prices

### Key Results

| Model | Test MAE | Test RMSE | Test R² Score |
|-------|----------|-----------|---------------|
| LSTM v2 (Absolute) | $31.83 | $35.08 | -1.32 |
| XGBoost (Absolute) | $89.22 | $91.80 | -14.92 |
| **LSTM (Returns-Based)** | **$2.82** | **$4.25** | **0.9738** |

The returns-based LSTM model achieved a **97.38% R² score**, demonstrating strong predictive power.

## 🎯 Features

### Data Collection & Processing
- Historical stock data from 2014-2024 (3000+ trading days)
- 24+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV)
- Macroeconomic indicators from FRED (Federal Funds Rate, GDP, CPI, unemployment, exchange rates)
- Comprehensive data cleaning and preprocessing pipeline

### Technical Indicators
- **Moving Averages**: SMA (5, 10, 20, 50, 200 days), EMA (12, 26 days)
- **Momentum**: RSI, MACD with signal line and histogram
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV (On-Balance Volume)
- **Price Features**: Daily returns, price changes, high-low range

### Model Architectures

#### LSTM (Returns-Based)
```
- Input: 20-day lookback window
- LSTM Layer 1: 64 units with return sequences
- Dropout: 0.2
- LSTM Layer 2: 32 units
- Dropout: 0.2
- Dense Layer: 16 units
- Dropout: 0.2
- Output: 1 unit (predicted return)
- Total Parameters: 39,329
```

#### XGBoost Configuration
- Objective: reg:squarederror
- Max depth: 6
- Learning rate: 0.05
- Subsample: 0.8
- Early stopping: 50 rounds




### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
XGBoost
pandas
numpy
scikit-learn
yfinance
fredapi
matplotlib
seaborn
```





## 📊 Model Performance Analysis

### Training Metrics (Returns-Based LSTM)

| Dataset | MAE (Returns) | MAE (Price) | RMSE | R² Score |
|---------|---------------|-------------|------|----------|
| Training | 1.29% | $1.06 | $1.77 | 0.9989 |
| Validation | 1.03% | $2.03 | $2.80 | 0.9831 |
| Test | 1.26% | $2.82 | $4.25 | 0.9738 |

### Key Insights

- **Positive R² Score**: Model has genuine predictive power (vs. random guessing)
- **Low MAE**: Average prediction error of $2.82 on test set
- **Consistent Performance**: Similar metrics across train/val/test splits
- **No Overfitting**: Small gap between training and test performance

### XGBoost Feature Importance

Top contributing features (by information gain):
1. Feature f1887: 117.33
2. Feature f2172: 108.67
3. Feature f141: 93.42
4. Feature f1630: 78.40
5. Feature f326: 63.49

### ⚠️ Limitations & Disclaimers

Past Performance ≠ Future Results: Historical data cannot predict unexpected events
Market Efficiency: Stock markets are highly efficient; consistent prediction is extremely difficult
External Factors: News, earnings, geopolitical events not captured in technical indicators
Research Purpose: This project is for educational/research purposes only
Not Financial Advice: Do not use for actual trading decisions without professional consultation

### Reality Check

Professional quantitative funds consider 55% directional accuracy a success
Even with ML, predicting stock prices remains challenging
A positive R² score is encouraging but doesn't guarantee profit
Transaction costs, slippage, and market impact not modeled

### 🔮 Future Improvements

 Add sentiment analysis from news/social media
 Implement ensemble methods (LSTM + XGBoost)
 Multi-stock prediction with sector analysis
 Attention mechanisms for LSTM
 Transformer-based models
 Real-time prediction API
 Backtesting framework with trading simulator
 Risk management and position sizing strategies
 Alternative data sources (options flow, insider trading)



 ---


 ### 📧 Contact
Osama Shabbar -  onama.os@gmail.com

Project Link: https://github.com/Os-Shabbar/Stock-Price-Prediction-Using-ML

