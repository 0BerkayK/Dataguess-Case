import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np


# 1. Data:
  # - Use publicly available stock market data (e.g., Yahoo Finance, Alpha Vantage API, or Kaggle datasets).
  #- Include at least the following features: Open, High, Low, Close, Volume.

# Yahoo Finance Tesla, Inc. (TSLA) Dataset

data = yf.download("TSLA", start="2020-02-03", end="2025-02-03")

print(data.isnull().sum()) # Missing value check

print(data) # Data Preview

#2. Feature Engineering:
  # - Incorporate technical indicators such as RSI, MACD, and Bollinger Bands into your feature set.
  # - Add additional features that you believe could improve model performance.

# RSI

window_length = 14  # A 14-day window is usually used for RSI

# Daily change of closing price
delta = data['Close'].diff()

# Positive and Negative changes
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# Average Gain and Loss
avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
avg_loss = loss.rolling(window=window_length, min_periods=1).mean()

# RSI Calculation
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

print(data['RSI'])

plt.figure(figsize=(12,6))
plt.plot(data.index, data['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
plt.xlabel("Tarih")
plt.ylabel("RSI Value")
plt.title("Tesla RSI Graph")
plt.legend()
plt.show()

# MCAD Hesaplama

short_window = 12
long_window = 26
signal_window = 9

data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()

data['MACD'] = data['EMA12'] - data['EMA26']
data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

plt.figure(figsize=(14, 7))

# Subplot for MACD and Signal line
plt.subplot(2, 1, 1)
plt.plot(data.index, data['MACD'], label='MACD', color='blue', alpha=0.7)
plt.plot(data.index, data['MACD_Signal'], label='MACD Signal', color='red', linestyle='--', alpha=0.7)
plt.title('MACD and MACD Signal')
plt.legend()

# Subplot for MACD Histogram
plt.subplot(2, 1, 2)
plt.bar(data.index, data['MACD'] - data['MACD_Signal'],
        label='MACD Histogram', color='gray', alpha=0.5)
plt.title('MACD Histogram')
plt.legend()

# Layout adjustment and show plot
plt.tight_layout()
plt.show()


# Bollinger Bantları

window = 20
data['SMA20'] = data['Close'].rolling(window=window).mean()
data['STD20'] = data['Close'].rolling(window=window).std()

data['BB_High'] = data['SMA20'] + (data['STD20'] * 2)
data['BB_Low'] = data['SMA20'] - (data['STD20'] * 2)


#Visualize Tesla closing prices, SMA20, upper and lower Bollinger Bands
plt.figure(figsize=(12,6))

# Plot the closing price
plt.plot(data['Close'], label='Tesla Close', color='blue', linewidth=1)

# Drawing SMA20
plt.plot(data['SMA20'], label='SMA20', color='orange', linestyle='--', linewidth=1)

# Drawing the upper Bollinger Band
plt.plot(data['BB_High'], label='Upper Bollinger Band', color='green', linestyle='--', linewidth=1)

# Drawing lower Bollinger Band
plt.plot(data['BB_Low'], label='Lower Bollinger Band', color='red', linestyle='--', linewidth=1)

# Title of graph chart
plt.title('Tesla Stock Price and Bollinger Bands')

#X and Y axes
plt.xlabel('Date')
plt.ylabel('Price (USD)')

plt.legend()

# Rotate date labels and make them visible
plt.xticks(rotation=45)

# Graph show
plt.tight_layout()
plt.show()

# 3. Model Development:
  #  - Choose an appropriate predictive modeling approach (e.g., regression, LSTM, or another deep learning model).
  #  - Train the model to predict the next day's closing price of a stock.


# Determine Properties and Target Variable
X = data[['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'BB_High', 'BB_Low']]
y = data['Close'].shift(-1)  # To predict the next day's closing price

# Clean NaN values from last line
X = X[:-1]
y = y[:-1]

# Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)


# 4. Evaluation:
#    - Use appropriate evaluation metrics (e.g., RMSE, MAPE, or R-squared) to assess your model's performance.
#    - Provide a detailed explanation of why you selected the chosen metric(s).

# RMSE Error (Root Mean Square Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost Model RMSE: {rmse:.2f}")

# MAPE Error (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"XGBoost Model MAPE: {mape:.2%}")

# R-Square Score (Explanatory Score)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost Model R² Skoru: {r2:.4f}")


plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Real Price", color="blue", linestyle="dashed")
plt.plot(y_pred, label="Predicted Price", color="red")
plt.xlabel("Day")
plt.ylabel("Price (USD)")
plt.title("XGBoost Model - Real and Predicted Prices")
plt.legend()
plt.show()

