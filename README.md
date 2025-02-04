# 📈 Stock Price Prediction with XGBoost

This project predicts Tesla stock's closing price using machine learning techniques, particularly the **XGBoost** algorithm. The model is trained on historical stock data with technical indicators like **RSI, MACD, and Bollinger Bands** to improve predictions.

---

## 🔧 Requirements & Dependencies

Before running the project, install the required dependencies:

```bash
pip install -r requirements.txt
```
Or manually install:

```bash
pip install numpy pandas yfinance xgboost scikit-learn ta matplotlib
```



🚀 Running the Project

1️⃣ Clone the Repository
```bash
git clone https://github.com/0BerkayK/Dataguess-Case
cd Dataguess-Case
```
```bash
2️⃣ Run the Main Script
python main.py
```

📊 Model Performance
Metric	Value
RMSE (Root Mean Squared Error)	9.29
MAPE (Mean Absolute Percentage Error)	3.37%
R² (Coefficient of Determination)	0.9876
🚀 The model accurately predicts stock prices with minimal error.
Future improvements can include feature engineering and hyperparameter tuning.
