To demonstrate a **Recurrent Neural Network (RNN) for stock price sequence analysis**, we'll use Google's stock price (ticker: `GOOGL`) and build an **LSTM-based model** to predict the future closing prices based on past sequences.

---

### 🧰 Tools Used:

* `yfinance`: for downloading stock data.
* `TensorFlow/Keras`: for building the RNN.
* `matplotlib`: for visualization.

---

## ✅ 1. **Install & Import Libraries**

```bash
pip install yfinance tensorflow matplotlib scikit-learn
```

```python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```

---

## 📥 2. **Load Google Stock Price Data**

```python
# Download GOOGL stock data (last 5 years)
data = yf.download('GOOGL', start='2018-01-01', end='2023-12-31')
closing_prices = data['Close'].values.reshape(-1, 1)

# Normalize to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(closing_prices)
```

---

## 🧱 3. **Prepare Sequences for RNN**

We’ll create a sliding window of sequences, e.g., using 60 days to predict the 61st day.

```python
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i - sequence_length:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, time steps, features)
```

---

## 🧠 4. **Build and Train LSTM Model**

```python
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(units=50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)
```

---

## 🔮 5. **Prediction and Visualization**

```python
# Predict on training data to visualize fit
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Inverse transform real prices
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Plot
plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label="Actual Prices", color='blue')
plt.plot(predicted_prices, label="Predicted Prices", color='red')
plt.title("GOOGL Stock Price Prediction using LSTM")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📊 Optional: Forecast Future Prices

To predict future prices, use the last 60 days as input and loop forward.

```python
future_steps = 30
last_sequence = scaled_prices[-60:].reshape(1, 60, 1)
future_predictions = []

for _ in range(future_steps):
    next_price = model.predict(last_sequence)[0, 0]
    future_predictions.append(next_price)
    last_sequence = np.append(last_sequence[:, 1:, :], [[[next_price]]], axis=1)

# Inverse transform predicted values
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot
plt.plot(range(len(closing_prices)), closing_prices, label="Historical")
plt.plot(range(len(closing_prices), len(closing_prices) + future_steps), future_prices, label="Future Forecast", color='green')
plt.title("GOOGL Future Price Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📌 Summary

| Feature    | Description                   |
| ---------- | ----------------------------- |
| Model Type | LSTM-based RNN                |
| Input      | Past 60 days of prices        |
| Output     | Next day's closing price      |
| Dataset    | GOOGL stock prices from Yahoo |
| Framework  | TensorFlow / Keras            |

---

Would you like to expand this into:

* A **multi-feature model** (e.g., OHLCV)?
* Use **GRU or Transformer** instead?
* Include **technical indicators** (MACD, RSI) for better signals?

Let me know your target analysis depth.
