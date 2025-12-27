# ================== IMPORTANT ==================
# This file is ready for Streamlit Cloud deployment
# File name MUST be: app.py
# ===============================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import ta

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Real-Time Stock Price Prediction using LSTM")
st.write(
    "This application predicts the **next trading day's stock price** using "
    "a deep learning LSTM model trained on **real-time market data**."
)

# ---------------- Sidebar ----------------
st.sidebar.header("Stock Selection")

stock = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL",
    help="Examples: AAPL, TSLA, MSFT, TCS.NS, INFY.NS"
)

years = st.sidebar.slider(
    "Select Years of Historical Data",
    min_value=1,
    max_value=10,
    value=5
)

# ---------------- Load Data ----------------
@st.cache_data
def load_data(ticker, years):
    data = yf.download(ticker, period=f"{years}y", interval="1d")
    return data

data = load_data(stock, years)

if data.empty:
    st.error("❌ Invalid stock ticker or no data available.")
    st.stop()

st.subheader("📊 Latest Stock Data")
st.dataframe(data.tail())

# ---------------- Feature Engineering ----------------
close_1d = pd.Series(data["Close"].values.flatten(), index=data.index)

data["SMA_20"] = ta.trend.sma_indicator(close=close_1d, window=20)
data["EMA_20"] = ta.trend.ema_indicator(close=close_1d, window=20)
data["RSI"] = ta.momentum.rsi(close=close_1d, window=14)

data.dropna(inplace=True)

features = ["Close", "Volume", "SMA_20", "EMA_20", "RSI"]
dataset = data[features]

# ---------------- Scaling ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# ---------------- Train-Test Split ----------------
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60 :]

# ---------------- Create Sequences ----------------
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60 : i])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# ---------------- Build LSTM Model ----------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(60, x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

early_stop = EarlyStopping(monitor="loss", patience=3)

with st.spinner("⏳ Training LSTM model..."):
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

st.success("✅ Model trained successfully!")

# ---------------- Test Prediction ----------------
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60 : i])

x_test = np.array(x_test)

predictions = model.predict(x_test)

close_scaler = MinMaxScaler()
close_scaler.fit(dataset[["Close"]])
predictions = close_scaler.inverse_transform(predictions)

# ---------------- Train vs Test Graph ----------------
st.subheader("📉 Train vs Test Prediction")

train = data["Close"][:train_size]
test = data["Close"][train_size:]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train, label="Training Data")
ax.plot(test.index, test.values, label="Actual Test Data")
ax.plot(test.index, predictions, label="Predicted Test Data")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ---------------- Next Day Prediction ----------------
last_60 = scaled_data[-60:].reshape(1, 60, scaled_data.shape[1])
next_day_pred = model.predict(last_60)
next_day_price = close_scaler.inverse_transform(next_day_pred)[0][0]

st.subheader("🔮 Next Day Price Prediction")
st.metric(
    label="Predicted Next Day Closing Price",
    value=f"{next_day_price:.2f}"
)

# ---------------- Next Day Graph ----------------
last_30_days = data["Close"][-30:]
next_day_index = last_30_days.index[-1] + pd.Timedelta(days=1)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(last_30_days.index, last_30_days.values, label="Last 30 Days")
ax2.scatter(next_day_index, next_day_price, color="red", s=120, label="Next Day Prediction")
ax2.plot(
    [last_30_days.index[-1], next_day_index],
    [float(last_30_days.values[-1]), next_day_price],
    linestyle="dashed"
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# ---------------- Footer ----------------
st.markdown(
    "---\n"
    "📌 *This project uses real-time Yahoo Finance data and an LSTM neural network "
    "for educational and demonstration purposes.*"
)
