📈 Real-Time Stock Price Prediction Web App
🔍 Project Overview

This project is a real-time stock price prediction web application built using Deep Learning (LSTM) and deployed permanently on Streamlit Cloud.
The application fetches live market data from Yahoo Finance, applies technical indicators, trains an LSTM model, and predicts the next trading day’s closing price with interactive visualizations.

🎯 Key Features

📊 Fetches real-time updated stock market data

🤖 Uses LSTM (Long Short-Term Memory) neural network

📈 Includes technical indicators:

Simple Moving Average (SMA)

Exponential Moving Average (EMA)

Relative Strength Index (RSI)

📉 Displays Train vs Test prediction graph

🔮 Predicts next day closing price

🌐 Permanently deployed using Streamlit Cloud

🧑‍💻 User-friendly web interface

🛠️ Tech Stack

Programming Language: Python

Web Framework: Streamlit

Deep Learning: TensorFlow / Keras (LSTM)

Data Source: Yahoo Finance (yfinance)

Libraries:

NumPy

Pandas

Matplotlib

Scikit-learn

TA (Technical Analysis library)

📂 Project Structure
stock-prediction-streamlit/
│
├── app.py
└── requirements.txt

⚙️ How the Model Works

Fetches historical stock data in real time

Computes technical indicators (SMA, EMA, RSI)

Scales the data using MinMaxScaler

Trains an LSTM model on 80% of the data

Tests the model on remaining 20% data

Visualizes predictions and forecasts the next day price

🚀 Deployment

The application is deployed on Streamlit Cloud, making it permanently accessible via a public URL.

Deployment Steps:

Push app.py and requirements.txt to a public GitHub repository

Connect the repository to Streamlit Cloud

Deploy the app with app.py as the main file

🧪 How to Run Locally (Optional)
pip install -r requirements.txt
streamlit run app.py

🎓 Academic & Learning Value

Demonstrates time-series forecasting

Applies deep learning to real-world financial data

Shows end-to-end ML deployment

Suitable for:

Final-year projects

AIML portfolios

Resume and placement demonstrations

⚠️ Disclaimer

This project is for educational and demonstration purposes only.
The predictions should not be used for real financial trading or investment decisions.

👤 Author

Pritam Maity
B.Tech (AIML)
Stock Prediction using Deep Learning

⭐ If You Like This Project

Give it a ⭐ on GitHub

Share the deployed app link

Use it as a base to build advanced forecasting systems
