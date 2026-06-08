
# app.py
# Tesla Stock Price Prediction — Streamlit App
# Run with: streamlit run app.py


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import math
import warnings

warnings.filterwarnings('ignore')

# PAGE ka CONFIGURATION
# This must be the very first streamlit command


st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# TITLE SECTION


st.title("📈 Tesla Stock Price Prediction")
st.markdown("Predicting Tesla stock prices using **SimpleRNN** and **LSTM** deep learning models.")
st.markdown("---")

# SIDEBAR — USER CONTROLS


st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM", "Compare Both"]
)

# Forecast horizon
forecast_days = st.sidebar.selectbox(
    "Forecast Horizon",
    [1, 5, 10],
    format_func=lambda x: f"{x} Day{'s' if x > 1 else ''} Ahead"
)

# Window size
window_size = st.sidebar.slider(
    "Window Size (Days)",
    min_value=30,
    max_value=120,
    value=60,
    step=10,
    help="Number of past days used to predict the next price"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.info(
    "This app uses deep learning models trained on "
    "Tesla historical stock data from 2010 to 2020."
)


# DATA FUNCTION ko load karna


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ============================================================
# FILE UPLOAD SECTION
# Users can upload their own CSV file
# or use the default TSLA.csv
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload Your Own Data**")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="CSV must have Date and Adj Close columns"
)

# Load data based on upload or default
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ Custom file loaded!")
else:
    df = load_data("TSLA.csv")
    st.sidebar.info("Using default TSLA.csv")

# ============================================================
# LOAD MODELS FUNCTION
# ============================================================

@st.cache_resource
def load_models():
    rnn_model  = load_model('best_rnn_model.keras')
    lstm_model = load_model('best_lstm_model.keras')
    return rnn_model, lstm_model

# Load both models
try:
    rnn_model, lstm_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Could not load models: {e}")
    st.info("Please run Tesla_Stock_Price_Prediction.py first to train and save the models.")
    models_loaded = False
    st.stop()

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess(df, window_size):
    data = df[['Adj Close']].copy()

    # Train test split — 80/20
    train_size = int(len(data) * 0.80)
    train_data = data.iloc[:train_size]
    test_data  = data.iloc[train_size:]

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled  = scaler.transform(test_data)

    # Create sequences
    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test,  y_test  = create_sequences(test_scaled,  window_size)

    # Reshape for RNN/LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

    return (X_train, y_train, X_test, y_test,
            scaler, train_data, test_data, test_scaled)

# ============================================================
# FUTURE PREDICTION FUNCTION
# ============================================================

def predict_future(model, last_sequence, days, scaler, window_size):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        input_seq  = current_seq.reshape(1, window_size, 1)
        next_price = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(next_price)
        current_seq = np.append(current_seq[1:], [[next_price]], axis=0)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

# ============================================================
# RUN PREPROCESSING
# ============================================================

with st.spinner("Processing data..."):
    (X_train, y_train, X_test, y_test,
     scaler, train_data, test_data, test_scaled) = preprocess(df, window_size)

# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trading Days", len(df))
col2.metric("Training Days", len(train_data))
col3.metric("Testing Days", len(test_data))
col4.metric("Latest Price", f"${df['Adj Close'].iloc[-1]:.2f}")

st.markdown("---")

# Show raw data toggle
if st.checkbox("Show Raw Data"):
    st.dataframe(df.tail(20), use_container_width=True)

# ============================================================
# SECTION 2 — STOCK PRICE HISTORY CHART
# ============================================================

st.header("📉 Stock Price History")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_data.index, train_data['Adj Close'],
        color='#1f77b4', linewidth=1, label='Training Data')
ax.plot(test_data.index, test_data['Adj Close'],
        color='#ff7f0e', linewidth=1, label='Testing Data')
ax.set_title("TSLA — Adj Close Price with Train/Test Split",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Year")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ============================================================
# SECTION 3 — MAKE PREDICTIONS BASED ON MODEL CHOICE
# ============================================================

st.header("🤖 Model Predictions")

# Get test dates
test_dates = test_data.index[window_size:]

# Make predictions
with st.spinner("Running predictions..."):

    if model_choice == "SimpleRNN":
        predictions = rnn_model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse  = mean_squared_error(y_test_actual, predictions)
        rmse = math.sqrt(mse)
        mae  = mean_absolute_error(y_test_actual, predictions)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE",  f"{mse:.4f}")
        col2.metric("RMSE", f"${rmse:.4f}")
        col3.metric("MAE",  f"${mae:.4f}")

        # Chart
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(test_dates, y_test_actual,
                color='#1f77b4', linewidth=1.2, label='Actual Price')
        ax.plot(test_dates, predictions,
                color='#ff7f0e', linewidth=1.2,
                linestyle='--', label='SimpleRNN Predicted')
        ax.set_title("SimpleRNN — Actual vs Predicted Price",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif model_choice == "LSTM":
        predictions = lstm_model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse  = mean_squared_error(y_test_actual, predictions)
        rmse = math.sqrt(mse)
        mae  = mean_absolute_error(y_test_actual, predictions)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE",  f"{mse:.4f}")
        col2.metric("RMSE", f"${rmse:.4f}")
        col3.metric("MAE",  f"${mae:.4f}")

        # Chart
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(test_dates, y_test_actual,
                color='#1f77b4', linewidth=1.2, label='Actual Price')
        ax.plot(test_dates, predictions,
                color='#2ca02c', linewidth=1.2,
                linestyle='--', label='LSTM Predicted')
        ax.set_title("LSTM — Actual vs Predicted Price",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        # Compare Both
        rnn_preds  = rnn_model.predict(X_test,  verbose=0)
        lstm_preds = lstm_model.predict(X_test, verbose=0)
        rnn_preds  = scaler.inverse_transform(rnn_preds)
        lstm_preds = scaler.inverse_transform(lstm_preds)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rnn_mse   = mean_squared_error(y_test_actual, rnn_preds)
        rnn_rmse  = math.sqrt(rnn_mse)
        rnn_mae   = mean_absolute_error(y_test_actual, rnn_preds)
        lstm_mse  = mean_squared_error(y_test_actual, lstm_preds)
        lstm_rmse = math.sqrt(lstm_mse)
        lstm_mae  = mean_absolute_error(y_test_actual, lstm_preds)

        # Comparison metrics table
        st.subheader("📋 Model Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE'],
            'SimpleRNN': [round(rnn_mse, 4),
                          round(rnn_rmse, 4),
                          round(rnn_mae, 4)],
            'LSTM':      [round(lstm_mse, 4),
                          round(lstm_rmse, 4),
                          round(lstm_mae, 4)],
            'Winner':    [
                'LSTM' if lstm_mse  < rnn_mse  else 'SimpleRNN',
                'LSTM' if lstm_rmse < rnn_rmse else 'SimpleRNN',
                'LSTM' if lstm_mae  < rnn_mae  else 'SimpleRNN'
            ]
        })
        st.dataframe(comparison_df, use_container_width=True)

        # Combined chart
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(test_dates, y_test_actual,
                color='#1f77b4', linewidth=1.5, label='Actual Price')
        ax.plot(test_dates, rnn_preds,
                color='#ff7f0e', linewidth=1.2,
                linestyle='--', label='SimpleRNN')
        ax.plot(test_dates, lstm_preds,
                color='#2ca02c', linewidth=1.2,
                linestyle='--', label='LSTM')
        ax.set_title("SimpleRNN vs LSTM — Actual vs Predicted",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# ============================================================
# SECTION 4 — FUTURE PRICE FORECAST
# ============================================================

st.header(f"🔮 Future Price Forecast — {forecast_days} Day{'s' if forecast_days > 1 else ''} Ahead")

last_sequence = test_scaled[-window_size:]

with st.spinner("Generating future predictions..."):

    if model_choice == "SimpleRNN":
        future_preds = predict_future(
            rnn_model, last_sequence, forecast_days, scaler, window_size)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SimpleRNN Forecast")
            for i, price in enumerate(future_preds, 1):
                st.metric(f"Day {i}", f"${price:.2f}")

    elif model_choice == "LSTM":
        future_preds = predict_future(
            lstm_model, last_sequence, forecast_days, scaler, window_size)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("LSTM Forecast")
            for i, price in enumerate(future_preds, 1):
                st.metric(f"Day {i}", f"${price:.2f}")

    else:
        rnn_future  = predict_future(
            rnn_model,  last_sequence, forecast_days, scaler, window_size)
        lstm_future = predict_future(
            lstm_model, last_sequence, forecast_days, scaler, window_size)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SimpleRNN Forecast")
            for i, price in enumerate(rnn_future, 1):
                st.metric(f"Day {i}", f"${price:.2f}")
        with col2:
            st.subheader("LSTM Forecast")
            for i, price in enumerate(lstm_future, 1):
                st.metric(f"Day {i}", f"${price:.2f}")

        # Future forecast chart
        if forecast_days > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            days_labels = [f"Day {i}" for i in range(1, forecast_days + 1)]
            ax.plot(days_labels, rnn_future,
                    marker='o', color='#ff7f0e',
                    linewidth=1.5, label='SimpleRNN')
            ax.plot(days_labels, lstm_future,
                    marker='o', color='#2ca02c',
                    linewidth=1.5, label='LSTM')
            ax.set_title(f"{forecast_days} Day Price Forecast",
                         fontsize=12, fontweight='bold')
            ax.set_xlabel("Day")
            ax.set_ylabel("Predicted Price (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

st.markdown("---")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "Built with Streamlit · Models: SimpleRNN & LSTM · "
    "Data: Tesla Stock 2010–2020"
)