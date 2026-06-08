# Tesla Stock Price Prediction

Predicts Tesla (TSLA) stock prices using **SimpleRNN** and **LSTM** deep learning models, with an interactive Streamlit dashboard for exploration and forecasting.

---

## Project Overview

| Item | Detail |
|---|---|
| Dataset | Tesla stock data (2010–2020), ~2,500 trading days |
| Target | Adjusted Closing Price (`Adj Close`) |
| Models | SimpleRNN · LSTM |
| Evaluation | MSE · RMSE · MAE |
| App | Streamlit interactive dashboard |

---

## Project Structure

```
Tesla Stock Price Prediction/
├── Tesla_Stock_Price_Prediction.py   # Full ML pipeline (Steps 1–5)
├── app.py                            # Streamlit dashboard
├── TSLA.csv                          # Raw historical stock data
├── TSLA_cleaned.csv                  # Cleaned data (generated)
├── best_rnn_model.keras              # Saved SimpleRNN model (generated)
├── best_lstm_model.keras             # Saved LSTM model (generated)
└── plot1_*.png ... plot18_*.png      # All charts (generated)
```

---

## Pipeline Steps

### Step 1 — Data Loading & Cleaning
- Loads `TSLA.csv`, parses dates, sets `Date` as index
- Checks for missing values and duplicate rows
- Saves cleaned data as `TSLA_cleaned.csv`
- Generates overview chart (`tsla_overview.png`)

### Step 2 — Exploratory Data Analysis
Generates 8 charts covering:
- Correlation heatmap
- Adjusted close price history
- OHLC (Open, High, Low, Close)
- Daily trading volume
- 30-day and 100-day moving averages
- Daily returns and return distribution
- Yearly average prices and boxplots

### Step 3 — Data Preprocessing
- Selects `Adj Close` as the target feature
- 80/20 train-test split
- Scales prices to [0, 1] using `MinMaxScaler` (fit on train only)
- Creates sliding window sequences of **60 days** to predict the next day
- Reshapes arrays to `(samples, window, 1)` for RNN/LSTM input

### Step 4 — Model Building & Training
Two sequential models, each with identical architecture:

```
Layer 1  →  RNN/LSTM(60 units, return_sequences=True)
Dropout  →  0.2
Layer 2  →  RNN/LSTM(60 units, return_sequences=False)
Dropout  →  0.2
Output   →  Dense(1)
```

- Optimizer: Adam · Loss: MSE
- Up to 50 epochs, batch size 32, 10% validation split
- `EarlyStopping(patience=10)` + `ModelCheckpoint` save the best weights
- Best models saved as `best_rnn_model.keras` and `best_lstm_model.keras`

### Step 5 — Evaluation & Forecasting
- Predictions on unseen test data, inverse-transformed to USD
- Error metrics: MSE, RMSE, MAE for both models side-by-side
- Recursive future forecasts: 1-day, 5-day, and 10-day horizons
- Error distribution plots for both models
- 18 total charts saved as PNG files

---

## Streamlit Dashboard (`app.py`)

The interactive app lets you:
- Upload your own CSV or use the default `TSLA.csv`
- Switch between **SimpleRNN**, **LSTM**, or **Compare Both**
- Adjust window size (30–120 days) via slider
- Select forecast horizon (1, 5, or 10 days)
- View live metrics, prediction charts, and future price forecasts

---

## Getting Started

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow streamlit
```

### 2. Train the Models

Run the full pipeline to generate models and charts:

```bash
python Tesla_Stock_Price_Prediction.py
```

This creates `best_rnn_model.keras`, `best_lstm_model.keras`, and all 18 PNG plots.

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Generated Output Files

| File | Description |
|---|---|
| `TSLA_cleaned.csv` | Cleaned dataset |
| `best_rnn_model.keras` | Trained SimpleRNN weights |
| `best_lstm_model.keras` | Trained LSTM weights |
| `plot1_correlation_heatmap.png` | Feature correlation matrix |
| `plot2_adj_close_history.png` | Full price history |
| `plot3_ohlc.png` | Open/High/Low/Close chart |
| `plot4_volume.png` | Daily trading volume |
| `plot5_moving_averages.png` | 30-day and 100-day MAs |
| `plot6_daily_returns.png` | Returns over time + distribution |
| `plot7_yearly_avg_price.png` | Yearly average prices |
| `plot8_yearly_boxplot.png` | Price spread per year |
| `plot9_train_test_split.png` | Train vs test split |
| `plot10_scaled_data.png` | Normalized price data |
| `plot11_rnn_training_loss.png` | SimpleRNN learning curve |
| `plot12_lstm_training_loss.png` | LSTM learning curve |
| `plot13_rnn_vs_lstm_loss.png` | Side-by-side loss comparison |
| `plot14_rnn_predictions.png` | SimpleRNN: actual vs predicted |
| `plot15_lstm_predictions.png` | LSTM: actual vs predicted |
| `plot16_combined_predictions.png` | Both models on one chart |
| `plot17_future_predictions.png` | 5-day and 10-day forecasts |
| `plot18_error_distribution.png` | Prediction error histograms |

---

## Model Comparison

| Metric | SimpleRNN | LSTM |
|---|---|---|
| MSE | Higher | Lower |
| RMSE | Higher | Lower |
| MAE | Higher | Lower |

LSTM consistently outperforms SimpleRNN because its gating mechanism retains long-range temporal dependencies that SimpleRNN loses through vanishing gradients.

---

## Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras** — model building and training
- **scikit-learn** — scaling and error metrics
- **pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — visualizations
- **Streamlit** — interactive web app
