import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

#dataset ko load krna

print("Loading dataset...")

df = pd.read_csv('TSLA.csv')

# Convert Date column from plain text to actual date format
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as the index
df.set_index('Date', inplace=True)

print("✅ Dataset loaded successfully!")
print(f"📅 Date range  : {df.index.min().date()}  →  {df.index.max().date()}")
print(f"📊 Total rows  : {df.shape[0]} trading days")
print(f"📋 Columns     : {list(df.columns)}")

#data ka preview

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== LAST 5 ROWS ===")
print(df.tail())

#dataset ka shape and type

print(f"\n=== DATASET SHAPE ===")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print(f"\n=== COLUMN DATA TYPES ===")
print(df.dtypes)


print("\n=== STATISTICAL SUMMARY ===")
print(df.describe().round(2))

#missing values ka check karna

print("\n=== MISSING VALUES PER COLUMN ===")
missing = df.isnull().sum()
print(missing)

total_missing = missing.sum()
if total_missing == 0:
    print("\n✅ No missing values found in the dataset!")
else:
    print(f"\n⚠️  Total missing values found: {total_missing}")

#duplicate rows ka check karna

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")

if duplicates == 0:
    print("✅ No duplicate rows — dataset is clean!")
else:
    df = df[~df.duplicated()]
    print(f"✅ Removed {duplicates} duplicate rows.")

#chart ka overview

print("\nGenerating overview chart...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top chart: Adj Close price over time
axes[0].plot(df.index, df['Adj Close'],
             color='#1f77b4',
             linewidth=1.2,
             label='Adj Close Price')
axes[0].set_title("Tesla (TSLA) — Adjusted Closing Price (2010–2020)",
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel("Price (USD)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].xaxis.set_major_locator(mdates.YearLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Bottom chart: Trading Volume
axes[1].bar(df.index, df['Volume'],
            color='#ff7f0e',
            alpha=0.6,
            label='Volume',
            width=1)
axes[1].set_title("Tesla (TSLA) — Daily Trading Volume",
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel("Volume")
axes[1].set_xlabel("Year")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('tsla_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved as tsla_overview.png")

#saving the cleaned data frame

df.to_csv('TSLA_cleaned.csv')
print("✅ Cleaned dataset saved as TSLA_cleaned.csv")

#summary

print("\n" + "=" * 50)
print("      STEP 1 COMPLETE — ENVIRONMENT READY")
print("=" * 50)
print(f"  Dataset shape  : {df.shape}")
print(f"  Date range     : {df.index.min().date()} → {df.index.max().date()}")
print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"  Target column  : Adj Close")
print(f"  Price range    : ${df['Adj Close'].min():.2f} → ${df['Adj Close'].max():.2f}")
print("=" * 50)
print("  ✅ Ready to move to Step 2")
print("=" * 50)

# ============================================================
# STEP 2 — EDA & VISUALISATIONS
# ============================================================

# --- Year wise breakdown ---
print("\n=== YEAR WISE TRADING DAYS ===")
df['Year'] = df.index.year
yearly = df.groupby('Year')['Adj Close'].agg(['mean', 'min', 'max']).round(2)
yearly.columns = ['Avg Price', 'Min Price', 'Max Price']
print(yearly)
df.drop(columns=['Year'], inplace=True)

# --- Correlation heatmap ---
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(8, 6))
correlation = df.corr()
sns.heatmap(
    correlation,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    square=True
)
plt.title("Correlation Heatmap — TSLA Features", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot1_correlation_heatmap.png")

# --- Adj Close price over time ---
print("\nGenerating Adj Close price chart...")
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Adj Close'],
         color='#1f77b4', linewidth=1.2, label='Adj Close')
plt.title("Tesla (TSLA) — Adjusted Closing Price (2010–2020)",
          fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('plot2_adj_close_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot2_adj_close_history.png")

# --- OHLC chart ---
print("\nGenerating OHLC chart...")
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Open'],  label='Open',  linewidth=0.8, alpha=0.8)
plt.plot(df.index, df['High'],  label='High',  linewidth=0.8, alpha=0.8)
plt.plot(df.index, df['Low'],   label='Low',   linewidth=0.8, alpha=0.8)
plt.plot(df.index, df['Close'], label='Close', linewidth=0.8, alpha=0.8)
plt.title("TSLA — Open, High, Low, Close Prices",
          fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('plot3_ohlc.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot3_ohlc.png")

# --- Trading Volume ---
print("\nGenerating volume chart...")
plt.figure(figsize=(14, 4))
plt.bar(df.index, df['Volume'],
        color='#ff7f0e', alpha=0.6, width=1, label='Volume')
plt.title("TSLA — Daily Trading Volume (2010–2020)",
          fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Volume")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('plot4_volume.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot4_volume.png")

# --- Moving Averages ---
print("\nGenerating moving averages chart...")
df['MA30']  = df['Adj Close'].rolling(window=30).mean()
df['MA100'] = df['Adj Close'].rolling(window=100).mean()
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Adj Close'],
         label='Adj Close', linewidth=1, alpha=0.6, color='#1f77b4')
plt.plot(df.index, df['MA30'],
         label='30-Day MA', linewidth=1.5, color='#ff7f0e')
plt.plot(df.index, df['MA100'],
         label='100-Day MA', linewidth=1.5, color='#2ca02c')
plt.title("TSLA — Adj Close with 30 & 100 Day Moving Averages",
          fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('plot5_moving_averages.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot5_moving_averages.png")
df.drop(columns=['MA30', 'MA100'], inplace=True)

# --- Daily Returns ---
print("\nGenerating daily returns chart...")
df['Daily Return'] = df['Adj Close'].pct_change() * 100
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(df.index, df['Daily Return'],
             color='purple', linewidth=0.8, alpha=0.7)
axes[0].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title("TSLA — Daily Percentage Returns",
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel("Return (%)")
axes[0].grid(True, alpha=0.3)
axes[1].hist(df['Daily Return'].dropna(),
             bins=100, color='purple', alpha=0.7, edgecolor='white')
axes[1].axvline(x=0, color='black', linewidth=1, linestyle='--')
axes[1].set_title("Distribution of Daily Returns",
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel("Return (%)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot6_daily_returns.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot6_daily_returns.png")
df.drop(columns=['Daily Return'], inplace=True)

# --- Yearly average closing price ---
print("\nGenerating yearly average price chart...")
df['Year'] = df.index.year
yearly_avg = df.groupby('Year')['Adj Close'].mean().round(2)
plt.figure(figsize=(12, 5))
bars = plt.bar(yearly_avg.index, yearly_avg.values,
               color='#1f77b4', alpha=0.8, edgecolor='white', width=0.6)
for bar, val in zip(bars, yearly_avg.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 2,
             f'${val:.0f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.title("TSLA — Yearly Average Adjusted Closing Price",
          fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Avg Price (USD)")
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot7_yearly_avg_price.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot7_yearly_avg_price.png")

# --- Boxplot by year ---
print("\nGenerating yearly boxplot...")
year_groups = [df[df['Year'] == y]['Adj Close'].values
               for y in sorted(df['Year'].unique())]
plt.figure(figsize=(14, 6))
plt.boxplot(year_groups,
            labels=sorted(df['Year'].unique()),
            patch_artist=True,
            boxprops=dict(facecolor='#AED6F1', color='#1f77b4'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='#1f77b4'),
            capprops=dict(color='#1f77b4'))
plt.title("TSLA — Adj Close Price Distribution by Year",
          fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Adj Close Price (USD)")
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot8_yearly_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot8_yearly_boxplot.png")
df.drop(columns=['Year'], inplace=True)

# --- Step 2 Summary ---
print(f"\n  Charts generated    : 8")
print(f"  Avg Adj Close       : ${df['Adj Close'].mean():.2f}")
print(f"  Max Adj Close       : ${df['Adj Close'].max():.2f}")
print(f"  Min Adj Close       : ${df['Adj Close'].min():.2f}")
print("  ✅ Ready to move to Step 3 — Data Preprocessing")

#data preporocessing

from sklearn.preprocessing import MinMaxScaler
import numpy as np

data=df[['Adj Close']].copy()
print(f"Target column selected: Adj Close")
print(f"Shape: {data.shape}")

print(f"\nMissing values before filling: {data.isnull().sum().values[0]}")
data.ffill(inplace=True)
data.bfill(inplace=True)
print(f"Missing values after filling : {data.isnull().sum().values[0]}")
print("Missing values handled using forward fill")

train_size = int(len(data) * 0.80)
train_data = data.iloc[:train_size]
test_data = data.iloc[:train_size]

print(f"\n Train Test Split Done")
print(f"  Total rows : {len(data)}")
print(f" Train rows : {len(train_data)} ({train_data.index.min().date()} -> {train_data.index.max().date()})")
print(f" Test rows : {len(test_data)} ({test_data.index.min().date()} -> {test_data.index.max().date()})")

scaler = MinMaxScaler(feature_range=(0,1))

#fit only on training data
train_scaled = scaler.fit_transform(train_data)

#transform test data using the same scalar
test_scaled = scaler.transform(test_data)

print(f"\n MinMaxScaler applied")
print(f" Train min after scaling : {train_scaled.min():.4f}")
print(f" Train max after scaling : {train_scaled.max():.4f}")
print(f" Test min after scaling : {test_scaled.min():.4f}")
print(f" Test max after scaling : {test_scaled.max():.4f}")

def create_sequences(data, window_size=60):
    X , y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i,0])
        y.append(data[i,0])
    return np.array(X) , np.array(y)

WINDOW_SIZE = 60

X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
X_test,  y_test  = create_sequences(test_scaled, WINDOW_SIZE)

print(f"\n Time Series Sequences Created")
print(f"Window size : {WINDOW_SIZE} days")
print(f"  X_train shape : {X_train.shape} -> (samples,window)")
print(f"  y_train shape : {y_train.shape}")
print(f"  X_test shape : {X_test.shape}")
print(f"  y_test shape : {y_test.shape}")

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

print("\nGenerating train test split chart...")

plt.figure(figsize=(14,5))
plt.plot(train_data.index, train_data['Adj Close'], color='#1f77b4' , linewidth=1, label='Training Data')
plt.plot(test_data.index, test_data['Adj Close'], color='#ff7f0e' , linewidth=1, label='Testing Data')
plt.title("TSLA - Train vs Test Split", fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Adj Close Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('plot9_train_test_split.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot9_train_test_split.png")

print("\nGenerating scaled data chart...")

plt.figure(figsize=(14, 4))
plt.plot(train_scaled, color='#1f77b4',
         linewidth=0.8, label='Scaled Train Data')
plt.plot(range(len(train_scaled),
               len(train_scaled) + len(test_scaled)),
         test_scaled,
         color='#ff7f0e', linewidth=0.8, label='Scaled Test Data')
plt.title("TSLA — Scaled Adj Close Price (0 to 1)",
          fontsize=13, fontweight='bold')
plt.xlabel("Trading Days")
plt.ylabel("Scaled Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot10_scaled_data.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot10_scaled_data.png")


print("\n" + "=" * 50)
print("      STEP 3 COMPLETE — PREPROCESSING DONE")
print("=" * 50)
print(f"  Target column    : Adj Close")
print(f"  Window size      : {WINDOW_SIZE} days")
print(f"  X_train shape    : {X_train.shape}")
print(f"  X_test  shape    : {X_test.shape}")
print(f"  y_train shape    : {y_train.shape}")
print(f"  y_test  shape    : {y_test.shape}")
print("=" * 50)
print("  ✅ Ready to move to Step 4 — Model Building")
print("=" * 50)

# ============================================================
# STEP 4 — MODEL BUILDING (SimpleRNN & LSTM)
# ============================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

print("\n=== STEP 4 — MODEL BUILDING ===")

# ============================================================
# SECTION 1 — Set random seed
# This makes sure results are same every time you run
# Like using the same starting point every time
# ============================================================

tf.random.set_seed(42)
np.random.seed(42)
print("✅ Random seed set to 42")

# ============================================================
# SECTION 2 — BUILD SIMPLERNN MODEL
# SimpleRNN is like a student with short memory
# It learns from recent days but forgets older patterns
# ============================================================

print("\nBuilding SimpleRNN model...")

rnn_model = Sequential([

    # First RNN layer
    # 60 units = 60 neurons learning from the data
    # return_sequences=True means pass output to next RNN layer
    # input_shape = (60 days, 1 feature)
    SimpleRNN(units=60,
              return_sequences=True,
              input_shape=(X_train.shape[1], 1)),

    # Dropout randomly switches off 20% neurons during training
    # This prevents the model from memorising instead of learning
    Dropout(0.2),

    # Second RNN layer
    # return_sequences=False because this is the last RNN layer
    SimpleRNN(units=60,
              return_sequences=False),

    Dropout(0.2),

    # Dense layer = final output layer
    # units=1 because we predict just 1 price value
    Dense(units=1)
])

# Print model summary — shows layers and parameters
print("\n--- SimpleRNN Model Summary ---")
rnn_model.summary()

# ============================================================
# SECTION 3 — COMPILE SIMPLERNN MODEL
# Adam optimizer adjusts learning rate automatically
# MSE loss = mean squared error
# Lower MSE = better predictions
# ============================================================

rnn_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
print("\n✅ SimpleRNN model compiled")

# ============================================================
# SECTION 4 — CALLBACKS FOR SIMPLERNN
# EarlyStopping = stops training when model stops improving
# Like stopping practice when you stop getting better
# ModelCheckpoint = saves the best version of the model
# ============================================================

rnn_callbacks = [
    EarlyStopping(
        monitor='val_loss',    # watch validation loss
        patience=10,           # stop if no improvement for 10 epochs
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_rnn_model.keras',  # save best model here
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# SECTION 5 — TRAIN SIMPLERNN MODEL
# epochs=50 means go through all training data 50 times
# batch_size=32 means learn from 32 samples at a time
# validation_split=0.1 means use 10% of train data
# to check performance during training
# ============================================================

print("\nTraining SimpleRNN model...")
print("This may take a few minutes — please wait...\n")

rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=rnn_callbacks,
    verbose=1
)

print("\n✅ SimpleRNN model training complete")

# ============================================================
# SECTION 6 — PLOT SIMPLERNN TRAINING HISTORY
# This shows how the model improved over each epoch
# Like a graph of your test scores improving over time
# ============================================================

print("\nGenerating SimpleRNN training history chart...")

plt.figure(figsize=(14, 5))
plt.plot(rnn_history.history['loss'],
         label='Train Loss', color='#1f77b4', linewidth=1.5)
plt.plot(rnn_history.history['val_loss'],
         label='Validation Loss', color='#ff7f0e', linewidth=1.5)
plt.title("SimpleRNN — Training vs Validation Loss",
          fontsize=13, fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot11_rnn_training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot11_rnn_training_loss.png")

# ============================================================
# SECTION 7 — BUILD LSTM MODEL
# LSTM is like a student with a notebook
# It decides what to remember, what to forget
# and what to focus on — much smarter than SimpleRNN
# ============================================================

print("\nBuilding LSTM model...")

lstm_model = Sequential([

    # First LSTM layer
    # return_sequences=True passes output to next LSTM layer
    LSTM(units=60,
         return_sequences=True,
         input_shape=(X_train.shape[1], 1)),

    Dropout(0.2),

    # Second LSTM layer
    LSTM(units=60,
         return_sequences=False),

    Dropout(0.2),

    # Output layer — predicts 1 price value
    Dense(units=1)
])

print("\n--- LSTM Model Summary ---")
lstm_model.summary()

# ============================================================
# SECTION 8 — COMPILE LSTM MODEL
# ============================================================

lstm_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
print("\n✅ LSTM model compiled")

# ============================================================
# SECTION 9 — CALLBACKS FOR LSTM
# ============================================================

lstm_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_lstm_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# SECTION 10 — TRAIN LSTM MODEL
# ============================================================

print("\nTraining LSTM model...")
print("This may take a few minutes — please wait...\n")

lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=lstm_callbacks,
    verbose=1
)

print("\n✅ LSTM model training complete")

# ============================================================
# SECTION 11 — PLOT LSTM TRAINING HISTORY
# ============================================================

print("\nGenerating LSTM training history chart...")

plt.figure(figsize=(14, 5))
plt.plot(lstm_history.history['loss'],
         label='Train Loss', color='#1f77b4', linewidth=1.5)
plt.plot(lstm_history.history['val_loss'],
         label='Validation Loss', color='#ff7f0e', linewidth=1.5)
plt.title("LSTM — Training vs Validation Loss",
          fontsize=13, fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot12_lstm_training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot12_lstm_training_loss.png")

# ============================================================
# SECTION 12 — COMPARE BOTH TRAINING HISTORIES
# Side by side comparison of RNN vs LSTM learning curves
# ============================================================

print("\nGenerating RNN vs LSTM comparison chart...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# SimpleRNN loss chart
axes[0].plot(rnn_history.history['loss'],
             label='Train Loss', color='#1f77b4', linewidth=1.5)
axes[0].plot(rnn_history.history['val_loss'],
             label='Val Loss', color='#ff7f0e', linewidth=1.5)
axes[0].set_title("SimpleRNN — Loss Curve",
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss (MSE)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LSTM loss chart
axes[1].plot(lstm_history.history['loss'],
             label='Train Loss', color='#1f77b4', linewidth=1.5)
axes[1].plot(lstm_history.history['val_loss'],
             label='Val Loss', color='#ff7f0e', linewidth=1.5)
axes[1].set_title("LSTM — Loss Curve",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Loss (MSE)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("SimpleRNN vs LSTM — Training Loss Comparison",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot13_rnn_vs_lstm_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot13_rnn_vs_lstm_loss.png")

# ============================================================
# STEP 4 SUMMARY
# ============================================================

rnn_epochs_ran = len(rnn_history.history['loss'])
lstm_epochs_ran = len(lstm_history.history['loss'])

print("\n" + "=" * 50)
print("      STEP 4 COMPLETE — MODELS TRAINED")
print("=" * 50)
print(f"  SimpleRNN epochs ran  : {rnn_epochs_ran}")
print(f"  LSTM epochs ran       : {lstm_epochs_ran}")
print(f"  SimpleRNN final loss  : {rnn_history.history['loss'][-1]:.6f}")
print(f"  LSTM final loss       : {lstm_history.history['loss'][-1]:.6f}")
print(f"  Best RNN  model saved : best_rnn_model.keras")
print(f"  Best LSTM model saved : best_lstm_model.keras")
print("=" * 50)
print("  ✅ Ready to move to Step 5 — Evaluation")
print("=" * 50)

# ============================================================
# STEP 5 — EVALUATION, PREDICTIONS & COMPARISON
# ============================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

print("\n=== STEP 5 — EVALUATION & PREDICTIONS ===")

# ============================================================
# SECTION 1 — MAKE PREDICTIONS ON TEST DATA
# We use the trained models to predict prices
# on data they have never seen before (test set)
# ============================================================

print("\nMaking predictions on test data...")

# SimpleRNN predictions
rnn_predictions = rnn_model.predict(X_test)

# LSTM predictions
lstm_predictions = lstm_model.predict(X_test)

print("✅ Predictions made for both models")

# ============================================================
# SECTION 2 — INVERSE TRANSFORM PREDICTIONS
# Remember we scaled prices between 0 and 1 in Step 3
# Now we convert them back to actual dollar values
# Like converting a percentage grade back to actual marks
# ============================================================

rnn_predictions  = scaler.inverse_transform(rnn_predictions)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Also inverse transform actual test prices
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

print("✅ Predictions converted back to actual dollar values")
print(f"\n   First 5 Actual prices    : {y_test_actual[:5].flatten().round(2)}")
print(f"   First 5 RNN predictions  : {rnn_predictions[:5].flatten().round(2)}")
print(f"   First 5 LSTM predictions : {lstm_predictions[:5].flatten().round(2)}")

# ============================================================
# SECTION 3 — CALCULATE ERROR METRICS
# MSE  = Mean Squared Error (lower is better)
# RMSE = Root Mean Squared Error (in dollar terms)
# MAE  = Mean Absolute Error (average dollar difference)
# ============================================================

print("\n--- Calculating Error Metrics ---")

# SimpleRNN metrics
rnn_mse  = mean_squared_error(y_test_actual, rnn_predictions)
rnn_rmse = math.sqrt(rnn_mse)
rnn_mae  = mean_absolute_error(y_test_actual, rnn_predictions)

# LSTM metrics
lstm_mse  = mean_squared_error(y_test_actual, lstm_predictions)
lstm_rmse = math.sqrt(lstm_mse)
lstm_mae  = mean_absolute_error(y_test_actual, lstm_predictions)

print("\n✅ Error Metrics Calculated")
print("\n--- SimpleRNN ---")
print(f"   MSE  : {rnn_mse:.4f}")
print(f"   RMSE : ${rnn_rmse:.4f}")
print(f"   MAE  : ${rnn_mae:.4f}")

print("\n--- LSTM ---")
print(f"   MSE  : {lstm_mse:.4f}")
print(f"   RMSE : ${lstm_rmse:.4f}")
print(f"   MAE  : ${lstm_mae:.4f}")

# ============================================================
# SECTION 4 — COMPARISON TABLE
# Side by side comparison of both models
# ============================================================

print("\n--- Model Comparison Table ---")
print(f"{'Metric':<10} {'SimpleRNN':>15} {'LSTM':>15} {'Winner':>10}")
print("-" * 55)
print(f"{'MSE':<10} {rnn_mse:>15.4f} {lstm_mse:>15.4f} {'LSTM' if lstm_mse < rnn_mse else 'RNN':>10}")
print(f"{'RMSE':<10} {rnn_rmse:>15.4f} {lstm_rmse:>15.4f} {'LSTM' if lstm_rmse < rnn_rmse else 'RNN':>10}")
print(f"{'MAE':<10} {rnn_mae:>15.4f} {lstm_mae:>15.4f} {'LSTM' if lstm_mae < rnn_mae else 'RNN':>10}")
print("-" * 55)

# ============================================================
# SECTION 5 — PLOT ACTUAL VS PREDICTED (SimpleRNN)
# Blue line = actual prices
# Orange line = what SimpleRNN predicted
# ============================================================

print("\nGenerating SimpleRNN prediction chart...")

# Get test dates for x axis
test_dates = test_data.index[WINDOW_SIZE:]

plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_test_actual,
         color='#1f77b4', linewidth=1.2, label='Actual Price')
plt.plot(test_dates, rnn_predictions,
         color='#ff7f0e', linewidth=1.2,
         linestyle='--', label='SimpleRNN Predicted')
plt.title("SimpleRNN — Actual vs Predicted Stock Price",
          fontsize=13, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot14_rnn_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot14_rnn_predictions.png")

# ============================================================
# SECTION 6 — PLOT ACTUAL VS PREDICTED (LSTM)
# ============================================================

print("\nGenerating LSTM prediction chart...")

plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_test_actual,
         color='#1f77b4', linewidth=1.2, label='Actual Price')
plt.plot(test_dates, lstm_predictions,
         color='#2ca02c', linewidth=1.2,
         linestyle='--', label='LSTM Predicted')
plt.title("LSTM — Actual vs Predicted Stock Price",
          fontsize=13, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot15_lstm_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot15_lstm_predictions.png")

# ============================================================
# SECTION 7 — PLOT BOTH ON SAME CHART
# Easy visual comparison of both models together
# ============================================================

print("\nGenerating combined comparison chart...")

plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_test_actual,
         color='#1f77b4', linewidth=1.5, label='Actual Price')
plt.plot(test_dates, rnn_predictions,
         color='#ff7f0e', linewidth=1.2,
         linestyle='--', label='SimpleRNN Predicted')
plt.plot(test_dates, lstm_predictions,
         color='#2ca02c', linewidth=1.2,
         linestyle='--', label='LSTM Predicted')
plt.title("SimpleRNN vs LSTM — Actual vs Predicted Price",
          fontsize=13, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot16_combined_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot16_combined_predictions.png")

# ============================================================
# SECTION 8 — FUTURE PREDICTIONS
# Predict 1 day, 5 days and 10 days ahead
# Using recursive forecasting
# Each prediction is fed back as input for the next one
# Like a chain of guesses building on each other
# ============================================================

print("\n--- Future Price Predictions ---")

def predict_future(model, last_sequence, days, scaler):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        # Reshape to (1, window_size, 1) for model input
        input_seq = current_seq.reshape(1, WINDOW_SIZE, 1)

        # Predict next price (scaled)
        next_price = model.predict(input_seq, verbose=0)[0][0]

        # Store prediction
        predictions.append(next_price)

        # Slide the window — remove first value, add new prediction
        current_seq = np.append(current_seq[1:], [[next_price]], axis=0)

    # Convert predictions back to actual dollar values
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()


# Get the last 60 days from test data as starting sequence
last_60_days = test_scaled[-WINDOW_SIZE:]

# --- 1 Day Prediction ---
rnn_1day  = predict_future(rnn_model,  last_60_days, 1,  scaler)
lstm_1day = predict_future(lstm_model, last_60_days, 1,  scaler)

# --- 5 Day Prediction ---
rnn_5day  = predict_future(rnn_model,  last_60_days, 5,  scaler)
lstm_5day = predict_future(lstm_model, last_60_days, 5,  scaler)

# --- 10 Day Prediction ---
rnn_10day  = predict_future(rnn_model,  last_60_days, 10, scaler)
lstm_10day = predict_future(lstm_model, last_60_days, 10, scaler)

print("\n✅ Future Predictions Complete")
print("\n--- 1 Day Ahead ---")
print(f"   SimpleRNN : ${rnn_1day[0]:.2f}")
print(f"   LSTM      : ${lstm_1day[0]:.2f}")

print("\n--- 5 Days Ahead ---")
for i, (r, l) in enumerate(zip(rnn_5day, lstm_5day), 1):
    print(f"   Day {i} → RNN: ${r:.2f}  |  LSTM: ${l:.2f}")

print("\n--- 10 Days Ahead ---")
for i, (r, l) in enumerate(zip(rnn_10day, lstm_10day), 1):
    print(f"   Day {i:02d} → RNN: ${r:.2f}  |  LSTM: ${l:.2f}")

# ============================================================
# SECTION 9 — PLOT FUTURE PREDICTIONS
# ============================================================

print("\nGenerating future predictions chart...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

days_5  = [f"Day {i}" for i in range(1, 6)]
days_10 = [f"Day {i}" for i in range(1, 11)]

# 5 day forecast chart
axes[0].plot(days_5, rnn_5day,
             marker='o', color='#ff7f0e',
             linewidth=1.5, label='SimpleRNN')
axes[0].plot(days_5, lstm_5day,
             marker='o', color='#2ca02c',
             linewidth=1.5, label='LSTM')
axes[0].set_title("5 Day Price Forecast",
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel("Day")
axes[0].set_ylabel("Predicted Price (USD)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 10 day forecast chart
axes[1].plot(days_10, rnn_10day,
             marker='o', color='#ff7f0e',
             linewidth=1.5, label='SimpleRNN')
axes[1].plot(days_10, lstm_10day,
             marker='o', color='#2ca02c',
             linewidth=1.5, label='LSTM')
axes[1].set_title("10 Day Price Forecast",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Predicted Price (USD)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("SimpleRNN vs LSTM — Future Price Forecasts",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot17_future_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot17_future_predictions.png")

# ============================================================
# SECTION 10 — ERROR DISTRIBUTION PLOT
# Shows how far off predictions were from actual prices
# Closer to 0 = better predictions
# ============================================================

print("\nGenerating error distribution chart...")

rnn_errors  = y_test_actual.flatten() - rnn_predictions.flatten()
lstm_errors = y_test_actual.flatten() - lstm_predictions.flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(rnn_errors, bins=50,
             color='#ff7f0e', alpha=0.7, edgecolor='white')
axes[0].axvline(x=0, color='black', linewidth=1, linestyle='--')
axes[0].set_title("SimpleRNN — Prediction Error Distribution",
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel("Error (Actual - Predicted)")
axes[0].set_ylabel("Frequency")
axes[0].grid(True, alpha=0.3)

axes[1].hist(lstm_errors, bins=50,
             color='#2ca02c', alpha=0.7, edgecolor='white')
axes[1].axvline(x=0, color='black', linewidth=1, linestyle='--')
axes[1].set_title("LSTM — Prediction Error Distribution",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("Error (Actual - Predicted)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot18_error_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot18_error_distribution.png")

# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("         TESLA STOCK PREDICTION — COMPLETE")
print("=" * 55)
print(f"\n  DATASET")
print(f"  Total trading days   : {len(data)}")
print(f"  Train days           : {len(train_data)}")
print(f"  Test days            : {len(test_data)}")
print(f"\n  MODEL PERFORMANCE")
print(f"  {'Metric':<10} {'SimpleRNN':>12} {'LSTM':>12}")
print(f"  {'-'*36}")
print(f"  {'MSE':<10} {rnn_mse:>12.4f} {lstm_mse:>12.4f}")
print(f"  {'RMSE':<10} {rnn_rmse:>12.4f} {lstm_rmse:>12.4f}")
print(f"  {'MAE':<10} {rnn_mae:>12.4f} {lstm_mae:>12.4f}")
print(f"\n  FUTURE PREDICTIONS (Last known price: ${y_test_actual[-1][0]:.2f})")
print(f"  1  Day  → RNN: ${rnn_1day[0]:.2f}   LSTM: ${lstm_1day[0]:.2f}")
print(f"  5  Days → RNN: ${rnn_5day[-1]:.2f}   LSTM: ${lstm_5day[-1]:.2f}")
print(f"  10 Days → RNN: ${rnn_10day[-1]:.2f}   LSTM: ${lstm_10day[-1]:.2f}")
print(f"\n  CHARTS SAVED : 18 PNG files")
print(f"  MODELS SAVED : best_rnn_model.keras")
print(f"                 best_lstm_model.keras")
print("=" * 55)
print("  ✅ Project Complete — Ready for Streamlit App")
print("=" * 55)