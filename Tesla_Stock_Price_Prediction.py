

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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