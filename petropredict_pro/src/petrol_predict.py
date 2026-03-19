# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║           PetroPredict — Petrol Price Forecasting Engine     ║
║           Python 3.9+  |  Scikit-learn · Pandas · NumPy     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import warnings

# Force UTF-8 output (important on Windows)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# ─── IMPORTS ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ─── PATHS ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'petrol_prices.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── THEME ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#080810',
    'axes.facecolor':    '#0d0d1a',
    'axes.edgecolor':    '#1e1e3a',
    'axes.labelcolor':   '#e8e8f5',
    'text.color':        '#e8e8f5',
    'xtick.color':       '#6060a0',
    'ytick.color':       '#6060a0',
    'grid.color':        '#1e1e3a',
    'grid.alpha':        0.6,
    'legend.facecolor':  '#141428',
    'legend.edgecolor':  '#1e1e3a',
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

C_ORANGE = '#ff6b35'
C_TEAL   = '#00d4aa'
C_AMBER  = '#ff9500'
C_PURPLE = '#6c63ff'
C_GREY   = '#6060a0'

FEATURES = [
    'Crude_Oil_USD', 'USD_INR_Rate', 'Inflation_Index',
    'Month', 'Year',
    'Lag_1', 'Lag_2', 'Lag_3',
    'Rolling_3M', 'Rolling_6M', 'Price_Change'
]
TARGET = 'Petrol_Price_INR'

# ─── HELPERS ──────────────────────────────────────────────────
def banner(title='', char='═', width=66):
    if title:
        pad = (width - len(title) - 4) // 2
        print(f"\n{char*pad}  {title}  {char*(width - pad - len(title) - 4)}")
    else:
        print(char * width)

def tick(msg):
    print(f"  ✓  {msg}")

def info(msg):
    print(f"  ·  {msg}")

def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'Model': name,
        'MAE':   round(mae,  3),
        'RMSE':  round(rmse, 3),
        'R²':    round(r2,   4),
        'MAPE%': round(mape, 2)
    }

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
banner('STEP 1 · LOAD DATA')

if not os.path.isfile(DATA_PATH):
    print(f"\n  ERROR: Data file not found at:\n    {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

tick(f"Loaded {len(df):,} records  ·  {df.shape[1]} columns")
info(f"Date range : {df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}")
info(f"Price range: ₹{df[TARGET].min():.2f} – ₹{df[TARGET].max():.2f}/L")

# Basic data quality check
null_count = df.isnull().sum().sum()
tick(f"Data quality: {null_count} null values found")

# ══════════════════════════════════════════════════════════════
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════
banner('STEP 2 · EXPLORATORY DATA ANALYSIS')

fig = plt.figure(figsize=(16, 12), facecolor='#080810')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=.38, wspace=.32)

# Panel 1 — Price over time with fill
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(df['Date'], df[TARGET], color=C_ORANGE, lw=2.2, zorder=3)
ax1.fill_between(df['Date'], df[TARGET], alpha=.12, color=C_ORANGE)
ax1.set_title('Petrol Price Over Time (₹/L)', color=C_ORANGE, fontweight='bold', pad=10)
ax1.set_ylabel('Price (₹/L)')
ax1.grid(alpha=.4, ls='--')

# Panel 2 — Histogram
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df[TARGET], bins=22, color=C_ORANGE, edgecolor='#0d0d1a', alpha=.85, rwidth=.88)
ax2.set_title('Price Distribution', color=C_ORANGE, fontweight='bold', pad=10)
ax2.set_xlabel('Price (₹/L)')
ax2.grid(axis='y', alpha=.4, ls='--')

# Panel 3 — Crude vs Petrol scatter
ax3 = fig.add_subplot(gs[1, 0])
sc  = ax3.scatter(df['Crude_Oil_USD'], df[TARGET],
                  c=df['Year'], cmap='YlOrRd', alpha=.55, s=22, zorder=3)
ax3.set_title('Crude Oil vs Petrol Price', color=C_ORANGE, fontweight='bold', pad=10)
ax3.set_xlabel('Crude Oil (USD/bbl)')
ax3.set_ylabel('Petrol (₹/L)')
ax3.grid(alpha=.4, ls='--')
cbar = fig.colorbar(sc, ax=ax3, pad=.02)
cbar.set_label('Year', color=C_GREY)
cbar.ax.yaxis.set_tick_params(color=C_GREY)

# Panel 4 — USD/INR vs Petrol scatter
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(df['USD_INR_Rate'], df[TARGET],
            c=df['Year'], cmap='YlGn', alpha=.55, s=22)
ax4.set_title('USD/INR vs Petrol Price', color=C_ORANGE, fontweight='bold', pad=10)
ax4.set_xlabel('USD/INR Rate')
ax4.set_ylabel('Petrol (₹/L)')
ax4.grid(alpha=.4, ls='--')

# Panel 5 — Correlation heatmap
ax5  = fig.add_subplot(gs[1, 2])
cols = [TARGET, 'Crude_Oil_USD', 'USD_INR_Rate', 'Inflation_Index', 'Month', 'Year']
corr = df[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, ax=ax5, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0,
    linewidths=.5, linecolor='#141428',
    annot_kws={'size': 8, 'color': 'white'},
    cbar_kws={'shrink': .75}
)
ax5.set_title('Correlation Matrix', color=C_ORANGE, fontweight='bold', pad=10)
ax5.tick_params(axis='x', rotation=45)

fig.suptitle('PetroPredict — Exploratory Data Analysis',
             fontsize=15, color=C_ORANGE, fontweight='bold', y=.98)

eda_path = os.path.join(OUT_DIR, '01_eda_plots.png')
plt.savefig(eda_path, dpi=150, bbox_inches='tight', facecolor='#080810')
plt.close()
tick(f"EDA plots saved → {eda_path}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
banner('STEP 3 · FEATURE ENGINEERING')

df['Lag_1']        = df[TARGET].shift(1)
df['Lag_2']        = df[TARGET].shift(2)
df['Lag_3']        = df[TARGET].shift(3)
df['Rolling_3M']   = df[TARGET].rolling(3).mean()
df['Rolling_6M']   = df[TARGET].rolling(6).mean()
df['Price_Change'] = df[TARGET].diff()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

tick(f"Lag features created   : Lag_1, Lag_2, Lag_3")
tick(f"Rolling averages added : Rolling_3M, Rolling_6M")
tick(f"Price change diff added: Price_Change")
info(f"Final dataset shape    : {df.shape}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
banner('STEP 4 · TRAIN / TEST SPLIT')

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

tick(f"Train set : {len(X_train):>4} samples")
tick(f"Test set  : {len(X_test):>4} samples")
info(f"Split ratio: 80 / 20  (chronological, no shuffle)")

# ══════════════════════════════════════════════════════════════
# STEP 5 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════
banner('STEP 5 · TRAINING MODELS')

# Linear Regression
t0 = time.time()
lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)
tick(f"Linear Regression trained   ({time.time()-t0:.2f}s)")

# Random Forest
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=300, max_depth=12,
    min_samples_split=4, min_samples_leaf=2,
    random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
tick(f"Random Forest trained       ({time.time()-t0:.2f}s)")

# SVR
t0 = time.time()
svr = SVR(kernel='rbf', C=120, gamma=0.012, epsilon=0.08)
svr.fit(X_train_sc, y_train)
y_svr = svr.predict(X_test_sc)
tick(f"Support Vector Regression   ({time.time()-t0:.2f}s)")

# Gradient Boosting (bonus)
t0 = time.time()
gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.08, max_depth=5,
    subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_gb = gb.predict(X_test)
tick(f"Gradient Boosting trained   ({time.time()-t0:.2f}s)")

# ══════════════════════════════════════════════════════════════
# STEP 6 — EVALUATE
# ══════════════════════════════════════════════════════════════
banner('STEP 6 · EVALUATION')

results = pd.DataFrame([
    evaluate_model('Linear Regression',    y_test, y_lr),
    evaluate_model('Random Forest',        y_test, y_rf),
    evaluate_model('SVR (RBF)',            y_test, y_svr),
    evaluate_model('Gradient Boosting',    y_test, y_gb),
])

print(f"\n{'─'*54}")
print(f"  {'Model':<24} {'MAE':>6}  {'RMSE':>6}  {'R²':>7}  {'MAPE%':>6}")
print(f"{'─'*54}")
for _, row in results.iterrows():
    marker = '  ◀ BEST' if row['Model'] == results.loc[results['R²'].idxmax(), 'Model'] else ''
    print(f"  {row['Model']:<24} {row['MAE']:>6.3f}  {row['RMSE']:>6.3f}  {row['R²']:>7.4f}  {row['MAPE%']:>5.2f}%{marker}")
print(f"{'─'*54}")

best = results.loc[results['R²'].idxmax()]
info(f"\nBest model → {best['Model']} | R² = {best['R²']} | RMSE = ₹{best['RMSE']}/L")

# Cross-validation on best model
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
tick(f"5-Fold CV R² → {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 7 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
banner('STEP 7 · MODEL VISUALIZATIONS')

test_dates = df['Date'].iloc[len(X_train):].reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#080810')

# ── Plot 1: Actual vs RF Predicted
ax = axes[0, 0]
ax.plot(test_dates, y_test.values, color=C_ORANGE, lw=2.2, label='Actual', zorder=3)
ax.plot(test_dates, y_rf,          color=C_TEAL,   lw=1.8, ls='--', label='Random Forest', zorder=2)
ax.fill_between(test_dates, y_test.values, y_rf, alpha=.12, color=C_TEAL)
ax.set_title('Actual vs Random Forest Predictions', color=C_ORANGE, fontweight='bold')
ax.set_ylabel('Price (₹/L)')
ax.legend(); ax.grid(alpha=.4, ls='--')

# ── Plot 2: All Models vs Actual
ax = axes[0, 1]
ax.plot(test_dates, y_test.values, color=C_ORANGE, lw=2.5, label='Actual', zorder=5)
ax.plot(test_dates, y_rf,          color=C_TEAL,   lw=1.8, ls='--',    label='Random Forest')
ax.plot(test_dates, y_lr,          color=C_GREY,   lw=1.5, ls=':',     label='Linear Reg.')
ax.plot(test_dates, y_svr,         color=C_PURPLE, lw=1.5, ls='-.',    label='SVR')
ax.plot(test_dates, y_gb,          color=C_AMBER,  lw=1.5, ls='--',    label='Gradient Boosting')
ax.set_title('All Models — Comparison', color=C_ORANGE, fontweight='bold')
ax.set_ylabel('Price (₹/L)')
ax.legend(fontsize=8); ax.grid(alpha=.4, ls='--')

# ── Plot 3: Feature Importance
ax = axes[1, 0]
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
colors   = plt.cm.YlOrRd(np.linspace(.3, .9, len(feat_imp)))
feat_imp.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Importance (Random Forest)', color=C_ORANGE, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=.4, ls='--')

# ── Plot 4: Residuals Plot (RF)
ax = axes[1, 1]
residuals = y_test.values - y_rf
ax.scatter(y_rf, residuals, color=C_TEAL, alpha=.55, s=25)
ax.axhline(0, color=C_ORANGE, lw=1.5, ls='--')
ax.set_title('Residuals — Random Forest', color=C_ORANGE, fontweight='bold')
ax.set_xlabel('Predicted (₹/L)')
ax.set_ylabel('Residual (₹/L)')
ax.grid(alpha=.4, ls='--')

fig.suptitle('PetroPredict — Model Results',
             fontsize=15, color=C_ORANGE, fontweight='bold', y=.99)
plt.tight_layout()

res_path = os.path.join(OUT_DIR, '03_model_results.png')
plt.savefig(res_path, dpi=150, bbox_inches='tight', facecolor='#080810')
plt.close()
tick(f"Model results saved → {res_path}")

# ══════════════════════════════════════════════════════════════
# STEP 8 — FUTURE PREDICTIONS (6 months)
# ══════════════════════════════════════════════════════════════
banner('STEP 8 · 6-MONTH FORECAST')

last        = df.iloc[-1]
price       = last[TARGET]
lag1, lag2, lag3 = price, last['Lag_1'], last['Lag_2']
roll3, roll6    = last['Rolling_3M'], last['Rolling_6M']

preds = []
for i in range(1, 7):
    m  = int(((last['Month'] - 1 + i) % 12) + 1)
    yr = int(last['Year'] + ((last['Month'] - 1 + i) // 12))

    row = {
        'Crude_Oil_USD':  last['Crude_Oil_USD'],
        'USD_INR_Rate':   last['USD_INR_Rate'],
        'Inflation_Index': last['Inflation_Index'],
        'Month': m, 'Year': yr,
        'Lag_1': lag1, 'Lag_2': lag2, 'Lag_3': lag3,
        'Rolling_3M': roll3, 'Rolling_6M': roll6,
        'Price_Change': 0
    }

    p = rf.predict(pd.DataFrame([row])[FEATURES])[0]
    p = round(p, 2)
    preds.append({'Month': f"{m:02d}/{yr}", 'Predicted Price (₹/L)': p})

    # Roll lag window forward
    lag3 = lag2; lag2 = lag1; lag1 = p
    roll3 = (roll3 * 2 + p) / 3
    roll6 = (roll6 * 5 + p) / 6

forecast_df = pd.DataFrame(preds)

print(f"\n  {'Month':<10}  {'Predicted Price':>18}")
print(f"  {'─'*32}")
for _, r in forecast_df.iterrows():
    print(f"  {r['Month']:<10}  ₹{r['Predicted Price (₹/L)']:>14.2f}/L")

# Save forecast chart
fig, ax = plt.subplots(figsize=(10, 5), facecolor='#080810')
months_label = forecast_df['Month'].tolist()
prices_val   = forecast_df['Predicted Price (₹/L)'].tolist()

ax.plot(months_label, prices_val, color=C_ORANGE, lw=2.5, marker='o',
        markersize=7, markerfacecolor=C_TEAL, markeredgecolor=C_ORANGE, zorder=3)
ax.fill_between(range(len(months_label)), prices_val, alpha=.12, color=C_ORANGE)
ax.set_xticks(range(len(months_label)))
ax.set_xticklabels(months_label)
ax.set_title('6-Month Petrol Price Forecast (Random Forest)', color=C_ORANGE, fontweight='bold', pad=12)
ax.set_ylabel('Predicted Price (₹/L)')
ax.grid(alpha=.4, ls='--')

for i, p in enumerate(prices_val):
    ax.annotate(f'₹{p:.2f}', xy=(i, p), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9, color=C_TEAL)

fig.suptitle('PetroPredict — Forecast', fontsize=13, color=C_GREY, y=.99)
plt.tight_layout()

forecast_path = os.path.join(OUT_DIR, '04_forecast.png')
plt.savefig(forecast_path, dpi=150, bbox_inches='tight', facecolor='#080810')
plt.close()
tick(f"Forecast chart saved → {forecast_path}")

# ══════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════
banner()
print(f"""
  PetroPredict pipeline complete!

  Outputs generated:
    ✓  {os.path.join('outputs', '01_eda_plots.png')}
    ✓  {os.path.join('outputs', '03_model_results.png')}
    ✓  {os.path.join('outputs', '04_forecast.png')}

  Best Model  : {best['Model']}
  R² Score    : {best['R²']}
  RMSE        : ₹{best['RMSE']}/L
  MAPE        : {best['MAPE%']}%
""")
banner()
