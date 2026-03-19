# ⛽ PetroPredict — Petrol Price Forecasting Engine

> **ML Capstone Project** — Predicting India's petrol prices using Random Forest, SVR, Gradient Boosting & Linear Regression with 94.7% R² accuracy.

---

## 📁 Project Structure

```
PetroPredict/
│
├── index.html                          ← Web Dashboard (hostable, no server needed)
├── roadmap.html                        ← Project Roadmap
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
│
├── data/
│   └── petrol_prices.csv               ← Dataset (Jan 2010 – Mar 2024, 171 records)
│
├── notebooks/
│   └── petrol_price_prediction.ipynb   ← Jupyter Notebook (step-by-step walkthrough)
│
├── src/
│   └── petrol_predict.py               ← Standalone Python pipeline script
│
└── outputs/                            ← Auto-generated charts
    ├── 01_eda_plots.png                ← EDA visualizations
    ├── 03_model_results.png            ← Model comparison plots
    └── 04_forecast.png                 ← 6-month forecast chart
```

---

## 🎯 Problem Statement

Predict India's petrol price (₹/Litre) based on historical macroeconomic indicators:
- Crude oil prices (USD/barrel)
- USD/INR exchange rate
- Consumer price inflation index
- Engineered lag & rolling average features

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.2 | ML models, metrics, preprocessing, cross-validation |
| `pandas` | ≥1.5 | Data loading, cleaning, feature engineering |
| `numpy` | ≥1.23 | Numerical operations |
| `matplotlib` | ≥3.6 | Static visualizations & charts |
| `seaborn` | ≥0.12 | Heatmaps and statistical plots |
| `jupyter` | ≥1.0 | Interactive notebook environment |

---

## 🚀 Quick Start

### Option A — Python Script (Recommended)

```bash
# 1. Clone / download the project
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run from project root
python src/petrol_predict.py
```

### Option B — Jupyter Notebook

```bash
pip install -r requirements.txt
jupyter notebook
# Open: notebooks/petrol_price_prediction.ipynb
# Run all cells top-to-bottom
```

### Option C — View Web Dashboard

```
Open index.html in any browser — no server required!
```

---

## 📊 Dataset

**File:** `data/petrol_prices.csv`
**Source:** [Kaggle — Petrol Price Forecasting](https://www.kaggle.com/code/madhusagar029/petrol-price-forecasting)

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Monthly date (2010–2024) |
| `Petrol_Price_INR` | float | **Target** — Price in ₹/Litre |
| `Crude_Oil_USD` | float | Crude oil (USD/barrel) |
| `USD_INR_Rate` | float | Dollar-Rupee exchange rate |
| `Inflation_Index` | float | Consumer Price Index |
| `Month` | int | Month number (1–12) |
| `Year` | int | Year |

**Engineered Features:** `Lag_1`, `Lag_2`, `Lag_3`, `Rolling_3M`, `Rolling_6M`, `Price_Change`

---

## 🤖 ML Models

### 1. Linear Regression (Baseline)
- Fast, interpretable baseline
- R² Score: ~81.4%

### 2. Support Vector Regression (SVR)
- RBF kernel, C=120, γ=0.012
- Good generalization on non-linear patterns
- R² Score: ~89.2%

### 3. Gradient Boosting
- 200 estimators, lr=0.08, max_depth=5
- Strong ensemble performance
- R² Score: ~93.1%

### 4. Random Forest ⭐ Best
- 300 trees, max_depth=12
- Captures complex non-linear interactions
- **R² Score: 94.7% | RMSE: ₹1.23/L | MAPE: 2.1%**

---

## 📈 Results

| Model | MAE (₹) | RMSE (₹) | R² Score | MAPE % |
|-------|---------|---------|---------|--------|
| Linear Regression | 2.14 | 2.89 | 0.814 | 3.1% |
| SVR (RBF) | 1.42 | 1.87 | 0.892 | 2.6% |
| Gradient Boosting | 1.05 | 1.48 | 0.931 | 2.3% |
| **Random Forest** | **0.91** | **1.23** | **0.947** | **2.1%** |

---

## ⚙️ ML Pipeline

```
1. Data Load         → Read CSV, validate types & nulls
2. EDA               → Distributions, correlations, scatter plots
3. Feature Eng.      → Lag 1/2/3M, Rolling avg 3M/6M, Price diff
4. Train/Test Split  → 80% train / 20% test (chronological)
5. Model Training    → LR, SVR, GradientBoosting, RandomForest
6. Evaluation        → MAE, RMSE, R², MAPE, 5-fold CV
7. Visualization     → Actual vs Predicted, residuals, importance
8. Forecasting       → Rolling 6-month future price predictions
```

---

## 🌐 Hosting the Web Dashboard

### GitHub Pages (Free)
```bash
git init
git add .
git commit -m "PetroPredict v2"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/PetroPredict.git
git push -u origin main
# Then: Settings → Pages → Deploy from main branch → /root
# Live URL: https://YOUR_USERNAME.github.io/PetroPredict
```

### Netlify (Drag & Drop, Free)
1. Visit [netlify.com](https://netlify.com)
2. Drag the `PetroPredict/` folder into the deploy zone
3. Get a live HTTPS URL instantly

### Vercel (Free)
```bash
npm install -g vercel
vercel --yes
# Follow prompts → live in 30 seconds
```

---

## 📌 Key Findings

- **Crude Oil Price** = strongest predictor (41% feature importance)
- **USD/INR Rate** = 2nd most important (22%)
- **Lag-1M feature** boosts accuracy significantly (+8% R²)
- Random Forest outperforms all models due to non-linear pattern capture
- Price data shows strong autocorrelation (lag features critical)

---

## 👤 Author

Petrol Price Prediction — ML Capstone Project 2024
Dataset: [Kaggle](https://www.kaggle.com/code/madhusagar029/petrol-price-forecasting)

---

*Built with Python · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn*
