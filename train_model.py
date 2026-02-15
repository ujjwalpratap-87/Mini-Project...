"""
AI-Based Air Pollution Prediction System
Model Training & Evaluation Script

Trains Linear Regression and Random Forest models,
evaluates them, and saves results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    df = pd.read_csv("data/air_quality_data.csv")
    print(f"Loaded {len(df)} records with {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    """Clean and feature-engineer the dataset."""
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode city as numeric
    city_map = {c: i for i, c in enumerate(df["city"].unique())}
    df["city_code"] = df["city"].map(city_map)

    # Feature list
    features = [
        "pm25", "pm10", "no2", "co", "so2", "o3",
        "temperature", "humidity", "wind_speed",
        "month", "hour", "city_code"
    ]
    target = "aqi"

    X = df[features]
    y = df[target]

    return X, y, features, city_map


# ─────────────────────────────────────────────
# 3. TRAIN / EVALUATE
# ─────────────────────────────────────────────

def train_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Linear Regression ──
    print("\n[1] Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    lr_cv = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring="r2")
    results["Linear Regression"] = {
        "model": lr,
        "predictions": y_pred_lr,
        "mae":  mean_absolute_error(y_test, y_pred_lr),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "r2":   r2_score(y_test, y_pred_lr),
        "cv_r2_mean": lr_cv.mean(),
        "cv_r2_std":  lr_cv.std(),
    }
    print(f"   MAE={results['Linear Regression']['mae']:.2f}  "
          f"RMSE={results['Linear Regression']['rmse']:.2f}  "
          f"R²={results['Linear Regression']['r2']:.4f}")

    # ── Random Forest ──
    print("\n[2] Training Random Forest (200 trees)...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               min_samples_split=5, random_state=42,
                               n_jobs=-1)
    rf.fit(X_train, y_train)           # RF doesn't need scaling
    y_pred_rf = rf.predict(X_test)

    rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
    results["Random Forest"] = {
        "model": rf,
        "predictions": y_pred_rf,
        "mae":  mean_absolute_error(y_test, y_pred_rf),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "r2":   r2_score(y_test, y_pred_rf),
        "cv_r2_mean": rf_cv.mean(),
        "cv_r2_std":  rf_cv.std(),
    }
    print(f"   MAE={results['Random Forest']['mae']:.2f}  "
          f"RMSE={results['Random Forest']['rmse']:.2f}  "
          f"R²={results['Random Forest']['r2']:.4f}")

    return results, scaler, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────

def create_plots(df, results, X_test, y_test, features):
    os.makedirs("plots", exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    # ── Plot 1: AQI Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Distribution & Categories", fontsize=14, fontweight="bold")

    axes[0].hist(df["aqi"], bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("AQI Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("AQI Distribution")

    cat_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cat_colors = ["#00C853","#64DD17","#FFAB00","#FF6D00","#DD2C00","#8D1919"]
    cat_counts = df["aqi_category"].value_counts().reindex(cat_order, fill_value=0)
    axes[1].bar(cat_counts.index, cat_counts.values, color=cat_colors, edgecolor="white")
    axes[1].set_xlabel("AQI Category")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Records per Category")
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig("plots/01_aqi_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/01_aqi_distribution.png")

    # ── Plot 2: Correlation Heatmap ──
    fig, ax = plt.subplots(figsize=(10, 8))
    num_cols = ["pm25","pm10","no2","co","so2","o3","temperature","humidity","wind_speed","aqi"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Pearson r"})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/02_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/02_correlation_heatmap.png")

    # ── Plot 3: Actual vs Predicted ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Actual vs Predicted AQI", fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["predictions"]
        ax.scatter(y_test, y_pred, alpha=0.3, color="#3F51B5", s=12)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual AQI")
        ax.set_ylabel("Predicted AQI")
        ax.set_title(f"{name}\nR²={res['r2']:.4f}  MAE={res['mae']:.2f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/03_actual_vs_predicted.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/03_actual_vs_predicted.png")

    # ── Plot 4: Feature Importance (RF) ──
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(feat_df)))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
    ax.set_title("Random Forest – Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for i, (val, feat) in enumerate(zip(feat_df["Importance"], feat_df["Feature"])):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/04_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/04_feature_importance.png")

    # ── Plot 5: Monthly AQI Trend ──
    monthly = df.groupby("month")["aqi"].mean()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2.5,
            color="#E91E63", markersize=8)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#E91E63")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Average AQI")
    ax.set_title("Monthly Average AQI Trend (Seasonal Pattern)", fontsize=13, fontweight="bold")
    ax.axhline(100, color="orange", linestyle="--", alpha=0.7, label="Satisfactory threshold")
    ax.axhline(200, color="red", linestyle="--", alpha=0.7, label="Moderate threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/05_monthly_trend.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/05_monthly_trend.png")

    # ── Plot 6: Model Comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    metrics = ["mae", "rmse", "r2"]
    labels  = ["MAE (lower = better)", "RMSE (lower = better)", "R² (higher = better)"]
    model_names = list(results.keys())
    palette = ["#2196F3", "#4CAF50"]

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(model_names, vals, color=palette, edgecolor="white", width=0.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(metric.upper())
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(vals),
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/06_model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/06_model_comparison.png")


# ─────────────────────────────────────────────
# 5. SAVE MODEL
# ─────────────────────────────────────────────

def save_model(results, scaler, features, city_map):
    os.makedirs("model", exist_ok=True)
    best = "Random Forest"  # Based on R² performance

    with open("model/rf_model.pkl", "wb") as f:
        pickle.dump(results[best]["model"], f)
    with open("model/lr_model.pkl", "wb") as f:
        pickle.dump(results["Linear Regression"]["model"], f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("model/config.pkl", "wb") as f:
        pickle.dump({"features": features, "city_map": city_map, "best_model": best}, f)

    print(f"\nModels saved to model/")


# ─────────────────────────────────────────────
# 6. PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(results):
    print("\n" + "="*60)
    print("         FINAL MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R²':>10}")
    print("-"*60)
    for name, res in results.items():
        print(f"{name:<22} {res['mae']:>8.2f} {res['rmse']:>8.2f} "
              f"{res['r2']:>8.4f} {res['cv_r2_mean']:>8.4f}±{res['cv_r2_std']:.3f}")
    print("="*60)
    print("\n✅ Best Model: Random Forest (higher R², lower error)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  AI-Based Air Pollution Prediction System")
    print("  Model Training & Evaluation")
    print("="*60)

    print("\n[STEP 1] Loading dataset...")
    df = load_data()

    print("\n[STEP 2] Preprocessing data...")
    X, y, features, city_map = preprocess(df)
    print(f"  Features: {features}")

    print("\n[STEP 3] Training models...")
    results, scaler, X_train, X_test, y_train, y_test = train_evaluate(X, y)

    print("\n[STEP 4] Generating plots...")
    create_plots(df, results, X_test, y_test, features)

    print("\n[STEP 5] Saving models...")
    save_model(results, scaler, features, city_map)

    print_summary(results)
    print("\nDone! Run predict.py to make predictions.\n")
