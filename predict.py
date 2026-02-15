"""
AI-Based Air Pollution Prediction System
Prediction Script

Load trained models and predict AQI for new inputs.
"""

import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Load saved models
# ─────────────────────────────────────────────

def load_models():
    with open("model/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("model/lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/config.pkl", "rb") as f:
        config = pickle.load(f)
    return rf, lr, scaler, config


# ─────────────────────────────────────────────
# AQI Category & Health Advice
# ─────────────────────────────────────────────

def get_aqi_info(aqi):
    if aqi <= 50:
        cat    = "Good"
        color  = "\033[92m"   # green
        advice = "Air quality is good. Ideal for outdoor activities."
    elif aqi <= 100:
        cat    = "Satisfactory"
        color  = "\033[93m"
        advice = "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exposure."
    elif aqi <= 200:
        cat    = "Moderate"
        color  = "\033[33m"
        advice = "People with respiratory/heart issues should reduce outdoor activity."
    elif aqi <= 300:
        cat    = "Poor"
        color  = "\033[31m"
        advice = "Everyone may experience discomfort. Avoid outdoor activity."
    elif aqi <= 400:
        cat    = "Very Poor"
        color  = "\033[35m"
        advice = "Health alert! Limit all outdoor activity. Use masks."
    else:
        cat    = "Severe"
        color  = "\033[41m"
        advice = "Health emergency! Stay indoors. Avoid all outdoor activity."
    return cat, color, advice


# ─────────────────────────────────────────────
# Make a prediction
# ─────────────────────────────────────────────

def predict_aqi(inputs: dict, rf, lr, scaler, config):
    """
    inputs: dict with keys matching feature names.
    Returns: dict with predictions from both models.
    """
    features = config["features"]
    city_map = config["city_map"]

    # Encode city
    city_name = inputs.get("city", "Mathura")
    inputs["city_code"] = city_map.get(city_name, 0)

    # Build feature vector
    X = np.array([[inputs[f] for f in features]])
    X_df = pd.DataFrame(X, columns=features)

    X_scaled = scaler.transform(X_df)

    pred_rf = max(0, rf.predict(X_df)[0])
    pred_lr = max(0, lr.predict(X_scaled)[0])

    return {
        "Random Forest": round(pred_rf, 1),
        "Linear Regression": round(pred_lr, 1),
    }


# ─────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────

def run_interactive():
    reset = "\033[0m"
    bold  = "\033[1m"

    print(f"\n{bold}{'='*60}")
    print("  AI-Based Air Pollution Prediction System")
    print(f"{'='*60}{reset}")

    print("\nLoading trained models...")
    rf, lr, scaler, config = load_models()
    print("✅ Models loaded successfully!\n")

    cities = list(config["city_map"].keys())
    print(f"Available cities: {', '.join(cities)}\n")

    # ── Demo Scenarios ──
    scenarios = [
        {
            "name": "Delhi – Winter Morning (High Pollution)",
            "city": "Delhi",
            "pm25": 180, "pm10": 320, "no2": 85, "co": 2.1,
            "so2": 40, "o3": 15, "temperature": 12, "humidity": 75,
            "wind_speed": 2.5, "month": 12, "hour": 8
        },
        {
            "name": "Mathura – Monsoon Afternoon (Moderate)",
            "city": "Mathura",
            "pm25": 35, "pm10": 65, "no2": 28, "co": 0.6,
            "so2": 12, "o3": 42, "temperature": 30, "humidity": 85,
            "wind_speed": 14, "month": 7, "hour": 14
        },
        {
            "name": "Mumbai – Spring Evening (Good)",
            "city": "Mumbai",
            "pm25": 18, "pm10": 38, "no2": 15, "co": 0.3,
            "so2": 6, "o3": 52, "temperature": 28, "humidity": 70,
            "wind_speed": 18, "month": 3, "hour": 18
        },
    ]

    for scenario in scenarios:
        name = scenario.pop("name")
        print(f"{bold}── Scenario: {name}{reset}")

        preds = predict_aqi(scenario, rf, lr, scaler, config)
        aqi_rf = preds["Random Forest"]

        cat, color, advice = get_aqi_info(aqi_rf)

        print(f"  Input: PM2.5={scenario.get('pm25')} | PM10={scenario.get('pm10')} "
              f"| Temp={scenario.get('temperature')}°C | Humidity={scenario.get('humidity')}%")
        print(f"  {bold}Predicted AQI{reset}:")
        print(f"    • Random Forest:   {color}{bold}{preds['Random Forest']}{reset}")
        print(f"    • Linear Regress.: {preds['Linear Regression']}")
        print(f"  {color}Category: {cat}{reset}")
        print(f"  Health Advice: {advice}\n")
        scenario["name"] = name  # restore


def predict_single(pm25, pm10, no2, co, so2, o3, temperature,
                   humidity, wind_speed, month, hour, city="Mathura"):
    """Simple function for external use."""
    rf, lr, scaler, config = load_models()
    inputs = dict(pm25=pm25, pm10=pm10, no2=no2, co=co, so2=so2, o3=o3,
                  temperature=temperature, humidity=humidity,
                  wind_speed=wind_speed, month=month, hour=hour, city=city)
    return predict_aqi(inputs, rf, lr, scaler, config)


if __name__ == "__main__":
    run_interactive()
