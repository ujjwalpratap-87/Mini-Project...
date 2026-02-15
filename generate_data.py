"""
AI-Based Air Pollution Prediction System
Dataset Generation Script

Generates a synthetic but realistic dataset simulating
CPCB-style air quality data for Indian cities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_dataset(n_samples=2000):
    """Generate realistic air pollution dataset for Indian cities."""
    
    cities = ["Delhi", "Mumbai", "Lucknow", "Kanpur", "Agra", "Mathura", "Varanasi", "Allahabad"]
    
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(hours=i*12) for i in range(n_samples)]

    records = []

    for i, date in enumerate(dates):
        city = cities[i % len(cities)]
        month = date.month
        hour = date.hour

        # Seasonal factors: higher pollution in winter (Oct-Feb)
        if month in [11, 12, 1, 2]:
            season_factor = 1.8
        elif month in [3, 4, 10]:
            season_factor = 1.3
        elif month in [5, 6]:
            season_factor = 1.0
        else:
            season_factor = 0.7  # Monsoon cleans air

        # Rush hour factor
        if hour in [7, 8, 9, 17, 18, 19]:
            traffic_factor = 1.4
        elif hour in [0, 1, 2, 3, 4]:
            traffic_factor = 0.6
        else:
            traffic_factor = 1.0

        # City-specific base pollution (Delhi worst, Mathura moderate)
        city_base = {
            "Delhi": 1.5, "Kanpur": 1.4, "Lucknow": 1.3,
            "Agra": 1.2, "Varanasi": 1.2, "Allahabad": 1.1,
            "Mathura": 1.0, "Mumbai": 0.9
        }[city]

        base = city_base * season_factor * traffic_factor

        # Environmental parameters
        temp = np.random.normal(
            25 - 15 * np.cos(2 * np.pi * month / 12), 3
        )
        humidity = np.random.normal(
            60 + 25 * np.sin(2 * np.pi * month / 12), 10
        )
        humidity = np.clip(humidity, 20, 100)
        wind_speed = np.random.exponential(8) * (1 / base)
        wind_speed = np.clip(wind_speed, 0.5, 40)

        # Pollutant concentrations (µg/m³ or ppm)
        pm25 = np.random.gamma(shape=3, scale=20 * base) + np.random.normal(0, 5)
        pm10 = pm25 * np.random.uniform(1.5, 2.5) + np.random.normal(0, 10)
        no2 = np.random.gamma(shape=2, scale=15 * base) + np.random.normal(0, 3)
        co = np.random.gamma(shape=2, scale=0.5 * base) + np.random.normal(0, 0.1)
        so2 = np.random.gamma(shape=1.5, scale=10 * base) + np.random.normal(0, 2)
        o3 = np.random.gamma(shape=2, scale=20) * (1.2 - 0.2 * base) + np.random.normal(0, 5)

        # Clip to realistic ranges
        pm25 = max(5, pm25)
        pm10 = max(10, pm10)
        no2 = max(2, no2)
        co = max(0.1, co)
        so2 = max(1, so2)
        o3 = max(5, o3)

        # Calculate AQI (simplified India NAQI formula)
        def sub_index(conc, breakpoints):
            """Calculate sub-index for a pollutant."""
            bp_lo, bp_hi, aqi_lo, aqi_hi = breakpoints
            return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (conc - bp_lo) + aqi_lo

        # PM2.5 sub-index
        if pm25 <= 30:
            pm25_idx = sub_index(pm25, [0, 30, 0, 50])
        elif pm25 <= 60:
            pm25_idx = sub_index(pm25, [30, 60, 51, 100])
        elif pm25 <= 90:
            pm25_idx = sub_index(pm25, [60, 90, 101, 200])
        elif pm25 <= 120:
            pm25_idx = sub_index(pm25, [90, 120, 201, 300])
        else:
            pm25_idx = min(500, sub_index(pm25, [120, 250, 301, 500]))

        # PM10 sub-index
        if pm10 <= 50:
            pm10_idx = sub_index(pm10, [0, 50, 0, 50])
        elif pm10 <= 100:
            pm10_idx = sub_index(pm10, [50, 100, 51, 100])
        elif pm10 <= 250:
            pm10_idx = sub_index(pm10, [100, 250, 101, 200])
        elif pm10 <= 350:
            pm10_idx = sub_index(pm10, [250, 350, 201, 300])
        else:
            pm10_idx = min(500, sub_index(pm10, [350, 430, 301, 400]))

        aqi = max(pm25_idx, pm10_idx)
        aqi = round(np.clip(aqi + np.random.normal(0, 5), 0, 500), 1)

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "time": date.strftime("%H:%M"),
            "city": city,
            "month": month,
            "hour": hour,
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "pm25": round(pm25, 1),
            "pm10": round(pm10, 1),
            "no2": round(no2, 1),
            "co": round(co, 2),
            "so2": round(so2, 1),
            "o3": round(o3, 1),
            "aqi": aqi
        })

    df = pd.DataFrame(records)
    return df


def categorize_aqi(aqi):
    """Return AQI category label."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


if __name__ == "__main__":
    print("Generating air pollution dataset...")
    df = generate_dataset(2000)
    df["aqi_category"] = df["aqi"].apply(categorize_aqi)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/air_quality_data.csv", index=False)

    print(f"Dataset generated: {len(df)} records")
    print(f"\nShape: {df.shape}")
    print(f"\nAQI Statistics:")
    print(df["aqi"].describe().round(2))
    print(f"\nAQI Category Distribution:")
    print(df["aqi_category"].value_counts())
    print(f"\nFirst 5 rows:")
    print(df.head())
    print("\nDataset saved to data/air_quality_data.csv")
