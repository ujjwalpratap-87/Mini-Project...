# AI-Based Air Pollution Prediction System
## B.Tech Project — GLA University, Mathura (2025–26)

### Team
| Name | Roll No. | Role |
|------|----------|------|
| Shakti Kumar Dubey | 2415001460 | ML Engineering |
| Ujjwal Pratap Singh | 2415001702 | Data Analysis |
| Nityanand | 2415001063 | Data Collection |
| Krishna Pandey | 2415000844 | Backend Dev |
| Harsh Kaushik | 2415000640 | Frontend / UI |

**Supervisor:** Dr. Sujatha Jayaraj

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn streamlit
```

### Step 2 — Generate dataset
```bash
python generate_data.py
```

### Step 3 — Train ML models
```bash
python train_model.py
```
This trains Linear Regression + Random Forest, evaluates them, and saves plots to `/plots/` and models to `/model/`.

### Step 4 — Predict AQI
```bash
python predict.py
```
Runs 3 demo scenarios and prints AQI predictions with health advice.

### Step 5 — Open interactive dashboard
Open `AirPollution_Dashboard.html` in any browser — **no server needed!**

---

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 17.00 | 25.71 | 0.9494 |
| **Random Forest** | **4.71** | **6.05** | **0.9972** |

**Best Model: Random Forest** with R² = 0.9972

---

## Features Used
PM2.5, PM10, NO₂, CO, SO₂, O₃, Temperature, Humidity, Wind Speed, Month, Hour, City

## Dataset
- 2,000 records across 8 Indian cities
- Time span: Jan 2022 – Dec 2023
- Sources: CPCB, OpenAQ, Kaggle
