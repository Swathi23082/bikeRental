import os
import json
import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = joblib.load(os.path.join(APP_DIR, "best_model.pkl"))
    s = joblib.load(os.path.join(APP_DIR, "scaler.pkl"))
    return m, s

@st.cache_data
def load_results():
    path = os.path.join(APP_DIR, "model_results.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return sorted(
            [{"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
             for k, v in raw.items()],
            key=lambda r: r["RMSE"],
        )
    # Fallback — original 6 models from notebook output
    return [
        {"Model": "GB (Tuned)",        "MAE": 21.70, "RMSE": 35.69, "R²": 0.9598},
        {"Model": "Gradient Boosting", "MAE": 25.06, "RMSE": 39.68, "R²": 0.9503},
        {"Model": "Random Forest",     "MAE": 24.22, "RMSE": 40.98, "R²": 0.9470},
        {"Model": "DT (Tuned)",        "MAE": 31.66, "RMSE": 54.35, "R²": 0.9067},
        {"Model": "RF (Tuned)",        "MAE": 38.01, "RMSE": 58.13, "R²": 0.8933},
        {"Model": "Decision Tree",     "MAE": 34.08, "RMSE": 59.03, "R²": 0.8900},
    ]

try:
    model, scaler = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚲 Bike Rental Demand Predictor")
st.markdown(
    "Predict **hourly bike rentals** for the Capital Bikeshare system "
    "(Washington D.C., 2011–2012 dataset) using a tuned Gradient Boosting model."
)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Input Parameters")

    # ── Date & time ──
    st.subheader("📅 Date & Time")
    selected_date = st.date_input(
        "Date",
        value=datetime.date(2012, 6, 15),
        min_value=datetime.date(2011, 1, 1),
        max_value=datetime.date(2012, 12, 31),
        help="Model trained on 2011–2012 data only.",
    )
    hour = st.slider("Hour of Day", 0, 23, 12, format="%d:00")

    # ── Season & weather ──
    st.subheader("🌤 Season & Weather")
    season = st.selectbox("Season", ["Fall", "Spring", "Summer", "Winter"])
    weather = st.selectbox(
        "Weather Condition",
        ["Clear / Few Clouds", "Mist / Cloudy", "Light Snow / Rain", "Heavy Rain / Thunderstorm"],
    )

    # ── Numerical weather ──
    st.subheader("🌡 Temperature & Conditions")
    temp_c   = st.slider("Temperature (°C)",         -8.0, 41.0, 20.0, 0.5)
    atemp_c  = st.slider("Feels-Like Temp (°C)",    -16.0, 50.0, 18.0, 0.5)
    humidity = st.slider("Humidity (%)",               0,  100,  60)
    wind_kmh = st.slider("Wind Speed (km/h)",         0.0, 57.0, 15.0, 0.5)

    # ── Calendar ──
    st.subheader("📆 Calendar Flags")
    workingday = st.checkbox("Working Day",     value=True)
    holiday    = st.checkbox("Public Holiday",  value=False)

# ── Feature engineering ───────────────────────────────────────────────────────
yr          = float(selected_date.year)
mnth        = float(selected_date.month)
py_wd       = selected_date.weekday()       # Python: Mon=0 … Sun=6
weekday_ds  = float((py_wd + 1) % 7)       # dataset: Sun=0 … Sat=6
day_of_year = float(selected_date.timetuple().tm_yday)
is_weekend  = 1.0 if int(weekday_ds) in (0, 6) else 0.0

# Normalise (Capital Bikeshare dataset divisors)
temp_norm  = float(np.clip(temp_c   / 41.0,  0, 1))
atemp_norm = float(np.clip(atemp_c  / 50.0,  0, 1))
hum_norm   = float(np.clip(humidity / 100.0, 0, 1))
wind_norm  = float(np.clip(wind_kmh / 57.0,  0, 1))

# One-hot season (fall = baseline, dropped by get_dummies)
season_springer = 1.0 if season == "Spring" else 0.0
season_summer   = 1.0 if season == "Summer" else 0.0
season_winter   = 1.0 if season == "Winter" else 0.0

# One-hot weather (clear = baseline, dropped)
ws_heavy = 1.0 if weather == "Heavy Rain / Thunderstorm" else 0.0
ws_snow  = 1.0 if weather == "Light Snow / Rain"          else 0.0
ws_mist  = 1.0 if weather == "Mist / Cloudy"              else 0.0

# Working day / holiday — a holiday is never a working day
wd_flag  = 1.0 if (workingday and not holiday) else 0.0
hol_flag = 1.0 if holiday else 0.0

# Column order must exactly match the training DataFrame X
FEATURE_COLS = [
    "yr", "mnth", "hr", "weekday", "temp", "atemp", "hum", "windspeed",
    "day_of_year", "is_weekend",
    "season_springer", "season_summer", "season_winter",
    "weathersit_Heavy Rain", "weathersit_Light Snow", "weathersit_Mist",
    "workingday_Working Day", "holiday_Yes",
]

input_df = pd.DataFrame([{
    "yr": yr, "mnth": mnth, "hr": float(hour), "weekday": weekday_ds,
    "temp": temp_norm, "atemp": atemp_norm, "hum": hum_norm, "windspeed": wind_norm,
    "day_of_year": day_of_year, "is_weekend": is_weekend,
    "season_springer": season_springer, "season_summer": season_summer,
    "season_winter": season_winter,
    "weathersit_Heavy Rain": ws_heavy, "weathersit_Light Snow": ws_snow,
    "weathersit_Mist": ws_mist,
    "workingday_Working Day": wd_flag, "holiday_Yes": hol_flag,
}])[FEATURE_COLS]

if model_loaded:
    NUM_FEATS = ["temp", "atemp", "hum", "windspeed"]
    input_df[NUM_FEATS] = scaler.transform(input_df[NUM_FEATS])
    prediction = max(0, int(round(model.predict(input_df)[0])))
else:
    prediction = None

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Comparison", "ℹ️ About"])

# ───────────────────────────────────── TAB 1 ──────────────────────────────────
with tab1:
    if not model_loaded:
        st.error(
            "**Model files not found.**  "
            "Run the Jupyter notebook first to generate `best_model.pkl`, `scaler.pkl`, and "
            "`model_results.json`, then place them in the same folder as `app.py`."
        )
    else:
        # Demand label
        if prediction < 50:
            lvl, badge, bar_color = "Low",       "🟢", "#2ecc71"
        elif prediction < 150:
            lvl, badge, bar_color = "Moderate",  "🟡", "#f1c40f"
        elif prediction < 350:
            lvl, badge, bar_color = "High",      "🟠", "#e67e22"
        else:
            lvl, badge, bar_color = "Very High", "🔴", "#e74c3c"

        # Time-of-day label
        if   7 <= hour <= 9:   period = "Morning Rush Hour"
        elif 17 <= hour <= 19: period = "Evening Rush Hour"
        elif 10 <= hour <= 16: period = "Midday"
        elif 20 <= hour <= 23: period = "Evening"
        else:                  period = "Night / Early AM"

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Rentals / Hour", f"{prediction:,}")
        c2.metric("Demand Level",   f"{badge} {lvl}")
        c3.metric("Time Period",    period)
        c4.metric("Day of Week",    selected_date.strftime("%A"))

        st.divider()

        # Demand gauge
        MAX_SCALE = 800
        pct = min(prediction / MAX_SCALE, 1.0)
        st.markdown("**Demand gauge** (scale: 0 → 800; dataset peak ≈ 977)")
        st.markdown(
            f'<div style="background:#e0e0e0;border-radius:10px;height:32px;overflow:hidden">'
            f'<div style="width:{pct*100:.1f}%;background:{bar_color};height:100%;'
            f'display:flex;align-items:center;padding-left:12px;'
            f'color:white;font-weight:bold;font-size:14px">'
            f'{prediction} rentals</div></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Input summary grid
        st.subheader("Current Input Summary")
        items = [
            ("Date",         selected_date.strftime("%b %d, %Y")),
            ("Hour",         f"{hour:02d}:00"),
            ("Season",       season),
            ("Weather",      weather),
            ("Temperature",  f"{temp_c:.1f} °C (feels {atemp_c:.1f} °C)"),
            ("Humidity",     f"{humidity} %"),
            ("Wind Speed",   f"{wind_kmh:.1f} km/h"),
            ("Working Day",  "Yes" if workingday else "No"),
            ("Holiday",      "Yes" if holiday else "No"),
            ("Weekend",      "Yes" if is_weekend else "No"),
        ]
        cols = st.columns(5)
        for i, (k, v) in enumerate(items):
            cols[i % 5].markdown(f"**{k}**  \n{v}")

# ───────────────────────────────────── TAB 2 ──────────────────────────────────
with tab2:
    st.subheader("All Models — Test-Set Performance")

    results_data = load_results()
    df_res = pd.DataFrame(results_data)

    best_rmse = df_res["RMSE"].min()
    best_r2   = df_res["R²"].max()

    def _highlight(row):
        if row["RMSE"] == best_rmse:
            return ["background-color:#d4edda;font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_res.style
              .apply(_highlight, axis=1)
              .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    # Side-by-side charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    c_rmse = ["#1565C0" if r == best_rmse else "#90CAF9" for r in df_res["RMSE"]]
    ax1.barh(df_res["Model"], df_res["RMSE"], color=c_rmse, edgecolor="white")
    ax1.set_xlabel("RMSE  (lower is better)")
    ax1.set_title("RMSE Comparison")
    ax1.invert_yaxis()
    for bar, val in zip(ax1.patches, df_res["RMSE"]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=8)

    c_r2 = ["#2E7D32" if r == best_r2 else "#A5D6A7" for r in df_res["R²"]]
    ax2.barh(df_res["Model"], df_res["R²"], color=c_r2, edgecolor="white")
    ax2.set_xlabel("R²  (higher is better)")
    ax2.set_title("R² Comparison")
    ax2.invert_yaxis()
    ax2.set_xlim(df_res["R²"].min() - 0.02, 1.01)
    for bar, val in zip(ax2.patches, df_res["R²"]):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    best_row = df_res[df_res["RMSE"] == best_rmse].iloc[0]
    st.success(
        f"**Best model:** {best_row['Model']}  —  "
        f"MAE = {best_row['MAE']:.2f}, RMSE = {best_row['RMSE']:.2f}, R² = {best_row['R²']:.4f}"
    )

# ───────────────────────────────────── TAB 3 ──────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
**Dataset:** Capital Bikeshare — Washington D.C. (2011–2012)
- **17,379** hourly records · **Target:** `cnt` (total rentals = casual + registered)

---

### Models Trained & Compared

| # | Model | Type |
|---|---|---|
| 1 | Decision Tree | Single decision tree |
| 2 | Random Forest | Bagging ensemble |
| 3 | Gradient Boosting | Sequential boosting |
| 4 | Ridge Regression | Linear with L2 regularisation |
| 5 | Extra Trees | Randomised tree ensemble |
| 6 | XGBoost | Extreme gradient boosting |

Each model is also hyperparameter-tuned (GridSearchCV / RandomizedSearchCV, 5-fold CV).

---

### Key Findings
- **Hour of day** is the strongest predictor — bimodal commuter peaks at 8 am & 5–6 pm on working days
- **Year** captures business growth (2012 >> 2011)
- **Temperature** and **season** have strong positive effects
- **Weather** has a significant negative effect (rain/snow → far fewer riders)

---

### Files Required
```
best_model.pkl       ← trained best model  (saved by notebook Step 12)
scaler.pkl           ← fitted MinMaxScaler  (saved by notebook Step 12)
model_results.json   ← all model metrics   (saved by notebook Step 12)
app.py               ← this Streamlit app
```

Run the notebook fully, ensure all four files are in the same folder, then:
```bash
streamlit run app.py
```
""")
