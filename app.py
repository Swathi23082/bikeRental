import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Bike Rental Demand Prediction Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

with st.spinner("Loading Dashboard..."):
    time.sleep(1)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "Dataset.csv")

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(APP_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(APP_DIR, "scaler.pkl"))
    return model, scaler

@st.cache_data
def load_results():
    f = os.path.join(APP_DIR, "model_results.json")
    if os.path.exists(f):
        with open(f) as fp:
            raw = json.load(fp)
        return pd.DataFrame([
            {"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]}
            for k, v in raw.items()
        ])
    return pd.DataFrame()

df = load_dataset()
model, scaler = load_model()
results = load_results()

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🚲 Bike Rental Demand Prediction Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "📂 Dataset",
        "🔮 Prediction",
        "📈 Model Comparison",
        "ℹ️ About"
    ]
)

df["registered"] = pd.to_numeric(df["registered"], errors="coerce")
df["casual"] = pd.to_numeric(df["casual"], errors="coerce")
df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce")

registered = int(df["registered"].sum())
casual = int(df["casual"].sum())
total_rentals = int(df["cnt"].sum())

# Global stats reused across tabs (esp. the Advanced Prediction tab)
avg_rentals_all = round(df["cnt"].mean(), 2)
max_rentals_all = int(df["cnt"].max())
hourly_avg_all = df.groupby("hr")["cnt"].mean()
registered_ratio = registered / total_rentals if total_rentals else 0.8
casual_ratio = casual / total_rentals if total_rentals else 0.2

with tab1:

    st.caption("Real-time dashboard showing bike rental trends and statistics.")

    # ============================
    # KPI Cards
    # ============================

    total_rentals = int(df["cnt"].sum())
    avg_rentals = round(df["cnt"].mean(), 2)
    max_rentals = int(df["cnt"].max())
    min_rentals = int(df["cnt"].min())
    registered = int(df["registered"].sum())
    casual = int(df["casual"].sum())

    c1, c2, c3 = st.columns(3)

    c1.metric("🚲 Total Rentals", f"{total_rentals:,}", delta="Overall")
    c2.metric("📈 Average Rentals", avg_rentals)
    c3.metric("🔥 Maximum Rentals", max_rentals)

    c4, c5, c6 = st.columns(3)

    c4.metric("❄ Minimum Rentals", min_rentals)
    c5.metric("👤 Registered Users", f"{registered:,}")
    c6.metric("😊 Casual Users", f"{casual:,}")

    st.divider()

    # ====================================
    # Row 1 — two charts, fixed 50% / 50% width each
    # ====================================

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("📈 Hourly Bike Rentals")

        hourly = df.groupby("hr")["cnt"].mean().reset_index()

        fig = px.line(
            hourly,
            x="hr",
            y="cnt",
            markers=True,
            template="plotly_white",
            color_discrete_sequence=["#2563EB"]
        )

        fig.update_layout(
            height=480,
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            # title=None,
            xaxis_title="Hour",
            yaxis_title="Average Rentals"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False}
        )

    with col2:
        st.subheader("🥧 Season Distribution")

        season = df.groupby("season")["cnt"].sum().reset_index()

        fig = px.pie(
            season,
            names="season",
            values="cnt",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            height=480,
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=True
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False}
        )

    st.divider()

    # ====================================
    # Row 2
    # ====================================

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.subheader("🌡 Temperature vs Rentals")

        fig = px.scatter(
            df,
            x="temp",
            y="cnt",
            color="cnt",
            size="cnt",
            opacity=.6,
            template="plotly_dark"
        )

        fig.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))

        st.plotly_chart(fig, width="stretch")

    with col4:
        st.subheader("🌤 Weather Analysis")

        weather = df.groupby("weathersit")["cnt"].mean().reset_index()

        fig = px.bar(
            weather,
            x="weathersit",
            y="cnt",
            color="cnt",
            template="plotly_dark"
        )

        fig.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))

        st.plotly_chart(fig, width="stretch")

    st.divider()

    # ====================================
    # Row 3
    # ====================================

    col5, col6 = st.columns(2, gap="medium")

    with col5:
        st.subheader("🔥 Top 10 Busy Hours")

        top = df.groupby("hr")["cnt"].mean().sort_values(
            ascending=False
        ).head(10).reset_index()

        fig = px.bar(
            top,
            x="hr",
            y="cnt",
            color="cnt",
            template="plotly_dark"
        )

        st.plotly_chart(fig, width="stretch")

    with col6:
        st.subheader("📅 Monthly Rentals")

        month = df.groupby("mnth")["cnt"].mean().reset_index()

        fig = px.area(
            month,
            x="mnth",
            y="cnt",
            template="plotly_dark"
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("📂 Dataset Preview")

    st.dataframe(df.head(10), width="stretch")


with tab2:
    st.header("Dataset")
    search = st.text_input("Search")
    data = df
    if search:
        data = df[df.astype(str).apply(lambda s: s.str.contains(search, case=False)).any(axis=1)]
    st.dataframe(data, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), "Dataset.csv", "text/csv")
    st.subheader("Statistics")
    st.dataframe(df.describe(), width="stretch")
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().to_frame("Missing"), width="stretch")

with tab3:

    st.header("🔮 Advanced Bike Rental Prediction")
    st.caption(
        "Configure the date, time, weather and day-type conditions below "
        "to get an AI-powered rental demand forecast with context."
    )

    st.divider()

    WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    with st.form("prediction_form"):

        st.subheader("📅 Date & Time")
        d1, d2, d3 = st.columns(3)

        with d1:
            year = st.selectbox("Year", [2011, 2012])
            month_name = st.select_slider("Month", options=MONTH_NAMES, value="Jun")
            month = MONTH_NAMES.index(month_name) + 1

        with d2:
            hour = st.slider("Hour of Day", 0, 23, 12)
            weekday_name = st.select_slider("Weekday", options=WEEKDAY_NAMES, value="Monday")
            weekday = WEEKDAY_NAMES.index(weekday_name)

        with d3:
            working = st.checkbox("Working Day", True)
            holiday = st.checkbox("Holiday", False)

        st.divider()

        st.subheader("🌦 Season & Weather")
        w1, w2 = st.columns(2)

        with w1:
            season = st.selectbox("Season", ["Fall", "Spring", "Summer", "Winter"])
            weather = st.selectbox("Weather Condition", ["Clear", "Mist", "Light Snow", "Heavy Rain"])

        with w2:
            temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
            atemp = st.slider("Feels-Like Temperature (normalized)", 0.0, 1.0, 0.5)

        w3, w4 = st.columns(2)

        with w3:
            hum = st.slider("Humidity", 0.0, 1.0, 0.5)

        with w4:
            wind = st.slider("Wind Speed", 0.0, 1.0, 0.2)

        st.write("")
        submitted = st.form_submit_button("🔮 Predict Rental Demand", width="stretch")

    if submitted:

        season_spring = 1 if season == "Spring" else 0
        season_summer = 1 if season == "Summer" else 0
        season_winter = 1 if season == "Winter" else 0
        ws_heavy = 1 if weather == "Heavy Rain" else 0
        ws_snow = 1 if weather == "Light Snow" else 0
        ws_mist = 1 if weather == "Mist" else 0

        num_df = pd.DataFrame(
            [[temp, atemp, hum, wind]],
            columns=["temp", "atemp", "hum", "windspeed"]
        )
        num = scaler.transform(num_df)[0]

        X = pd.DataFrame([{
            "yr": float(year),
            "mnth": float(month),
            "hr": float(hour),
            "weekday": float(weekday),
            "temp": num[0],
            "atemp": num[1],
            "hum": num[2],
            "windspeed": num[3],
            "day_of_year": 180,
            "is_weekend": 1 if weekday in [0, 6] else 0,
            "season_springer": season_spring,
            "season_summer": season_summer,
            "season_winter": season_winter,
            "weathersit_Heavy Rain": ws_heavy,
            "weathersit_Light Snow": ws_snow,
            "weathersit_Mist": ws_mist,
            "workingday_Working Day": 1 if working else 0,
            "holiday_Yes": 1 if holiday else 0
        }])

        pred = max(0, int(model.predict(X)[0]))

        # Demand level classification relative to overall dataset distribution
        if pred < avg_rentals_all * 0.5:
            level, color, emoji = "Low Demand", "#DC2626", "🔴"
        elif pred < avg_rentals_all * 1.2:
            level, color, emoji = "Moderate Demand", "#F59E0B", "🟠"
        else:
            level, color, emoji = "High Demand", "#16A34A", "🟢"

        est_registered = int(pred * registered_ratio)
        est_casual = pred - est_registered

        hist_hour_avg = round(float(hourly_avg_all.get(hour, avg_rentals_all)), 1)
        diff_vs_hist = pred - hist_hour_avg
        diff_pct = (diff_vs_hist / hist_hour_avg * 100) if hist_hour_avg else 0

        st.divider()
        st.subheader("📊 Prediction Result")

        r1, r2, r3 = st.columns([1.3, 1, 1])

        with r1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"suffix": " bikes", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, max(max_rentals_all, pred + 50)]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, avg_rentals_all * 0.5], "color": "#FEE2E2"},
                        {"range": [avg_rentals_all * 0.5, avg_rentals_all * 1.2], "color": "#FEF3C7"},
                        {"range": [avg_rentals_all * 1.2, max(max_rentals_all, pred + 50)], "color": "#DCFCE7"}
                    ],
                    "threshold": {
                        "line": {"color": "#1D4ED8", "width": 4},
                        "thickness": 0.8,
                        "value": avg_rentals_all
                    }
                },
                title={"text": "Predicted Rentals"}
            ))
            gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

        with r2:
            st.markdown(f"### {emoji} {level}")
            st.metric("Predicted Rentals", f"{pred:,}")
            st.metric(
                "vs. Historical Avg (this hour)",
                f"{hist_hour_avg:,.1f}",
                delta=f"{diff_vs_hist:+.0f} ({diff_pct:+.1f}%)"
            )

        with r3:
            st.markdown("### 👥 Estimated User Split")
            st.metric("Registered Users", f"{est_registered:,}")
            st.metric("Casual Users", f"{est_casual:,}")

        st.divider()

        st.subheader("📈 Predicted vs. Historical Pattern (by Hour)")

        chart_df = hourly_avg_all.reset_index()
        chart_df.columns = ["hr", "cnt"]

        fig = px.bar(
            chart_df,
            x="hr",
            y="cnt",
            template="plotly_white",
            color_discrete_sequence=["#CBD5E1"]
        )
        fig.add_scatter(
            x=[hour],
            y=[pred],
            mode="markers+text",
            marker=dict(size=16, color=color),
            text=["Your Prediction"],
            textposition="top center",
            name="Prediction"
        )
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Hour of Day",
            yaxis_title="Average Rentals",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with st.expander("🔍 View Model Input Features"):
            st.dataframe(X, width="stretch")

with tab4:
    st.header("Model Comparison")
    if not results.empty:
        st.dataframe(results, width="stretch")


with tab5:

    st.markdown("""
    ### Intelligent Bike Rental Forecasting using Machine Learning
    """)

    st.write(
        "This dashboard predicts hourly bike rental demand using Machine Learning "
        "and provides interactive analytics for better decision-making."
    )

    st.divider()

    st.header("📌 Project Overview")

    st.info(
        "This project predicts bike rental demand based on weather, season, "
        "date and time information."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Objective")
        st.write("""
Predict hourly bike rental demand accurately using Machine Learning
to help transport services improve bike availability.
        """)

    with col2:
        st.subheader("🌍 Real World Application")
        st.write("""
Bike sharing companies can estimate future demand and manage
their bikes more efficiently.
        """)

    st.divider()

    st.header("📂 Dataset")

    st.success("Capital Bikeshare Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", "17,379")
    c2.metric("Columns", "17")
    c3.metric("Target", "cnt")

    st.write("""
The dataset contains hourly bike rental information collected
during 2011 and 2012 in Washington D.C.
    """)

    st.divider()

    st.header("📋 Input Features")

    st.write("""
The prediction model uses the following features.
    """)

    features = pd.DataFrame({
        "Feature": [
            "Year", "Month", "Hour", "Weekday", "Season", "Weather",
            "Temperature", "Feels Like Temperature", "Humidity",
            "Wind Speed", "Working Day", "Holiday"
        ],
        "Description": [
            "Rental Year", "Rental Month", "Rental Hour", "Day of Week",
            "Current Season", "Weather Condition", "Actual Temperature",
            "Perceived Temperature", "Humidity Percentage", "Wind Speed",
            "Working Day Indicator", "Holiday Indicator"
        ]
    })

    st.dataframe(features, width="stretch")

    st.divider()

    st.header("🤖 Machine Learning Models")

    models = pd.DataFrame({
        "Model": [
            "Decision Tree", "Random Forest", "Gradient Boosting",
            "Ridge Regression", "Extra Trees", "XGBoost"
        ],
        "Purpose": [
            "Tree Model", "Bagging", "Boosting", "Linear Model",
            "Random Trees", "Advanced Boosting"
        ]
    })

    st.dataframe(models, width="stretch")

    st.success("Gradient Boosting achieved the best prediction accuracy.")

    st.divider()

    st.header("⚙ Technology Stack")

    tech = pd.DataFrame({
        "Technology": [
            "Python", "Pandas", "NumPy", "Scikit-Learn",
            "Matplotlib", "Streamlit", "Joblib"
        ],
        "Purpose": [
            "Programming", "Data Processing", "Numerical Computing",
            "Machine Learning", "Visualization", "Dashboard", "Model Storage"
        ]
    })

    st.dataframe(tech, width="stretch")

    st.divider()

    st.header("📊 Project Workflow")

    st.markdown("""
1️⃣ Load Dataset

⬇

2️⃣ Data Cleaning

⬇

3️⃣ Data Preprocessing

⬇

4️⃣ Feature Engineering

⬇

5️⃣ Train Machine Learning Models

⬇

6️⃣ Hyperparameter Tuning

⬇

7️⃣ Model Evaluation

⬇

8️⃣ Save Best Model

⬇

9️⃣ Streamlit Deployment

⬇

🔟 Bike Rental Prediction
""")

    st.divider()

    st.header("⭐ Advantages")

    st.markdown("""
✅ Fast Prediction

✅ Interactive Dashboard

✅ Easy to Use

✅ Business Insights

✅ Accurate Results

✅ Machine Learning Based
""")

    st.divider()

    st.header("🚀 Future Scope")

    st.markdown("""
• Real-Time Weather API

• Mobile Application

• Cloud Deployment

• Deep Learning Models

• Live Prediction

• User Login System

• Google Maps Integration
""")

    st.divider()

    st.header("👨‍🎓 Developer")

    st.info("""
**Developed By**

Sandhya

**Project**

Bike Rental Demand Prediction Dashboard

**Course**

Master of Computer Applications (MCA)

**Technology**

Machine Learning + Streamlit

**Language**

Python
""")

    st.success("Thank you for visiting the Bike Rental Dashboard 🚲")