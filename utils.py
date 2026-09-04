import os
import json
import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# Project Folder
# ----------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(APP_DIR, "Dataset.csv")
MODEL_PATH = os.path.join(APP_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(APP_DIR, "scaler.pkl")
RESULT_PATH = os.path.join(APP_DIR, "model_results.json")


# ----------------------------
# Load Dataset
# ----------------------------

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df


# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ----------------------------
# Load Model Results
# ----------------------------

@st.cache_data
def load_model_results():

    if os.path.exists(RESULT_PATH):

        with open(RESULT_PATH) as f:
            data = json.load(f)

        results = pd.DataFrame([
            {
                "Model": k,
                "MAE": v["MAE"],
                "RMSE": v["RMSE"],
                "R²": v["R2"]
            }

            for k, v in data.items()
        ])

        return results

    return pd.DataFrame()


# ----------------------------
# Dashboard KPIs
# ----------------------------

def get_dashboard_metrics(df):

    metrics = {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Total Rentals": int(df["cnt"].sum()),

        "Average Rentals": round(df["cnt"].mean(),2),

        "Maximum Rentals": int(df["cnt"].max()),

        "Minimum Rentals": int(df["cnt"].min()),

        "Working Days": int(df["workingday"].sum()),

        "Holidays": int(df["holiday"].sum()),

        "Registered Users": int(df["registered"].sum()),

        "Casual Users": int(df["casual"].sum())

    }

    return metrics


# ----------------------------
# Missing Values
# ----------------------------

def missing_values(df):

    return pd.DataFrame(

        df.isnull().sum(),

        columns=["Missing Values"]

    )


# ----------------------------
# Dataset Summary
# ----------------------------

def dataset_summary(df):

    return df.describe()


# ----------------------------
# Search Dataset
# ----------------------------

def search_dataset(df, keyword):

    if keyword == "":
        return df

    result = df[
        df.astype(str)
          .apply(lambda x: x.str.contains(keyword,
                                          case=False,
                                          na=False))
          .any(axis=1)
    ]

    return result


# ----------------------------
# Filter Dataset
# ----------------------------

def filter_dataset(df, year=None, season=None):

    data = df.copy()

    if year is not None:
        data = data[data["yr"] == year]

    if season is not None:
        data = data[data["season"] == season]

    return data


# ----------------------------
# Download CSV
# ----------------------------

def convert_csv(df):

    return df.to_csv(index=False).encode("utf-8")


# ----------------------------
# Top Rentals
# ----------------------------

def top_rentals(df):

    return df.sort_values(

        "cnt",

        ascending=False

    ).head(10)


# ----------------------------
# Bottom Rentals
# ----------------------------

def bottom_rentals(df):

    return df.sort_values(

        "cnt"

    ).head(10)