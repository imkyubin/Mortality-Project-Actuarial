"""TEMPLATE - cached data loaders and the credibility lookup the dashboard views read from.

Kept free of layout code so the same lookup logic can be reused by the API and
reporting layers (api_layer.credibility_gate in config.yaml). Deaths/Population are
loaded only to derive rates; views must never display them.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.step2.life_table import add_life_tab_col
from src.step2.load_filter import age_value, load_filter

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

SEX_LABELS = {"F": "Female", "M": "Male", "T": "Total"}

CAUSE_LABELS = {
    "All": "All causes",
    "#Diseases of heart (I00-I09,I11,I13,I20-I51)": "Heart disease",
    "#Malignant neoplasms (C00-C97)": "Cancer",
    "#Cerebrovascular diseases (I60-I69)": "Cerebrovascular disease",
    "#Chronic lower respiratory diseases (J40-J47)": "COPD",
    "#Diabetes mellitus (E10-E14)": "Diabetes mellitus",
    "#Alzheimer disease (G30)": "Alzheimer's disease",
    "#Influenza and pneumonia (J09-J18)": "Influenza & Pneumonia",
    "#Chronic liver disease and cirrhosis (K70,K73-K74)": "Chronic liver disease & cirrhosis",
}

LOCATIONS = ["United States", "California", "Texas", "Florida", "New York"]

CREDIBILITY_STANDARD = 1082


@st.cache_data
def load_historic_life_table():
    """Harmonized 1968-2024 national data, 20-84 grouped ages, with life-table columns."""
    df = load_filter(PROCESSED / "harmonized_1968-2024.csv", age_col="Age Group")
    df = add_life_tab_col(df)
    df["Death Rate"] = df["Deaths"] / df["Population"] * 100000
    df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_value, ordered=True)
    return df


@st.cache_data
def load_validation_report():
    return pd.read_csv(PROCESSED / "validation_report.csv")


@st.cache_data
def load_subsection_tables():
    single = pd.read_csv(PROCESSED / "single_age_final.csv")
    grouped = pd.read_csv(PROCESSED / "grouped_age_final.csv")
    for df in (single, grouped):
        df["Death Rate"] = df["Deaths"] / df["Population"] * 100000
    return single, grouped


def lookup_combination(state, sex_code, cause):
    """Return the validation_report row (table 3) for one (state, sex, cause) combination."""
    report = load_validation_report()
    row = report[
        (report["State"] == state)
        & (report["Sex Code"] == sex_code)
        & (report["Cause List"] == cause)
    ]
    return row.iloc[0]


def subsection_data(state, sex_code, cause, resolution):
    """Model-ready rows for one combination, from whichever table its resolution points to."""
    single, grouped = load_subsection_tables()
    df = single if resolution == "single" else grouped
    df = df[
        (df["State"] == state)
        & (df["Sex Code"] == sex_code)
        & (df["Cause List"] == cause)
    ].copy()
    if resolution == "grouped":
        df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_value, ordered=True)
    return df
