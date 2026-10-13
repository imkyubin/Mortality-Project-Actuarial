"""Cached data loaders, the credibility lookup, and the rate math the three pages share.

No layout code here, so the API and reporting layers can reuse the same lookup
(api_layer.credibility_gate in config.yaml).

Data source: data/dashboard/ only. Those tables are derived (rates, life-table values, ratios,
shares) and committed to the repo, so the app runs from a fresh clone or a hosted deployment
without any CDC WONDER counts. scripts/export_dashboard_data.py and
scripts/national/forecast_precompute.py write them from the local pipeline outputs.

CDC WONDER display rule: no Deaths or Population values reach a page. Every function that feeds
a chart returns derived columns only, because Altair embeds the whole DataFrame it is given into
the page.

st.cache_data does not notice a regenerated CSV: clear the cache (C in the app, or
`streamlit cache clear`) after re-exporting data/dashboard/.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.actuarial.load_filter import age_value

DASHBOARD_DATA = Path(__file__).resolve().parents[2] / "data" / "dashboard"
MODEL = DASHBOARD_DATA / "model"

SEX_LABELS = {"T": "Total", "F": "Female", "M": "Male"}

CAUSE_LABELS = {
    "All": "All causes",
    "#Diseases of heart (I00-I09,I11,I13,I20-I51)": "Heart disease",
    "#Malignant neoplasms (C00-C97)": "Cancer",
    "#Cerebrovascular diseases (I60-I69)": "Cerebrovascular disease",
    "#Chronic lower respiratory diseases (J40-J47)": "COPD",
    "#Diabetes mellitus (E10-E14)": "Diabetes mellitus",
    "#Alzheimer disease (G30)": "Alzheimer's disease",
    "#Influenza and pneumonia (J09-J18)": "Influenza & pneumonia",
    "#Chronic liver disease and cirrhosis (K70,K73-K74)": "Chronic liver disease",
}

LOCATIONS = ["United States", "California", "Florida", "New York", "Texas"]
BENCHMARK = "United States"

AGE_LABELS = {group: group.replace(" years", "") for group in age_value}

CREDIBILITY_STANDARD = 1082
STANDARD_YEAR = 2000


# ---------------------------------------------------------------- page 1 (harmonized 1968-2024)

@st.cache_data
def load_historic():
    """National all-cause life table, 20-84 grouped ages: death rate per 100,000, lx, px, ex."""
    df = pd.read_csv(DASHBOARD_DATA / "historic.csv")
    df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_value, ordered=True)
    return df


@st.cache_data
def load_summary():
    """One row per (year, sex): expected years lived 20-85 and the chance a 20-year-old reaches 85."""
    df = load_historic()

    young = df[df["Age Group"] == "20-24 years"]
    summary = young[["Year Code", "Sex Code", "ex"]]

    old = df[df["Age Group"] == "75-84 years"].copy()
    old["Prob Reach 85"] = old["lx"] * old["px"] / 100000

    return summary.merge(old[["Year Code", "Sex Code", "Prob Reach 85"]], on=["Year Code", "Sex Code"])


# ---------------------------------------------------------------- page 2 (single-age 1999-2024)

@st.cache_data
def load_validation():
    """Table 3 of the sublevel validation audit trail: one row per (sex, state, cause)."""
    return pd.read_csv(DASHBOARD_DATA / "validation_report.csv")


def lookup_combination(state, sex_code, cause):
    """The validation_report row for one combination (used for the Drop Reason fallback)."""
    report = load_validation()
    row = report[(report["State"] == state) & (report["Sex Code"] == sex_code)
                 & (report["Cause List"] == cause)]
    return row.iloc[0]


def passed_combinations(sex_code):
    """Combinations that survived validation for one sex - the source of every page 2 menu."""
    report = load_validation()
    return report[(report["Sex Code"] == sex_code) & (report["Resolution"] != "dropped")]


@st.cache_data
def load_rates():
    """Every passed combination on the shared 7-group age axis: rate per 100,000 and exposure weight.

    Combinations that passed at single-year resolution were bucketed into the groups when the
    table was exported. Exposure Weight is each cell's population relative to its own 1999-2024
    average, enough to pool a window exactly (see pooled_rates) without the counts.
    """
    df = pd.read_csv(DASHBOARD_DATA / "rates.csv")
    df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_value, ordered=True)
    return df


@st.cache_data
def standard_weights():
    """Direct-standardization weights: US, both sexes, all causes, year 2000, share by age group.

    Population is the same whatever the cause, so All causes is used because it keeps all 7
    groups. Exported as shares, so no counts are involved.
    """
    df = pd.read_csv(DASHBOARD_DATA / "standard_weights.csv")
    df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_value, ordered=True)
    return df.set_index("Age Group")["Weight"]


def comparison_rates(sex_code, causes=None, states=None):
    """Death rate per 100,000 by year x age group for the requested combinations - derived only."""
    df = load_rates()
    df = df[df["Sex Code"] == sex_code]
    if causes is not None:
        df = df[df["Cause List"].isin(causes)]
    if states is not None:
        df = df[df["State"].isin(states)]
    return df[["Year Code", "Age Group", "State", "Cause List", "Rate"]]


def pooled_rates(sex_code, cause, start, end):
    """Rate per age group pooled over start-end (sum deaths / sum exposure) per location.

    sum(D) / sum(P) equals sum(w * rate) / sum(w) for any weights w proportional to P within a
    cell, so the exported Exposure Weight reproduces it exactly. Pooling the window is steadier
    than a single year for the relative-mortality heatmap.
    """
    df = load_rates()
    df = df[(df["Sex Code"] == sex_code) & (df["Cause List"] == cause)
            & df["Year Code"].between(start, end)]
    df = df.assign(Weighted=df["Rate"] * df["Exposure Weight"])
    sums = df.groupby(["State", "Age Group"], observed=True)[["Weighted", "Exposure Weight"]].sum()
    return (sums["Weighted"] / sums["Exposure Weight"]).rename("Rate").reset_index()


def common_bands(rates, series_col):
    """Age groups present for every series in `rates` (ordered youngest to oldest).

    Validation drops a bin in every year or none, so presence anywhere means presence
    throughout; a bin missing for one series was truncated from that series.
    """
    n_series = rates[series_col].nunique()
    counts = rates.groupby("Age Group", observed=True)[series_col].nunique()
    return [g for g in age_value if g in counts.index and counts[g] == n_series]


def age_standardize(rates, series_col, bands):
    """Direct age standardization over `bands`: sum(w * rate) / sum(w), per series and year.

    Weights are renormalized over the bands kept, so the result is a rate for that age
    range (e.g. ages 45-84), not for 20-84.
    """
    df = rates[rates["Age Group"].isin(bands)].merge(
        standard_weights().reset_index(), on="Age Group")
    df["Weighted"] = df["Rate"] * df["Weight"]
    out = df.groupby([series_col, "Year Code"], as_index=False)[["Weighted", "Weight"]].sum()
    out["Std Rate"] = out["Weighted"] / out["Weight"]
    return out[[series_col, "Year Code", "Std Rate"]]


def annual_improvement(rate_from, rate_to, start, end):
    """Average annual mortality improvement: 1 - (rate_To / rate_From) ** (1 / (To - From)).

    Positive = death rates falling. Works elementwise on Series.
    """
    return 1 - (rate_to / rate_from) ** (1 / (end - start))


def improvement_table(df, keys, value, start, end):
    """Annual improvement of `value` between start and end, per `keys` group."""
    wide = df[df["Year Code"].isin([start, end])].pivot_table(
        index=keys, columns="Year Code", values=value, observed=True)
    wide = wide.dropna(subset=[start, end])
    wide["Improvement"] = annual_improvement(wide[start], wide[end], start, end)
    return wide["Improvement"].reset_index()


def band_start(group):
    """'45-54 years' -> 45."""
    return int(group.split("-")[0])


def age_range_label(bands):
    """['45-54 years', ..., '75-84 years'] -> 'ages 45-84'."""
    return f"ages {band_start(bands[0])}-{AGE_LABELS[bands[-1]].split('-')[1]}"


# ---------------------------------------------------------------- page 3 (model outputs)

@st.cache_data
def load_model(name):
    """One precomputed model table from data/dashboard/model/ (scripts/national/forecast_precompute.py).

    Every table is derived only (parameters, ex, improvement, A/E ratios), so it can go
    straight to a chart.
    """
    return pd.read_csv(MODEL / f"{name}.csv")


def model_slice(name, sex_code, model=None, calibration=None):
    """Rows of one model table for one sex (and model / calibration window where the table has them)."""
    df = load_model(name)
    df = df[df["Sex Code"] == sex_code]
    if model is not None:
        df = df[df["Model"] == model]
    if calibration is not None:
        df = df[df["Calibration"] == calibration]
    return df
