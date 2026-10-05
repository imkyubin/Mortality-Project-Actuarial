"""Streamlit entry point.

Run from the repo root: python -m streamlit run src/step3/dashboard/streamlit_app.py

"""
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from src.step3.dashboard.data import load_summary, load_historic
from src.step2.load_filter import age_value

st.set_page_config(page_title="US Mortality Explorer", layout="wide")

st.title("US Mortality Explorer")

metric_df = load_summary()
years = sorted(metric_df["Year Code"].unique())

left, right = st.columns(2)

start = left.selectbox("Start Year", years, index=0)
end = right.selectbox("End Year", years, index=len(years) - 1)

SEX_NAMES = {"T": "Total", "F": "Female", "M": "Male"}
sex = st.radio("Options", ["T", "M", "F"], format_func=SEX_NAMES.get, horizontal=True)

if start >= end:
    st.warning("'Start Year' must be earlier than 'End Year'")
    st.stop()

def value(year, sex_code, col):
    return metric_df.loc[(metric_df["Year Code"] == year) & (metric_df["Sex Code"] == sex_code), col].item()
    
prob_final, prob_start = value(end, sex, "Prob Reach 85"), value(start, sex, "Prob Reach 85")
ex_final, ex_start = value(end, sex, "ex"), value(start, sex, "ex")
lead_final = value(end, "F", "Prob Reach 85")-value(end, "M", "Prob Reach 85")
lead_start = value(start, "F", "Prob Reach 85")-value(start, "M", "Prob Reach 85")

st.subheader(f"Key Metrics for {SEX_NAMES[sex]} in Year {end}")

left1, middle1, right1 = st.columns(3)

left1.metric(
    "Probability of a 20-year-old Reaching Age 85", 
    f"{prob_final:.1%}", 
    delta=f"{(prob_final-prob_start)*100:+.1f} pts since {start}", 
    help="Out of 100 people alive at age 20, how many would reach their 85th birthday if that year's death rates stayed the same throughout their lives."
)

middle1.metric(
    "Expected Years Lived 20-85", 
    f"{ex_final:.1f} of 65", 
    delta=f"{(ex_final-ex_start):+.1f} years since {start}", 
    help="Average number of years a 20-year-old would live between age 20 and 85 under that year's death rates. The maximum is 65, reached only if nobody died before 85. Covers ages 20-84 only, so it is not life expectancy at birth."
)

right1.metric(
    "Women's Lead in Reaching 85", 
    f"{lead_final*100:.1f} pts", 
    delta=f"{(lead_final-lead_start)*100:+.1f} pts since {start}", 
    help="Women's chance of reaching 85 minus men's, in percentage points. Always compares women with men, whatever sex is selected above. A falling number means the gap between the sexes is narrowing."
)

st.caption("Covers ages 20-84 only, so these are not life expectancy at birth. "
           "pts = percentage points (the plain difference between two percentages).")

chart_df = metric_df.copy()
chart_df["Sex"] = chart_df["Sex Code"].map(SEX_NAMES)
SEX_COLORS = {"Total": "#978722", "Female": "#b83030", "Male": "#357ed8"}
band_df = pd.DataFrame({"start": [start], "end": [end]})
band_lines = chart_df[(chart_df["Year Code"]>=start) & (chart_df["Year Code"]<=end)]
band_highlights = chart_df[(chart_df["Year Code"]>=start) & (chart_df["Year Code"]<=end) & (chart_df["Sex Code"]==sex)]

st.subheader("Probability of a 20-year-old Reaching Age 85 over Time")

def lines(df, opac):
    chart = (
        alt.Chart(df)
        .mark_line(opacity=opac)
        .encode(
            x=alt.X("Year Code:Q", title="Year", axis=alt.Axis(format="d"),
                    scale=alt.Scale(domain=[years[0], years[-1]], padding=10)),
            y=alt.Y("Prob Reach 85:Q", title="Probability", axis=alt.Axis(format=".1%")),
            color=alt.Color(
                "Sex:N",
                scale=alt.Scale(
                    domain=list(SEX_COLORS.keys()),
                    range=list(SEX_COLORS.values())
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Sex:N", title="Sex"),
                alt.Tooltip("Year Code:N", title="Year"),
                alt.Tooltip("Prob Reach 85:Q", title="Probability", format=".1%"),
            ],
        )   
    )

    return chart

def band(df):
    band = (
        alt.Chart(df)
        .mark_rect(opacity=0.075, color="#63AE1D")
        .encode(
            x=alt.X("start:Q", title="Year", scale=alt.Scale(domain=[years[0], years[-1]], padding=10)),
            x2="end:Q",
        )
    )
    return band

thick_lines = (
    alt.Chart(band_highlights)
    .mark_line(point=alt.OverlayMarkDef(size=100, opacity=0.5), strokeWidth=4)
    .encode(
        x=alt.X("Year Code:Q", title="Year", axis=alt.Axis(format="d"),
                scale=alt.Scale(domain=[years[0], years[-1]], padding=10)),
        y=alt.Y("Prob Reach 85:Q", title="Probability", axis=alt.Axis(format=".1%")),
        color=alt.Color(
            "Sex:N",
            scale=alt.Scale(
                domain=list(SEX_COLORS.keys()),
                range=list(SEX_COLORS.values())
            ),
        ),
    )   
)

prob_chart = lines(chart_df, 0.3)
highlight_chart = lines(band_lines, 1)
year_band = band(band_df)

st.altair_chart(year_band + thick_lines + highlight_chart + prob_chart, width='stretch')
st.caption("Period measure: each point uses only that year's death rates at every age. "
           "It is not a forecast for people who were actually 20 in that year.")

historic_df = load_historic()
drate_df = historic_df.loc[historic_df["Sex Code"]==sex, ["Year Code", "Age Group", "Sex Code", "Deaths per 100k"]]
drate_band_lines = drate_df[(drate_df["Year Code"]>=start) & (drate_df["Year Code"]<=end)]
AGE_COLORS = dict(zip(age_value, ['#E06666', '#2A7B7B', '#D4A373', '#8E7CC3', '#81A185', '#5B7095', '#A64D79']))

st.subheader(f"Death Rate per 100,000 by Age Group: {SEX_NAMES[sex]}")

def lines2(df, opac):
    chart1 = (
        alt.Chart(df)
        .mark_line(opacity=opac)
        .encode(
            x=alt.X("Year Code:Q", title="Year", axis=alt.Axis(format="d"),
                    scale=alt.Scale(domain=[years[0], years[-1]], padding=10)),
            y=alt.Y("Deaths per 100k:Q", title="Deaths per 100,000 (log scale)", scale=alt.Scale(type="log"), axis=alt.Axis(format=",")),
            color=alt.Color(
                "Age Group:N",
                scale=alt.Scale(
                    domain=list(AGE_COLORS.keys()),
                    range=list(AGE_COLORS.values())
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Year Code:N", title="Year"),
                alt.Tooltip("Age Group:N", title="Age Group"),
                alt.Tooltip("Deaths per 100k:Q", title="Deaths per 100,000 Scaled", format=",.1f"),
            ],
        )   
    )
    return chart1

year_chart = lines2(drate_band_lines, 1)
drate_chart = lines2(drate_df, 0.3)

st.altair_chart(year_band + year_chart + drate_chart, width='stretch')

def improvement(drate_df):
    drate1_df = drate_df[drate_df["Year Code"].isin([start, end])]
    pivot_df = drate1_df.pivot(index='Age Group', columns='Year Code', values='Deaths per 100k')
    pivot_df = pivot_df.reset_index()
    pivot_df['Improvement'] = 1 - (pivot_df[end] / pivot_df[start]) ** (1 / (end - start))
    return pivot_df

improvement_df = improvement(drate_df)
zero_df = pd.DataFrame({"zero": [0]})

st.subheader(f"Average Annual Mortality Improvement by Age Group: {SEX_NAMES[sex]}, {start}-{end}")

bar = (alt.Chart(improvement_df).mark_bar()
       .encode(
            x=alt.X("Improvement:Q", title="Average Annual Mortality Improvement", axis=alt.Axis(format=".1%"),),
            y=alt.Y("Age Group:N", sort=age_value, title=None),
            color=alt.Color(
                "Age Group:N",
                scale=alt.Scale(
                    domain=list(AGE_COLORS.keys()),
                    range=list(AGE_COLORS.values())
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Age Group:N", title="Age Group"),
                alt.Tooltip("Improvement:Q", title="Avg Improvement", format=".1%"),
            ],
       )
)

zero = (alt.Chart(zero_df).mark_rule(color="gray")
       .encode(
            x=alt.X("zero:Q", title=None)
        )
)

st.altair_chart(bar + zero, width='stretch')

st.caption("Source: Centers for Disease Control and Prevention, National Center for Health Statistics. Compressed Mortality File 1968-1998 and Underlying Cause of Death 1999-2024, CDC WONDER Online Database.")