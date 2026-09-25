"""TEMPLATE - View 1: national historic trend, harmonized 1968-2024 grouped-age data, all causes.

No cause stratification and no credibility gate needed: every national all-cause
20-84 cell is far above the 1,082-death standard.

CDC WONDER display rule: only derived rates and life-table values are shown; no raw
death or population counts (no table view).
"""
import streamlit as st

from src.step2.load_filter import age_value
from templates.dashboard.charts import AGE_BAND_COLORS, SEX_COLORS, color_scale, line_chart
from templates.dashboard.data import SEX_LABELS, load_historic_life_table

AGE_LABELS = {group: group.replace(" years", "") for group in age_value}


def render():
    st.header("National mortality trends, 1968-2024")
    st.caption(
        "United States, all causes, ages 20-84 in grouped age bands. Life expectancy here is "
        "*temporary*: expected years lived between the start of the age band and age 85, since "
        "the table stops at 84."
    )

    df = load_historic_life_table()
    df["Sex"] = df["Sex Code"].map(SEX_LABELS)
    years = (int(df["Year Code"].min()), int(df["Year Code"].max()))

    col_years, col_sex = st.columns([2, 1])
    start, end = col_years.slider("Years", *years, value=years)
    sexes = col_sex.multiselect("Sex", list(SEX_LABELS.values()), default=["Female", "Male"])
    if not sexes:
        st.info("Select at least one sex.")
        return

    view = df[df["Year Code"].between(start, end) & df["Sex"].isin(sexes)]

    # KPI row: change in temporary life expectancy at 20 and 65 over the selected window.
    st.subheader(f"Change {start} to {end}")
    kpi_cols = st.columns(len(sexes) * 2)
    i = 0
    for sex in sexes:
        for group, label in [("20-24 years", "from 20 to 85"), ("65-74 years", "from 65 to 85")]:
            cells = view[(view["Sex"] == sex) & (view["Age Group"] == group)].set_index("Year Code")["ex"]
            kpi_cols[i].metric(
                f"{sex}: years lived {label}",
                f"{cells[end]:.2f}",
                f"{cells[end] - cells[start]:+.2f} since {start}",
            )
            i += 1

    st.divider()

    st.subheader("Temporary life expectancy over time")
    group = st.selectbox("Starting age band", age_value, index=age_value.index("65-74 years"),
                         format_func=lambda g: AGE_LABELS[g])
    ex_df = view[view["Age Group"] == group]
    st.altair_chart(
        line_chart(ex_df, "Year Code", "ex", "Sex",
                   color_scale(SEX_COLORS, keep=sexes),
                   "Year", f"Years lived from {AGE_LABELS[group].split('-')[0]} to 85",
                   y_format=".2f"),
        width="stretch",
    )

    st.subheader("Death rate by age band")
    sex = st.radio("Sex", sexes, horizontal=True, key="rate_sex")
    rate_df = view[view["Sex"] == sex].copy()
    rate_df["Age band"] = rate_df["Age Group"].map(AGE_LABELS).astype(str)
    st.altair_chart(
        line_chart(rate_df, "Year Code", "Death Rate", "Age band",
                   color_scale(AGE_BAND_COLORS),
                   "Year", "Deaths per 100,000 (log scale)", log_y=True),
        width="stretch",
    )
    st.caption(
        "Cause-of-death coding moves through ICD-8, ICD-9 and ICD-10 over this period; all-cause "
        "totals are unaffected, but the 1998/1999 seam also marks the switch from grouped to "
        "single-year CDC WONDER exports (re-bucketed here)."
    )
