"""TEMPLATE - View 2: state x sex x cause subsection comparison, single-age 1999-2024 data only.

Every filter combination goes through the precomputed credibility lookup
(validation_report.csv) first: its resolution decides which pre-aggregated table
and age axis are shown, and a dropped combination renders as an explicit message
in place of a chart (api_layer.credibility_gate).

CDC WONDER display rule: only rates from cells that passed credibility are shown.
No raw death or population counts appear anywhere (no tables, no count tooltips);
the audit tables stay offline as project deliverables, not dashboard content.
"""
import streamlit as st

from templates.dashboard.charts import (
    AGE_BAND_COLORS, PALETTE, PERIOD_COLORS, color_scale, line_chart,
)
from templates.dashboard.data import (
    CAUSE_LABELS, CREDIBILITY_STANDARD, LOCATIONS, SEX_LABELS,
    lookup_combination, subsection_data,
)


def render():
    st.header("State and cause comparison, 1999-2024")
    st.caption(
        "Pick a location, cause and sex. Only ages with enough data to be reliable in every "
        "year are shown, so the age detail differs between combinations."
    )

    c1, c2, c3 = st.columns(3)
    state = c1.selectbox("Location", LOCATIONS)
    cause = c2.selectbox("Cause of death", list(CAUSE_LABELS), format_func=CAUSE_LABELS.get)
    sex_code = c3.selectbox("Sex", list(SEX_LABELS), index=2, format_func=SEX_LABELS.get)

    combo = lookup_combination(state, sex_code, cause)
    resolution = combo["Resolution"]

    if resolution == "dropped":
        st.warning(
            f"**Not enough data to show this combination reliably.** "
            f"{state}, {CAUSE_LABELS[cause]}, {SEX_LABELS[sex_code]}: {combo['Drop Reason']}.",
            icon=":material/block:",
        )
        return

    _credibility_summary(combo, cause)

    df = subsection_data(state, sex_code, cause, resolution)
    years = (int(df["Year Code"].min()), int(df["Year Code"].max()))
    start, end = st.slider("Years", *years, value=years)
    df = df[df["Year Code"].between(start, end)]

    if resolution == "single":
        _single_year_charts(df, start, end)
    else:
        _grouped_charts(df, start, end)


def _credibility_summary(combo, cause):
    min_age, max_age = int(combo["Min Age Kept"]), int(combo["Max Age Kept"])
    floor = int(combo["Age Floor"])
    label = "single-year ages" if combo["Resolution"] == "single" else (
        f"{int(combo['Groups Kept'])} grouped age bands")

    m1, m2, m3 = st.columns(3)
    m1.metric("Age detail", label)
    m2.metric("Ages shown", f"{min_age}-{max_age}")
    m3.metric(
        "Variance explained (1st factor)", f"{combo['Variance Explained']:.1%}",
        f"{combo['Variance Explained'] - combo['Benchmark']:+.1%} vs national benchmark",
        delta_color="off",
    )
    if min_age > floor:
        st.info(
            f"Ages {floor}-{min_age - 1} are not shown: at least one year fell below the "
            f"{CREDIBILITY_STANDARD:,}-death credibility standard at those ages. The starting "
            f"age for {CAUSE_LABELS[cause]} is {floor}.",
            icon=":material/info:",
        )


def _single_year_charts(df, start, end):
    ages = sorted(df["Single-Year Ages Code"].unique())

    st.subheader("Death rate over time at one age")
    age = st.select_slider("Age", ages, value=ages[len(ages) // 2])
    trend = df[df["Single-Year Ages Code"] == age].assign(Series=f"Age {age}")
    st.altair_chart(
        line_chart(trend, "Year Code", "Death Rate", "Series",
                   color_scale({f"Age {age}": PALETTE[0]}),
                   "Year", "Deaths per 100,000"),
        width="stretch",
    )

    st.subheader(f"Death rate by age, {start} vs {end}")
    st.altair_chart(
        line_chart(_period_frame(df, start, end), "Single-Year Ages Code", "Death Rate", "Period",
                   color_scale(PERIOD_COLORS), "Age", "Deaths per 100,000 (log scale)", log_y=True),
        width="stretch",
    )


def _grouped_charts(df, start, end):
    df = df.assign(**{"Age band": df["Age Group"].astype(str).str.replace(" years", "")})

    st.subheader("Death rate over time by age band")
    st.altair_chart(
        line_chart(df, "Year Code", "Death Rate", "Age band",
                   color_scale(AGE_BAND_COLORS, keep=set(df["Age band"])),
                   "Year", "Deaths per 100,000 (log scale)", log_y=True),
        width="stretch",
    )

    st.subheader(f"Death rate by age band, {start} vs {end}")
    st.altair_chart(
        line_chart(_period_frame(df, start, end), "Age band", "Death Rate", "Period",
                   color_scale(PERIOD_COLORS), "Age band", "Deaths per 100,000 (log scale)",
                   log_y=True, x_type="O"),
        width="stretch",
    )


def _period_frame(df, start, end):
    period = df[df["Year Code"].isin([start, end])]
    return period.assign(Period=period["Year Code"].map({start: "Start year", end: "End year"}))
