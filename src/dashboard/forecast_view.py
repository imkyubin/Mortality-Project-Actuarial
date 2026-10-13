"""Page 3: Lee-Carter mortality forecast, national all-cause, single ages 20-84, by sex.

Design: dashboard_strategy.primary.forecast_view_design in config.yaml.
Tabs: Forecast, Backtest, Model & limitations ("a credible forecast, then the evidence it
can be trusted").

- Nothing is fitted live: scripts/national/forecast_precompute.py writes data/dashboard/model/.
- Labeling rule: the life table stops at 84, so ex is "expected years lived from age x to 85",
  never "e65" or "life expectancy".
- CDC WONDER display rule: parameters, ex, improvement rates and A/E ratios only.
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    COVID_COLOR, NEUTRAL, PALETTE, SEX_COLORS, color_scale, fan_chart, series_lines,
)
from src.dashboard.data import SEX_LABELS, load_model, model_slice

MODEL_LABELS = {"Poisson": "Poisson Lee-Carter", "SVD": "Lee-Carter (SVD)"}
MODEL_COLORS = {"Poisson Lee-Carter": PALETTE[0], "Lee-Carter (SVD)": PALETTE[1]}
CALIBRATION_LABELS = {"1999-2024": "1999-2024 (all data)", "1999-2019": "1999-2019 (pre-COVID)"}

AGES = [20, 35, 45, 55, 65, 75]
HORIZONS = [2030, 2035, 2040, 2045, 2050]
LAST_ACTUAL = 2024
TRAIN_END = 2014
PRE_COVID_END = 2019
COVID = (2020, 2021)


def render():
    st.title("Mortality Forecast, Lee-Carter")
    st.caption("United States, all causes, single ages 20-84. Models fitted offline on CDC WONDER "
               "1999-2024 data; the page only reads the results.")

    c1, c2, c3 = st.columns(3)
    sex = c1.radio("Sex", list(SEX_LABELS), format_func=SEX_LABELS.get, horizontal=True)
    model = c2.radio("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get, horizontal=True,
                     help="Poisson Lee-Carter is the primary model; the SVD fit is shown for comparison.")
    calibration = c3.radio("Calibration window", list(CALIBRATION_LABELS),
                           format_func=CALIBRATION_LABELS.get, horizontal=True,
                           help="Which years the model learns its trend from. The Backtest tab "
                                "always trains on 1999-2014.")
    c4, c5 = st.columns(2)
    age = c4.selectbox("Starting age", AGES, index=AGES.index(65))
    horizon = c5.select_slider("Forecast to", HORIZONS, value=2035)

    tab_forecast, tab_backtest, tab_model = st.tabs(["Forecast", "Backtest", "Model & limitations"])
    with tab_forecast:
        _forecast(sex, model, calibration, age, horizon)
    with tab_backtest:
        _backtest(sex, model, age)
    with tab_model:
        _model(sex, model, calibration)


def _years_lived_label(age):
    return f"Expected years lived from {age} to 85"


def _actual_ex(sex, age):
    actual = model_slice("actual_ex", sex)
    return actual.loc[actual["Age"] == age, ["Year", "ex"]].rename(columns={"ex": "Value"})


# ======================================================================== Tab 1

def _forecast(sex, model, calibration, age, horizon):
    end_year = int(calibration.split("-")[1])
    actual = _actual_ex(sex, age)
    fan = model_slice("forecast_ex", sex, model, calibration)
    fan = fan[(fan["Age"] == age) & (fan["Year"] <= horizon)].rename(
        columns={"ex_forecast": "Forecast", "ex_lower": "Lower", "ex_upper": "Upper"})
    at_horizon = fan[fan["Year"] == horizon].iloc[0]
    last_actual = actual.loc[actual["Year"] == LAST_ACTUAL, "Value"].item()

    params = model_slice("params", sex, model, calibration)
    older = params[params["Age"] >= age]
    lc_imp, obs_imp = older["LC Improvement"].mean(), older["Observed Improvement"].mean()

    # (a) metric row
    st.subheader(f"{_years_lived_label(age)}: {SEX_LABELS[sex]}, {MODEL_LABELS[model]}")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Projected for {horizon}", f"{at_horizon['Forecast']:.2f} of {85 - age}",
              f"{at_horizon['Forecast'] - last_actual:+.2f} years vs {LAST_ACTUAL} actual",
              help=f"Average years someone aged {age} would live before 85 under the projected "
                   f"{horizon} death rates (a period measure). The maximum is {85 - age}.")
    m2.metric(f"95% interval, {horizon}", f"{at_horizon['Lower']:.2f} - {at_horizon['Upper']:.2f}",
              f"±{(at_horizon['Upper'] - at_horizon['Lower']) / 2:.2f} years", delta_color="off",
              help="From the random walk in kt only, so it understates the real uncertainty "
                   "(see Model & limitations).")
    m3.metric(f"Implied improvement, ages {age}-84", f"{lc_imp:.2%}/yr",
              f"{(lc_imp - obs_imp) * 100:+.2f} pts vs observed {calibration}", delta_color="off",
              help="Average over ages of 1 - exp(bx * drift), the yearly fall in death rates the "
                   "model projects, against the log-linear trend actually seen in the window.")

    # (b) fan chart
    st.markdown(f"**{_years_lived_label(age)}: actual and forecast**")
    sex_color = SEX_COLORS[SEX_LABELS[sex]]
    shade = [(COVID[0] - 0.5, COVID[1] + 0.5, "COVID", COVID_COLOR)]
    st.altair_chart(fan_chart(actual, fan[["Year", "Forecast", "Lower", "Upper"]], sex_color,
                              "Years lived to 85", shade=shade), width="stretch")
    overlap = fan.merge(actual, on="Year")
    outside = overlap[(overlap["Value"] < overlap["Lower"]) | (overlap["Value"] > overlap["Upper"])]
    note = (f"Gray = actual, dashed = central forecast, shaded = 95% interval. The forecast starts in "
            f"{end_year + 1} from the model's fitted {end_year} level, not the actual one, so the "
            "first year can sit off the actual line.")
    if not overlap.empty:
        note += (f" Calibrated before COVID, it also covers {len(overlap)} years that have since "
                 f"happened; {len(outside)} of them fell outside the band.")
    st.caption(note)

    # Calibration choice, computed for the selected model and sex
    summary = model_slice("summary", sex, model).set_index("Calibration")
    full, pre = summary.loc["1999-2024"], summary.loc["1999-2019"]
    st.info(
        f"**Calibration choice.** Fitting on 1999-2024 instead of 1999-2019 changes the kt drift from "
        f"{pre['drift']:.3f} to {full['drift']:.3f} a year and its volatility (sigma) from "
        f"{pre['sigma']:.2f} to {full['sigma']:.2f}. The drift is the mean yearly change in kt, which "
        "telescopes to (last kt - first kt) / (years - 1), so only the end years set the long-run "
        "trend; the 2020-2021 jumps mostly widen the band. Choosing a window is an assumption about "
        "whether COVID-era mortality is a temporary shock or part of the trend.",
        icon=":material/tune:")

    # (c) projected vs observed improvement by age
    st.markdown(f"**Mortality improvement by age: model-implied vs observed {calibration}**")
    imp = pd.concat([
        params[["Age", "LC Improvement"]].rename(columns={"LC Improvement": "Improvement"})
        .assign(Series="Lee-Carter implied"),
        params[["Age", "Observed Improvement"]].rename(columns={"Observed Improvement": "Improvement"})
        .assign(Series=f"Observed trend {calibration}"),
    ])
    scale = color_scale({"Lee-Carter implied": sex_color, f"Observed trend {calibration}": NEUTRAL})
    st.altair_chart(series_lines(imp, "Age", "Improvement", "Series", scale, "Age",
                                 "Average annual improvement", y_format=".2%", axis_format=".1%",
                                 dashed=f"Observed trend {calibration}", rule=(0.01, "1% a year")),
                    width="stretch")
    st.caption("Implied improvement is 1 - exp(bx * drift): Lee-Carter fixes the age pattern of "
               "improvement (bx) and lets only its pace (kt) move. Observed is 1 - exp(slope) of a "
               "log-linear trend in each age's death rate. Published improvement scales (e.g. SOA "
               "MP-2021) assume roughly 1% a year long-term. Where the two lines part, the model is "
               "smoothing over ages whose recent trend broke from the rest, such as the young-adult "
               "ages hit by drug overdoses.")


# ======================================================================== Tab 2

def _backtest(sex, model, age):
    actual = _actual_ex(sex, age)
    bt = model_slice("backtest_ex", sex, model)
    bt = bt[bt["Age"] == age].rename(
        columns={"ex_forecast": "Forecast", "ex_lower": "Lower", "ex_upper": "Upper"})
    err = model_slice("backtest_error", sex, model)
    err = err[err["Age"] == age].iloc[0]
    ae = model_slice("backtest_ae", sex, model)
    last_year = int(bt["Year"].max())

    pre = bt[bt["Year"] <= PRE_COVID_END]
    outside = pre[(pre["ex_actual"] < pre["Lower"]) | (pre["ex_actual"] > pre["Upper"])]
    ae_pre = ae.loc[ae["Year"] <= PRE_COVID_END, "AE Ratio"].mean()
    peak = ae.loc[ae["AE Ratio"].idxmax()]

    st.subheader(f"Out-of-sample backtest: trained 1999-{TRAIN_END}, forecast {TRAIN_END + 1}-{last_year}")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Error before COVID, {TRAIN_END + 1}-{PRE_COVID_END}", f"{err['MAPE Pre-COVID']:.2%}",
              help=f"Mean absolute % error of the forecast years lived from {age} to 85.")
    m2.metric(f"Error over full holdout, {TRAIN_END + 1}-{last_year}", f"{err['MAPE Full']:.2%}",
              help="Same measure including the COVID years.")
    m3.metric("Peak actual / expected deaths", f"{peak['AE Ratio']:.2f}", f"in {int(peak['Year'])}",
              delta_color="off",
              help="Actual deaths divided by deaths expected from the forecast death rates, all "
                   "ages 20-84. 1.00 = forecast exactly right; above 1 = more deaths than forecast.")

    st.markdown(f"**{_years_lived_label(age)}: actual vs the 1999-{TRAIN_END} forecast**")
    shade = [(1999, TRAIN_END + 0.5, "Training", NEUTRAL), (COVID[0] - 0.5, COVID[1] + 0.5, "COVID", COVID_COLOR)]
    st.altair_chart(fan_chart(actual, bt[["Year", "Forecast", "Lower", "Upper"]],
                              SEX_COLORS[SEX_LABELS[sex]], "Years lived to 85", shade=shade),
                    width="stretch")
    st.caption("The model only saw the gray-shaded training years. COVID (2020-2021) is an expected "
               "divergence, a shock the random walk cannot anticipate, not a bug in the fit.")

    st.markdown("**Actual / expected deaths per holdout year (ages 20-84)**")
    # Bars grow from the A/E = 1 line, so the length of each bar is the miss itself.
    ae = ae.assign(Base=1.0, Period=ae["Year"].map(
        lambda y: "COVID" if COVID[0] <= y <= COVID[1] else
        ("Pre-COVID" if y <= PRE_COVID_END else "Post-COVID")))
    bars = alt.Chart(ae).mark_bar().encode(
        x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("AE Ratio:Q", title="Actual / expected", scale=alt.Scale(zero=False)), y2="Base:Q",
        color=alt.Color("Period:N", scale=color_scale({"Pre-COVID": PALETTE[0], "COVID": COVID_COLOR,
                                                       "Post-COVID": NEUTRAL}),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("AE Ratio:Q", title="A/E", format=".3f")],
    )
    one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="black").encode(y="y:Q")
    st.altair_chart(alt.layer(bars, one).properties(height=260), width="stretch")

    direction = "more" if ae_pre > 1 else "fewer"
    st.info(
        f"Trained on 1999-{TRAIN_END}, the {MODEL_LABELS[model]} forecast of years lived from {age} "
        f"to 85 was off by {err['MAPE Pre-COVID']:.2%} on average in {TRAIN_END + 1}-{PRE_COVID_END}, "
        f"with {len(outside)} of {len(pre)} pre-COVID years outside its 95% band. Even before COVID, "
        f"{abs(ae_pre - 1):.1%} {direction} people died than forecast (average A/E {ae_pre:.3f}): "
        "the 1999-2014 trend did not anticipate the 2010s slowdown in improvement. Deaths then "
        f"peaked at {peak['AE Ratio']:.2f}x expected in {int(peak['Year'])}.",
        icon=":material/lightbulb:")


# ======================================================================== Tab 3

def _model(sex, model, calibration):
    summary = model_slice("summary", sex, calibration=calibration).set_index("Model")
    params = model_slice("params", sex, calibration=calibration)
    variance = summary["Variance Explained"].iloc[0]

    st.subheader(f"Poisson vs SVD Lee-Carter: {SEX_LABELS[sex]}, calibrated {calibration}")
    rows = "\n".join(
        f"| {MODEL_LABELS[m]} | {r['drift']:+.3f} | {r['sigma']:.2f} | {r['In-sample AE']:.4f} | "
        f"{r['Mean Abs Cell Deviation']:.1%} |"
        for m, r in summary.loc[list(MODEL_LABELS)].iterrows())
    st.markdown(
        "| Model | kt drift / yr | kt sigma | In-sample A/E | Mean cell deviation from A/E = 1 |\n"
        "|---|---|---|---|---|\n" + rows)
    st.caption("A negative drift means death rates are projected to fall. In-sample A/E is total "
               "actual over total expected deaths across the calibration window.")

    young = params[params["Age"].between(25, 44)].groupby("Model")["bx"].sum()
    svd_note = ""
    if summary.loc["SVD", "drift"] > 0:
        svd_note = (f" In this window the SVD drift is positive: its first factor puts "
                    f"{young['SVD']:.0%} of its age weight (bx) on ages 25-44 (Poisson: "
                    f"{young['Poisson']:.0%}), so it is tracking the young-adult overdose and COVID "
                    "shocks rather than the long-run decline at older ages, where most deaths are.")
    st.markdown(
        "**Why Poisson is primary.** The SVD fit minimizes squared error on log death rates and "
        "treats every age-year cell equally, so small, noisy young-adult cells count as much as "
        "ages 70-84 with many times the deaths. Poisson Lee-Carter maximizes the likelihood of the "
        "observed death counts given exposure, which weights each cell by how much data it holds and "
        f"reproduces total deaths exactly (A/E = 1.0000 in-sample).{svd_note}")
    other = next(c for c in CALIBRATION_LABELS if c != calibration)
    other_variance = model_slice("summary", sex, model, other)["Variance Explained"].item()
    st.markdown(
        f"**Single-factor check.** The first factor explains **{variance:.0%}** of the variance in "
        f"log death rates over {calibration}, against {other_variance:.0%} over {other}: the "
        "2020-2024 shock is not shaped like the long-run trend, so one factor captures less of it. "
        "Sub-national combinations on page 2 must reach the 1999-2024 national figure minus 0.05.")

    with st.expander("Model parameters: ax, bx, kt"):
        p = params.assign(Model=params["Model"].map(MODEL_LABELS))
        scale = color_scale(MODEL_COLORS)
        left, right = st.columns(2)
        left.markdown("**ax: average log death rate by age**")
        left.altair_chart(series_lines(p, "Age", "ax", "Model", scale, "Age", "ax", height=240),
                          width="stretch")
        right.markdown("**bx: how strongly each age follows kt**")
        right.altair_chart(series_lines(p, "Age", "bx", "Model", scale, "Age", "bx", y_format=".4f",
                                        height=240), width="stretch")
        kt = load_model("kt")
        kt = kt[(kt["Sex Code"] == sex) & (kt["Calibration"] == calibration) & (kt["Type"] == "Fitted")]
        kt = kt.assign(Model=kt["Model"].map(MODEL_LABELS))
        st.markdown("**kt: the mortality index over time**")
        st.altair_chart(series_lines(kt, "Year", "kt", "Model", scale, "Year", "kt", height=240),
                        width="stretch")
        st.caption("ln(death rate at age x, year t) = ax + bx * kt. Identified by sum(bx) = 1 and "
                   "mean(kt) = 0. kt is forecast as a random walk with drift.")

    with st.expander(f"In-sample actual / expected by age and year ({MODEL_LABELS[model]})"):
        ae = model_slice("ae_in_sample", sex, model, calibration)
        heat = alt.Chart(ae).mark_rect().encode(
            x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("Age:O", title="Age", sort="descending", axis=alt.Axis(values=list(range(20, 85, 5)))),
            color=alt.Color("AE Ratio:Q", title="A/E", scale=alt.Scale(
                scheme="redblue", domainMid=1, reverse=True), legend=alt.Legend(format=".2f")),
            tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Age:O"),
                     alt.Tooltip("AE Ratio:Q", title="A/E", format=".3f")],
        )
        st.altair_chart(heat.properties(height=480), width="stretch")
        st.caption("Red = more deaths than the fitted model, blue = fewer. Diagonal streaks would "
                   "point to a cohort effect, which Lee-Carter has no term for; horizontal bands "
                   "at 2020-2021 are the COVID shock.")

    st.subheader("Limitations")
    st.markdown("""
- **The band is too narrow.** It covers only the random-walk process error in kt, not uncertainty in
  the estimated ax, bx or drift, nor model error. This is a well-known Lee-Carter criticism; treat the
  95% interval as a lower bound on the real uncertainty.
- **One factor, fixed age pattern.** Every age improves in the fixed proportion bx, so a shock that
  hits some ages and not others (overdoses at 25-44, COVID at older ages) is smoothed away.
- **No cohort effect.** Generations born in the same years are not modeled separately.
- **Ages 20-84 only.** "Years lived from x to 85" is a temporary life expectancy, not full life
  expectancy, because the data stops at 84.
- **Period measure.** Each forecast year applies that year's projected rates at every age; it is not
  the experience of a person who is aged x in that year.
- **COVID and calibration.** Results depend on whether 2020-2024 is in the calibration window; both
  are shown so the assumption is visible rather than buried.
""")
