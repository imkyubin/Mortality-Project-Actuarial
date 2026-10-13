"""Page 2: state and cause comparison, single-age 1999-2024 data, 7 grouped age bins.

Design: dashboard_strategy.primary.subsection_view_design in config.yaml.
Tabs: Compare locations, Compare causes (United States), Data & methodology.

- Menus come from validation_report.csv (passed combinations only), never a hard-coded list.
- Every comparison uses the 7 harmonized bins so series share one age axis.
- Headline rates are age-standardized (direct method, US 2000 weights) over only the bins
  every compared series kept; crude rates mislead when age mixes differ.
- CDC WONDER display rule: rates, ratios, improvement and variance explained only.
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    CAUSE_COLORS, LOCATION_COLORS, PALETTE, color_scale, heatmap, improvement_bars, window_lines,
)
from src.dashboard.data import (
    AGE_LABELS, BENCHMARK, CAUSE_LABELS, CREDIBILITY_STANDARD, LOCATIONS, SEX_LABELS,
    STANDARD_YEAR, age_range_label, age_standardize, common_bands, comparison_rates,
    improvement_table, load_validation, lookup_combination, passed_combinations, pooled_rates,
)

REPO_URL = "https://github.com/imkyubin/Mortality-Project-Actuarial"
YEARS = list(range(1999, 2025))


def render():
    st.title("State & Cause Comparison, 1999-2024")

    # Controls shared by every tab, matching page 1.
    left, right = st.columns(2)
    start = left.selectbox("From", YEARS, index=0)
    end = right.selectbox("To", YEARS, index=len(YEARS) - 1)
    sex = st.radio("Sex", list(SEX_LABELS), format_func=SEX_LABELS.get, horizontal=True)
    if start >= end:
        st.warning("'From' must be earlier than 'To'.")
        st.stop()

    passed = passed_combinations(sex)

    tab_loc, tab_cause, tab_method = st.tabs(
        ["Compare locations", "Compare causes (US)", "Data & methodology"])
    with tab_loc:
        _compare_locations(passed, sex, start, end)
    with tab_cause:
        _compare_causes(passed, sex, start, end)
    with tab_method:
        _methodology(sex)


# ======================================================================== Tab 1

def _compare_locations(passed, sex, start, end):
    # Causes that passed in every location (today: All causes, Heart disease, Cancer).
    n_locations = passed.groupby("Cause List")["State"].nunique()
    causes = [c for c in CAUSE_LABELS if n_locations.get(c, 0) == len(LOCATIONS)]
    if not causes:
        # Fallback (not reached with today's data): say why instead of drawing a broken chart.
        for state in LOCATIONS:
            combo = lookup_combination(state, sex, "All")
            if combo["Resolution"] == "dropped":
                st.warning(f"{state}, All causes, {SEX_LABELS[sex]}: {combo['Drop Reason']}.",
                           icon=":material/block:")
        return
    cause = st.selectbox("Cause of death", causes, format_func=CAUSE_LABELS.get)
    st.caption("State-level data is only credible enough for All causes, Heart disease and Cancer; "
               "the other causes are compared nationally in the next tab.")

    # Click states on/off; the US stays in as the benchmark every comparison is measured against.
    states = [s for s in LOCATIONS if s != BENCHMARK]
    picked = st.pills("States", states, selection_mode="multi", default=states)
    if not picked:
        st.info("Click at least one state to compare against the United States.")
        return
    shown = [BENCHMARK] + [s for s in states if s in picked]

    rates = comparison_rates(sex, causes=[cause], states=shown)
    bands = common_bands(rates, "State")
    std = age_standardize(rates, "State", bands)
    span = age_range_label(bands)

    # (a) metric row
    to_year = std[std["Year Code"] == end].set_index("State")["Std Rate"]
    std_improve = improvement_table(std, ["State"], "Std Rate", start, end).set_index("State")["Improvement"]
    us_rate, us_imp = to_year[BENCHMARK], std_improve[BENCHMARK]
    state_rates, state_imp = to_year.drop(BENCHMARK), std_improve.drop(BENCHMARK)
    hi, lo, fast = state_rates.idxmax(), state_rates.idxmin(), state_imp.idxmax()

    st.subheader(f"{CAUSE_LABELS[cause]}, {SEX_LABELS[sex]}: age-standardized, {span}")
    m1, m2, m3 = st.columns(3)
    if len(picked) == 1:
        # One state: highest/lowest/fastest would all name it, so show it against the US instead.
        m1.metric(f"{hi} rate in {end}", f"{to_year[hi]:,.0f}",
                  f"{to_year[hi] / us_rate - 1:+.0%} vs US", delta_color="inverse",
                  help="Age-standardized deaths per 100,000.")
        m2.metric(f"US rate in {end}", f"{us_rate:,.0f}", help="Age-standardized deaths per 100,000.")
        m3.metric(f"{hi} improvement, {start}-{end}", f"{state_imp[hi]:.2%}/yr",
                  f"{(state_imp[hi] - us_imp) * 100:+.2f} pts vs US",
                  help="Average annual fall in the age-standardized death rate.")
    else:
        m1.metric(f"Highest rate in {end}", f"{hi}: {to_year[hi]:,.0f}",
                  f"{to_year[hi] / us_rate - 1:+.0%} vs US", delta_color="inverse",
                  help="Age-standardized deaths per 100,000.")
        m2.metric(f"Lowest rate in {end}", f"{lo}: {to_year[lo]:,.0f}",
                  f"{to_year[lo] / us_rate - 1:+.0%} vs US", delta_color="inverse",
                  help="Age-standardized deaths per 100,000.")
        m3.metric(f"Fastest improving, {start}-{end}", f"{fast}: {std_improve[fast]:.2%}/yr",
                  f"{(std_improve[fast] - us_imp) * 100:+.2f} pts vs US",
                  help="Average annual fall in the age-standardized death rate.")

    # (b) headline chart: standardized rate over time per location
    st.markdown("**Age-standardized death rate per 100,000**")
    st.altair_chart(
        window_lines(std, "Std Rate", "State", color_scale(LOCATION_COLORS), start, end,
                     "Deaths per 100,000 (age-standardized)", height=320),
        width="stretch",
    )
    st.caption(f"Direct standardization to the US {STANDARD_YEAR} population (both sexes), over "
               f"{span}: the age groups every shown location kept for this cause and sex, so the "
               "range can widen when you hide a state. Without it, Florida's older population "
               "would make its crude rate look worse than its risk. The US line is the benchmark.")

    # (c) relative mortality heatmap: state rate / US rate, pooled over From-To
    pooled = pooled_rates(sex, cause, start, end)
    pooled = pooled[pooled["State"].isin(shown)]
    us = pooled[pooled["State"] == BENCHMARK].set_index("Age Group")["Rate"]
    rel = pooled[pooled["State"] != BENCHMARK].copy()
    rel["Relative to US"] = rel["Rate"] / rel["Age Group"].map(us).astype(float)
    rel["Age band"] = rel["Age Group"].map(AGE_LABELS).astype(str)
    st.markdown(f"**Death rate relative to the US, by age group ({start}-{end} pooled)**")
    st.altair_chart(
        heatmap(rel[["State", "Age band", "Relative to US"]], "Age band", "State", "Relative to US",
                "State / US", ".2f", "redblue", 1.0, reverse=True),
        width="stretch",
    )
    st.caption("1.00 = same as the US. Above 1 (red) = higher mortality than the national table, "
               "the same reading as an actual-to-expected ratio. "
               + _truncation_note(passed[passed["State"].isin(shown)], cause))

    # (d) improvement bars by age group, grouped by location
    rates_band = rates.assign(**{"Age band": rates["Age Group"].map(AGE_LABELS).astype(str)})
    imp = improvement_table(rates_band, ["State", "Age band"], "Rate", start, end)
    st.markdown(f"**Average annual mortality improvement by age group, {start}-{end}**")
    st.altair_chart(
        improvement_bars(imp.rename(columns={"State": "Location"}),
                         color_scale(LOCATION_COLORS), series="Location"),
        width="stretch",
    )

    # (e) takeaway
    st.info(_location_takeaway(std_improve, imp, cause, start, end), icon=":material/lightbulb:")


def _truncation_note(passed, cause):
    rows = passed[(passed["Cause List"] == cause) & (passed["Min Age Kept"] > passed["Age Floor"])]
    if rows.empty:
        return ""
    parts = [f"{r.State} (from {int(r['Min Age Kept'])})" for _, r in rows.iterrows()]
    return ("Blank cells are age groups dropped for too few deaths in at least one year: "
            + ", ".join(parts) + ".")


def _location_takeaway(std_improve, imp, cause, start, end):
    states = std_improve.drop(BENCHMARK)
    slow = states.idxmin()
    us_bands = imp[imp["State"] == BENCHMARK].set_index("Age band")["Improvement"]
    slow_bands = imp[imp["State"] == slow].set_index("Age band")["Improvement"]
    gap = (slow_bands - us_bands).dropna()
    us_imp = std_improve[BENCHMARK]
    versus = ("in line with the US" if abs(states[slow] - us_imp) < 0.0005
              else f"vs {us_imp:.2%} for the US")
    verb = "improved" if len(states) == 1 else "improved slowest"
    sentence = (f"From {start} to {end}, {slow} {verb} for {CAUSE_LABELS[cause].lower()} "
                f"({states[slow]:.2%} a year, age-standardized, {versus}).")
    if not gap.empty:
        band = gap.idxmin()
        sentence += (f" The gap was widest at ages {band}, where its death rate changed "
                     f"{slow_bands[band]:.2%} a year against {us_bands[band]:.2%} nationally.")
    return sentence


# ======================================================================== Tab 2

def _compare_causes(passed, sex, start, end):
    us = passed[passed["State"] == BENCHMARK]
    available = [c for c in CAUSE_LABELS if c in set(us["Cause List"]) and c != "All"]

    # Click causes on/off; All causes stays in as the reference line.
    picked = st.pills("Causes", available, selection_mode="multi", default=available,
                      format_func=CAUSE_LABELS.get)
    if not picked:
        st.info("Click at least one cause to compare.")
        return
    specific = [c for c in available if c in picked]
    causes = ["All"] + specific

    rates = comparison_rates(sex, causes=causes, states=[BENCHMARK])
    rates = rates.assign(Cause=rates["Cause List"].map(CAUSE_LABELS))
    bands = common_bands(rates, "Cause")
    std = age_standardize(rates, "Cause", bands)
    span = age_range_label(bands)
    std_improve = improvement_table(std, ["Cause"], "Std Rate", start, end).set_index("Cause")["Improvement"]

    # (a) metric row - specific causes only (All causes is the benchmark, not a competitor)
    specific_labels = [CAUSE_LABELS[c] for c in specific]
    imp_specific = std_improve[std_improve.index.isin(specific_labels)]
    to_year = std[(std["Year Code"] == end) & std["Cause"].isin(specific_labels)].set_index("Cause")["Std Rate"]
    fast, slow, lead = imp_specific.idxmax(), imp_specific.idxmin(), to_year.idxmax()

    st.subheader(f"United States, {SEX_LABELS[sex]}: age-standardized, {span}")
    all_imp = std_improve["All causes"]
    m1, m2, m3 = st.columns(3)
    if len(specific) == 1:
        # One cause: fastest/slowest would both name it, so show it against all causes instead.
        m1.metric(f"{lead} rate in {end}", f"{to_year[lead]:,.0f}", help="Age-standardized deaths per 100,000.")
        m2.metric(f"{lead} improvement, {start}-{end}", f"{imp_specific[lead]:.2%}/yr",
                  f"{(imp_specific[lead] - all_imp) * 100:+.2f} pts vs all causes")
        m3.metric(f"All causes improvement, {start}-{end}", f"{all_imp:.2%}/yr")
    else:
        m1.metric(f"Fastest improving, {start}-{end}", fast, f"{imp_specific[fast]:.2%} a year")
        m2.metric("Slowest improving" if imp_specific[slow] > 0 else "Worsening", slow,
                  f"{imp_specific[slow]:.2%} a year")
        m3.metric(f"Highest rate in {end}", lead, f"{to_year[lead]:,.0f} per 100,000",
                  delta_color="off", help="Among the causes shown.")

    # (b) standardized rate per cause, log scale
    st.markdown("**Age-standardized death rate per 100,000, by cause (log scale)**")
    st.altair_chart(
        window_lines(std, "Std Rate", "Cause", color_scale(CAUSE_COLORS), start, end,
                     "Deaths per 100,000 (log scale)", log_y=True, height=360),
        width="stretch",
    )
    st.caption(f"Standardized over {span}, the age groups every shown cause kept (most start at "
               "55, so hiding them can widen the range), using US 2000 weights. Log scale because "
               "causes differ by 10x or more. All causes is the reference line.")

    # (c) cause x age band improvement heatmap
    rates_band = rates.assign(**{"Age band": rates["Age Group"].map(AGE_LABELS).astype(str)})
    imp = improvement_table(rates_band, ["Cause", "Age band"], "Rate", start, end)
    st.markdown(f"**Average annual improvement by cause and age group, {start}-{end}**")
    st.altair_chart(
        heatmap(imp, "Age band", "Cause", "Improvement", "Improvement / yr", ".1%",
                "redyellowgreen", 0.0, y_sort=[CAUSE_LABELS[c] for c in causes]),
        width="stretch",
    )
    st.caption("Green = death rates falling, red = rising. Each cause shows only its own credible "
               "age groups, so younger cells are blank for late-onset diseases.")

    # (d) takeaway
    def change(rate):
        return "improved" if rate > 0 else "got worse"

    if len(specific) == 1:
        lead_in = (f"From {start} to {end}, {lead.lower()} {change(imp_specific[lead])} "
                   f"({imp_specific[lead]:.2%} a year, {span})")
    else:
        lead_in = (f"From {start} to {end}, {fast.lower()} improved fastest ({imp_specific[fast]:.2%} "
                   f"a year, {span}) and {slow.lower()} "
                   + ("improved slowest" if imp_specific[slow] > 0 else "got worse")
                   + f" ({imp_specific[slow]:.2%} a year)")
    st.info(f"{lead_in}; all causes combined {change(all_imp)} {all_imp:.2%} a year over the same ages.",
            icon=":material/lightbulb:")


# ======================================================================== Tab 3

RESOLUTION_COLORS = {"single": PALETTE[2], "grouped": PALETTE[3], "dropped": "#c9c7c0"}


def _methodology(sex):
    report = load_validation()
    cov = report[report["Sex Code"] == sex].copy()
    cov["Cause"] = cov["Cause List"].map(CAUSE_LABELS)
    cov["Ages kept"] = cov.apply(
        lambda r: "-" if r["Resolution"] == "dropped"
        else f"{int(r['Min Age Kept'])}-{int(r['Max Age Kept'])}", axis=1)
    cov["Variance"] = cov.apply(
        lambda r: "-" if pd.isna(r["Variance Explained"])
        else f"{r['Variance Explained']:.0%} (benchmark {r['Benchmark']:.0%})", axis=1)
    cov["Note"] = cov["Drop Reason"].fillna("passed")
    cov = cov[["State", "Cause", "Resolution", "Ages kept", "Variance", "Note"]]

    st.subheader(f"Coverage: which combinations are shown ({SEX_LABELS[sex]})")
    enc = dict(x=alt.X("Cause:N", sort=list(CAUSE_LABELS.values()), title=None,
                       axis=alt.Axis(labelAngle=-30)),
               y=alt.Y("State:N", sort=LOCATIONS, title=None))
    rect = alt.Chart(cov).mark_rect(stroke="white", strokeWidth=2).encode(
        **enc,
        color=alt.Color("Resolution:N", scale=color_scale(RESOLUTION_COLORS),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["State", "Cause", "Resolution", "Ages kept",
                 alt.Tooltip("Variance:N", title="1st-factor variance explained"), "Note"],
    )
    text = alt.Chart(cov).mark_text(fontSize=11, color="black").encode(**enc, text="Ages kept:N")
    st.altair_chart(alt.layer(rect, text).properties(height=260), width="stretch")
    st.caption("Cells show the ages kept. Hover for the reason a combination was dropped and for "
               "the Lee-Carter variance check. Labels only; no death counts are published.")

    st.subheader("How credibility is decided")
    st.markdown(f"""
- **Full credibility standard:** an age cell needs at least **{CREDIBILITY_STANDARD:,} deaths in
  every year** (the classical limited-fluctuation standard). Pass/fail is strict, with no partial credit.
- **Single-year first:** if every single age from the cause's starting age passes, the combination
  keeps single-year detail; otherwise the whole combination switches to 7 grouped age bands.
- **Truncate from the young end:** if grouped bands still fail, the youngest failing band and
  everything younger are dropped; if even 75-84 fails, the combination is dropped.
- **Structural minimum:** at least 3 grouped bands must survive.
- **Model check:** a single-factor Lee-Carter fit must explain at least the national benchmark's
  share of variance minus 0.05, so the series can later be forecast.
- **Years are never pooled** to rescue a combination: the year-by-year trend is what is being measured.
""")

    st.subheader("Limitations")
    st.markdown("""
- **COVID-19, 2020-2021:** a mortality shock, not a trend. Windows ending in 2020-2022 overstate
  worsening; compare with a pre-2020 To year.
- **CDC suppression:** counts under 10 are hidden by CDC WONDER and treated as failing cells.
- **Period, not cohort:** rates describe each calendar year, not the experience of people born together.
- **Influenza & pneumonia:** its 2020-2021 dip can remove an age group for every year, since a cell
  must pass in all years.
- **Alzheimer's disease:** not yet shown; its starting age (70) does not align with the 10-year bands.
- **Age standardization** covers only the age groups every compared series kept, stated under each chart.
""")
    st.markdown(f"Code and full audit trail: [{REPO_URL.split('github.com/')[1]}]({REPO_URL})")
