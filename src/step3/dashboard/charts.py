"""Altair chart builders shared by pages 2 and 3 (page 1 builds its own charts inline).

Colors are fixed per entity (a sex, age group, location or cause always keeps the same
hue whatever the filters), so changing a control never repaints a surviving series.

Interactivity rule (historic_view_design): full-history charts keep the whole period for
context but visibly mark the selection - a shaded From-To band, the window drawn at full
opacity, the rest faded.
"""
import altair as alt
import pandas as pd

from src.step2.load_filter import age_value

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#898781"
BAND_COLOR = "#63AE1D"
COVID_COLOR = "#e34948"

SEX_COLORS = {"Total": PALETTE[2], "Female": PALETTE[0], "Male": PALETTE[1]}
AGE_COLORS = dict(zip([g.replace(" years", "") for g in age_value], PALETTE))
LOCATION_COLORS = {"United States": NEUTRAL, "California": PALETTE[0], "Florida": PALETTE[1],
                   "New York": PALETTE[6], "Texas": PALETTE[3]}
CAUSE_COLORS = {"All causes": NEUTRAL, "Heart disease": PALETTE[7], "Cancer": PALETTE[0],
                "Cerebrovascular disease": PALETTE[6], "COPD": PALETTE[3],
                "Diabetes mellitus": PALETTE[2], "Influenza & pneumonia": PALETTE[4],
                "Chronic liver disease": PALETTE[1], "Alzheimer's disease": PALETTE[5]}

FADED = 0.3


def color_scale(colors, keep=None):
    """Fixed entity -> color scale; `keep` restricts the domain without repainting survivors."""
    items = [(k, v) for k, v in colors.items() if keep is None or k in keep]
    return alt.Scale(domain=[k for k, _ in items], range=[v for _, v in items])


def year_band(start, end, domain, color=BAND_COLOR, opacity=0.08, label=None):
    """Shaded rectangle over start-end, layered underneath a year-on-x chart.

    `label` writes a small caption at the top of the band (e.g. "COVID").
    """
    band_df = pd.DataFrame({"start": [start], "end": [end], "label": [label or ""]})
    x = alt.X("start:Q", scale=alt.Scale(domain=domain, nice=False))
    rect = alt.Chart(band_df).mark_rect(opacity=opacity, color=color).encode(x=x, x2="end:Q")
    if not label:
        return rect
    text = alt.Chart(band_df).mark_text(align="left", baseline="top", dx=3, dy=3, color=color,
                                        fontSize=11).encode(x=x, y=alt.value(0), text="label:N")
    return rect + text


def window_lines(df, y, series, scale, start, end, y_title, y_format=",.1f", axis_format=",",
                 log_y=False, emphasize=None, height=340):
    """Year x `y` lines: faded full history, full opacity inside start-end, plus the band.

    `emphasize` (a series value) is drawn thick inside the window.
    `df` must hold derived columns only (Year Code, series, y).
    """
    domain = [int(df["Year Code"].min()), int(df["Year Code"].max())]
    x = alt.X("Year Code:Q", title="Year", axis=alt.Axis(format="d"),
              scale=alt.Scale(domain=domain, nice=False))
    y_enc = alt.Y(f"{y}:Q", title=y_title,
                  scale=alt.Scale(type="log") if log_y else alt.Scale(zero=False),
                  axis=alt.Axis(format=axis_format))
    color = alt.Color(f"{series}:N", scale=scale, legend=alt.Legend(title=None, orient="top"))
    tooltip = [alt.Tooltip(f"{series}:N", title=series), alt.Tooltip("Year Code:Q", title="Year"),
               alt.Tooltip(f"{y}:Q", title=y_title, format=y_format)]

    window = df[df["Year Code"].between(start, end)]
    faded = alt.Chart(df).mark_line(opacity=FADED).encode(x=x, y=y_enc, color=color)
    bright = alt.Chart(window).mark_line(strokeWidth=2).encode(x=x, y=y_enc, color=color)
    # Invisible points give every year a hover target that carries the tooltip.
    hits = alt.Chart(df).mark_point(size=120, opacity=0).encode(x=x, y=y_enc, tooltip=tooltip)

    layers = [year_band(start, end, domain), faded, bright]
    if emphasize is not None:
        thick = window[window[series] == emphasize]
        layers.append(alt.Chart(thick).mark_line(strokeWidth=4).encode(x=x, y=y_enc, color=color))
    layers.append(hits)
    return alt.layer(*layers).properties(height=height)


def improvement_bars(df, scale, series=None, height=None):
    """Horizontal bars of average annual improvement by age group, plus a zero rule.

    With `series` (e.g. "Location"), bars are grouped within each age group via yOffset.
    `df` columns: Age band, Improvement[, series].
    """
    color_field = series or "Age band"
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("Improvement:Q", title="Average annual mortality improvement",
                axis=alt.Axis(format=".1%")),
        y=alt.Y("Age band:N", sort=list(AGE_COLORS), title=None),
        color=alt.Color(f"{color_field}:N", scale=scale,
                        legend=alt.Legend(title=None, orient="top") if series else None),
        tooltip=[alt.Tooltip("Age band:N", title="Age group")]
        + ([alt.Tooltip(f"{series}:N", title=series)] if series else [])
        + [alt.Tooltip("Improvement:Q", title="Avg improvement / yr", format=".2%")],
    )
    if series:
        bars = bars.encode(yOffset=alt.YOffset(f"{series}:N", scale=alt.Scale(domain=scale.domain)))
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color=NEUTRAL).encode(
        x=alt.X("zero:Q", title=None))
    chart = alt.layer(bars, zero)
    n_bands = df["Age band"].nunique()
    return chart.properties(height=height or max(160, n_bands * (70 if series else 34)))


def heatmap(df, x, y, value, title, fmt, scheme, mid, y_sort=None, reverse=False):
    """Rect heatmap with labeled cells and a diverging color scale centered on `mid`."""
    x_enc = alt.X(f"{x}:N", sort=list(AGE_COLORS), title=None, axis=alt.Axis(labelAngle=0))
    y_enc = alt.Y(f"{y}:N", sort=y_sort, title=None)
    rect = alt.Chart(df).mark_rect().encode(
        x=x_enc, y=y_enc,
        color=alt.Color(f"{value}:Q", title=title,
                        scale=alt.Scale(scheme=scheme, domainMid=mid, reverse=reverse),
                        legend=alt.Legend(format=fmt, orient="right")),
        tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:N", title="Age group"),
                 alt.Tooltip(f"{value}:Q", title=title, format=fmt)],
    )
    text = alt.Chart(df).mark_text(fontSize=12).encode(
        x=x_enc, y=y_enc, text=alt.Text(f"{value}:Q", format=fmt), color=alt.value("black"))
    return alt.layer(rect, text).properties(height=max(140, df[y].nunique() * 42))


# ---------------------------------------------------------------- page 3

def fan_chart(actual, forecast, color, y_title, shade=None, height=360):
    """Actual line, dashed central forecast and a shaded 95% band, all on one year axis.

    actual: Year, Value. forecast: Year, Forecast, Lower, Upper.
    shade: optional list of (start, end, label, color) rectangles, e.g. training window or COVID.
    """
    domain = [int(min(actual["Year"].min(), forecast["Year"].min())),
              int(max(actual["Year"].max(), forecast["Year"].max()))]
    x = alt.X("Year:Q", title="Year", axis=alt.Axis(format="d"), scale=alt.Scale(domain=domain, nice=False))
    y_scale = alt.Scale(zero=False)

    layers = [year_band(s, e, domain, color=c, opacity=0.1, label=label) for s, e, label, c in (shade or [])]
    layers.append(alt.Chart(forecast).mark_area(opacity=0.2, color=color).encode(
        x=x, y=alt.Y("Lower:Q", title=y_title, scale=y_scale), y2="Upper:Q",
        tooltip=[alt.Tooltip("Year:Q", title="Year"),
                 alt.Tooltip("Lower:Q", title="95% lower", format=".2f"),
                 alt.Tooltip("Upper:Q", title="95% upper", format=".2f")]))
    layers.append(alt.Chart(forecast).mark_line(strokeDash=[6, 4], strokeWidth=2, color=color).encode(
        x=x, y=alt.Y("Forecast:Q", scale=y_scale),
        tooltip=[alt.Tooltip("Year:Q", title="Year"),
                 alt.Tooltip("Forecast:Q", title="Forecast", format=".2f")]))
    layers.append(alt.Chart(actual).mark_line(strokeWidth=2.5, color=NEUTRAL, point=alt.OverlayMarkDef(
        size=30, color=NEUTRAL)).encode(
        x=x, y=alt.Y("Value:Q", scale=y_scale),
        tooltip=[alt.Tooltip("Year:Q", title="Year"), alt.Tooltip("Value:Q", title="Actual", format=".2f")]))
    return alt.layer(*layers).properties(height=height)


def series_lines(df, x, y, series, scale, x_title, y_title, y_format=".2f", axis_format=None,
                 dashed=None, rule=None, height=300):
    """Plain multi-series line chart (e.g. improvement by single age, ax/bx by age, kt by year).

    `dashed` is a series value drawn dashed; `rule` adds a labeled horizontal reference line,
    given as (value, label).
    """
    x_enc = alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format="d"), scale=alt.Scale(nice=False))
    y_enc = alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False),
                  axis=alt.Axis(format=axis_format) if axis_format else alt.Axis())
    stroke_dash = alt.value([1, 0])
    if dashed is not None:
        stroke_dash = alt.condition(alt.datum[series] == dashed, alt.value([6, 4]), alt.value([1, 0]))
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=x_enc, y=y_enc,
        color=alt.Color(f"{series}:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
        strokeDash=stroke_dash,
        tooltip=[alt.Tooltip(f"{series}:N", title=series), alt.Tooltip(f"{x}:Q", title=x_title),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=y_format)],
    )
    layers = [lines]
    if rule is not None:
        value, label = rule
        rule_df = pd.DataFrame({"y": [value], "label": [label]})
        layers.append(alt.Chart(rule_df).mark_rule(color=NEUTRAL, strokeDash=[2, 2]).encode(y="y:Q"))
        layers.append(alt.Chart(rule_df).mark_text(align="left", dx=4, dy=-6, color=NEUTRAL, fontSize=11)
                      .encode(y="y:Q", x=alt.value(0), text="label:N"))
    return alt.layer(*layers).properties(height=height)
