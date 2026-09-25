"""TEMPLATE - Altair chart builders shared by both dashboard views.

Colors are fixed per entity (a sex or an age group always keeps the same hue,
whatever the filters), drawn in order from a CVD-validated categorical palette.
"""
import altair as alt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEX_COLORS = {"Female": PALETTE[0], "Male": PALETTE[1], "Total": PALETTE[2]}

AGE_BAND_COLORS = dict(zip(["20-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84"], PALETTE))

PERIOD_COLORS = {"Start year": PALETTE[0], "End year": PALETTE[1]}


def color_scale(colors, keep=None):
    """Fixed entity -> color scale; `keep` restricts the domain without repainting survivors."""
    items = [(k, v) for k, v in colors.items() if keep is None or k in keep]
    return alt.Scale(domain=[k for k, _ in items], range=[v for _, v in items])


def line_chart(df, x, y, series, scale, x_title, y_title, log_y=False, y_format=",.1f",
               x_type="Q", height=340):
    """Multi-series line chart with a hover crosshair and tooltip."""
    x_enc = alt.X(f"{x}:{x_type}", title=x_title,
                  axis=alt.Axis(format="d") if x_type == "Q" else alt.Axis())
    y_enc = alt.Y(f"{y}:Q", title=y_title,
                  scale=alt.Scale(type="log") if log_y else alt.Scale(zero=False))
    color = alt.Color(f"{series}:N", scale=scale, legend=alt.Legend(title=None, orient="top"))

    hover = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)

    base = alt.Chart(df).encode(x=x_enc)
    lines = base.mark_line(strokeWidth=2).encode(y=y_enc, color=color)
    points = base.mark_point(filled=True, size=64).encode(
        y=y_enc, color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )
    rule = base.mark_rule(color="#898781").encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
    )
    # Invisible, larger hit targets drive the hover selection and carry the per-point tooltip.
    point_tips = base.mark_point(size=200, opacity=0).encode(
        y=y_enc,
        tooltip=[alt.Tooltip(f"{series}:N", title="Series"),
                 alt.Tooltip(f"{x}:{x_type}", title=x_title),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=y_format)],
    ).add_params(hover)
    return (lines + rule + points + point_tips).properties(height=height)
