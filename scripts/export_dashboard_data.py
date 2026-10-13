"""Export the derived-only tables the dashboard reads into data/dashboard/ (committed to the repo).

Run from the repo root, after harmonize_age_groups and validate_credibility:
    python -m scripts.export_dashboard_data

The dashboard never reads data/processed/. Everything it needs is written to data/dashboard/
without any Deaths or Population column, so that folder can be published, and the app run or
deployed from a fresh clone, without redistributing CDC WONDER counts. The script refuses to
write a table that still holds a count column.

- historic.csv: national life table 1968-2024 by the 7 age groups (page 1): death rate per
  100,000, lx, px and ex.
- validation_report.csv: the credibility outcome per (sex, location, cause): page 2's menus and
  its coverage matrix. Labels, age ranges and variance explained only.
- rates.csv: death rate per 100,000 by year x age group x sex x location x cause for every
  combination that passed credibility (page 2), plus an Exposure Weight: that cell's population
  divided by its own 1999-2024 average. A window's pooled rate sum(D) / sum(P) is then rebuilt
  exactly as sum(weight * rate) / sum(weight), without publishing either count.
- standard_weights.csv: US 2000 population shares by age group, for direct age standardization.

Page 3's model outputs are written to data/dashboard/model/ by scripts/national/forecast_precompute.py.
"""
from pathlib import Path

import pandas as pd

from src.actuarial.life_table import add_life_tab_col
from src.actuarial.load_filter import age_value, load_filter
from src.dashboard.data import BENCHMARK, STANDARD_YEAR
from src.data.harmonization import add_harmonized_col

PROCESSED = Path("data/processed")
OUT = Path("data/dashboard")

COUNT_COLUMNS = {"Deaths", "Population", "Expected Deaths"}
CELL = ["Sex Code", "State", "Cause List", "Age Group"]


def historic_table():
    """Page 1: national life table on the 7 age groups, derived columns only."""
    df = add_life_tab_col(load_filter(PROCESSED / "harmonized_1968-2024.csv", age_col="Age Group",
                                      age_values=age_value))
    df["Deaths per 100k"] = df["Deaths"] / df["Population"] * 100000
    return df[["Year Code", "Age Group", "Sex Code", "Deaths per 100k", "lx", "px", "ex"]]


def grouped_counts():
    """Every passed combination on the shared 7-group age axis (counts; never written out).

    Combinations that passed at single-year resolution are bucketed into the groups. That is
    safe: a single-year combination passed at every age from its floor, every floor sits on a
    group edge, and the two input tables never share a combination.
    """
    grouped = pd.read_csv(PROCESSED / "grouped_age_final.csv")
    single = add_harmonized_col(pd.read_csv(PROCESSED / "single_age_final.csv"), "Single-Year Ages Code")
    single = single.groupby(["Year Code", "Age Group", "Sex Code", "State", "Cause List"],
                            as_index=False)[["Deaths", "Population"]].sum()
    return pd.concat([grouped, single], ignore_index=True)


def rates_table(counts):
    """Page 2: rate per 100,000 per cell plus the relative exposure weight for pooling windows."""
    df = counts.copy()
    df["Rate"] = df["Deaths"] / df["Population"] * 100000
    df["Exposure Weight"] = df["Population"] / df.groupby(CELL)["Population"].transform("mean")
    return df[["Year Code", "Age Group", "Sex Code", "State", "Cause List", "Rate", "Exposure Weight"]]


def standard_weights_table(counts):
    """US, both sexes, all causes, STANDARD_YEAR population share by age group."""
    pop = counts[(counts["State"] == BENCHMARK) & (counts["Sex Code"] == "T")
                 & (counts["Cause List"] == "All")
                 & (counts["Year Code"] == STANDARD_YEAR)].set_index("Age Group")["Population"]
    return (pop / pop.sum()).rename("Weight").reset_index()


def write(frame, name):
    leaked = COUNT_COLUMNS & set(frame.columns)
    if leaked:
        raise ValueError(f"{name} still holds count columns {sorted(leaked)}; not writing it")
    frame.to_csv(OUT / name, index=False)
    print(f"Wrote {OUT / name} ({len(frame):,} rows)")


def run_export():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = grouped_counts()

    write(historic_table(), "historic.csv")
    write(rates_table(counts), "rates.csv")
    write(standard_weights_table(counts), "standard_weights.csv")

    write(pd.read_csv(PROCESSED / "validation_report.csv"), "validation_report.csv")


if __name__ == "__main__":
    run_export()
