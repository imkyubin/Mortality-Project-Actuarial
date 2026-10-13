"""Combine the state and US cause-of-death exports into one table with a Total sex.

Run from the repo root: python -m scripts.sublevel.combine_exports

Reads eleven pre-cleaned CDC WONDER single-year-age exports from data/processed/: one per state
(California, Florida, New York, Texas; 8 causes, 1999-2020), "State All Causes", "US 8 Causes"
and "US All Causes" (1999-2020), and their "... 2" counterparts for 2021-2024. Stacks them, adds
Sex Code "T" rows as Female + Male, and writes data/processed/final.csv, the input to
validate_credibility.py. ("US All Causes.csv" also carries a text Sex column, which passes
through to final.csv; nothing downstream reads it.)
"""

import pandas as pd

ny_df = pd.read_csv("data/processed/New York.csv")

ca_df = pd.read_csv("data/processed/California.csv")

tx_df = pd.read_csv("data/processed/Texas.csv")

fl_df = pd.read_csv("data/processed/Florida.csv")

st82_df = pd.read_csv("data/processed/State 8 Causes 2.csv")

stall1_df = pd.read_csv("data/processed/State All Causes.csv")

stall2_df = pd.read_csv("data/processed/State All Causes 2.csv")

us81_df = pd.read_csv("data/processed/US 8 Causes.csv")

us82_df = pd.read_csv("data/processed/US 8 Causes 2.csv")

usall1_df = pd.read_csv("data/processed/US All Causes.csv")

usall2_df = pd.read_csv("data/processed/US All Causes 2.csv")

dfs = [ny_df, ca_df, tx_df, fl_df, st82_df, stall1_df, stall2_df, us81_df, us82_df, usall1_df, usall2_df]

mf_df = pd.concat(dfs, ignore_index=True)

total_df = mf_df.groupby(
    ["Year Code", "Single-Year Ages Code", "State", "Cause List"], as_index=False
    )[["Deaths", "Population"]].sum()

total_df["Sex Code"] = "T"

combined_df = pd.concat([mf_df, total_df], ignore_index=True)

combined_df = combined_df.sort_values(by=["State", "Cause List", "Sex Code", "Year Code", "Single-Year Ages Code"])

combined_df.to_csv("data/processed/final.csv", index=False)

