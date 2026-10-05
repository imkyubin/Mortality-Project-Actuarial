"""Cached data loaders and the credibility lookup the views read from.

No layout code here, so the API and reporting layers can reuse the same lookup
(api_layer.credibility_gate in config.yaml).

TODO:
- A path constant pointing at data/processed/.
- Label maps: Sex Code -> display name, CDC WONDER Cause List string -> short cause name.
- A @st.cache_data loader for harmonized_1968-2024.csv with life-table columns
  (src.step2.load_filter.load_filter, src.step2.life_table.add_life_tab_col) plus a
  per-100,000 death rate.
- A @st.cache_data loader for validation_report.csv (the credibility lookup).
- A @st.cache_data loader for single_age_final.csv / grouped_age_final.csv.
- lookup_combination(state, sex_code, cause): the one validation_report row for a combination.
- subsection_data(state, sex_code, cause, resolution): that combination's rows from the table
  its Resolution points to.

Deaths/Population are loaded only to derive rates; nothing here should hand them to a view
for display (see the CDC WONDER display rule in CLAUDE.md).

Reference: templates/dashboard/data.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.step2.life_table import add_life_tab_col
from src.step2.load_filter import age_value, load_filter

PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"

@st.cache_data
def load_historic():
  print("Loading 1968-2024 Grouped Ages CSV...")
  df = load_filter(path=PROCESSED / "harmonized_1968-2024.csv", age_col="Age Group", age_values=age_value)
  df = add_life_tab_col(df)
  df["Deaths per 100k"] = df["Deaths"] / df["Population"] * 100000
  df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_value, ordered=True)
  return df

@st.cache_data
def load_summary():
  df = load_historic()

  young = df[df["Age Group"]=="20-24 years"]
  summary = young[["Year Code", "Sex Code", "ex"]]

  old = df[df["Age Group"]=="75-84 years"].copy()
  old["Prob Reach 85"] = old["lx"] * old["px"] / 100000

  return summary.merge(old[["Year Code", "Sex Code", "Prob Reach 85"]], on=["Year Code", "Sex Code"])


