"""Streamlit entry point for the US Mortality Explorer dashboard (three pages in src/dashboard/).

Run from the repo root: streamlit run streamlit_app.py
Streamlit puts this file's folder (the repo root) on the import path, so `src` imports. This
is also the entrypoint Streamlit Community Cloud uses for the hosted app.

The pages read only data/dashboard/: derived tables committed to the repo, so the app runs
from a fresh clone without the CDC WONDER data.
"""
import streamlit as st

from src.dashboard import forecast_view, historic_view, subsection_view

st.set_page_config(page_title="US Mortality Explorer", layout="wide")

page = st.navigation([
    st.Page(historic_view.render, title="National Trends 1968-2024", url_path="historic",
            icon=":material/timeline:", default=True),
    st.Page(subsection_view.render, title="State & Cause Comparison", url_path="subsection",
            icon=":material/compare_arrows:"),
    st.Page(forecast_view.render, title="Mortality Forecast", url_path="forecast",
            icon=":material/trending_down:"),
])
page.run()
