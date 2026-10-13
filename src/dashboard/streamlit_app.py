"""Streamlit entry point.

Run from the repo root: python -m streamlit run src/dashboard/streamlit_app.py
(`python -m` puts the repo root on sys.path so `src` imports.)

Page 3 reads data/processed/model/, written by: python -m scripts.national.forecast_precompute
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
