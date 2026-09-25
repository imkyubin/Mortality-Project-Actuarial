"""TEMPLATE - Streamlit entry point.

Run from the repo root: python -m streamlit run templates/dashboard/streamlit_app.py
(`python -m` puts the repo root on sys.path so `src` and `templates` import.)
"""
import streamlit as st

from templates.dashboard import historic_view, subsection_view

st.set_page_config(page_title="US Mortality Explorer", layout="wide")

page = st.navigation([
    st.Page(historic_view.render, title="National trends 1968-2024", url_path="historic",
            icon=":material/timeline:", default=True),
    st.Page(subsection_view.render, title="State & cause comparison", url_path="subsection",
            icon=":material/compare_arrows:"),
])
page.run()
