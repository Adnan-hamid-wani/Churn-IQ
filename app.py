"""ChurnIQ Streamlit entry point."""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="ChurnIQ",
    page_icon=":material/monitoring:",
    layout="wide",
    menu_items={"About": "ChurnIQ is an academic portfolio demo on IBM's fictional telecom sample."},
)

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/data_explorer.py", title="Data Explorer", url_path="explore"),
    st.Page("views/predict_churn.py", title="Predict Churn", url_path="predict"),
    st.Page("views/model_performance.py", title="Model Performance", url_path="performance"),
    st.Page("views/batch_predictions.py", title="Batch Predictions", url_path="batch"),
]
page = st.navigation(pages, position="top")
st.html(Path(__file__).resolve().parent / "assets" / "styles.css")
st.html(
    '<div class="churn-brand"><span class="churn-logo" aria-hidden="true">IQ</span>'
    '<span>ChurnIQ</span><span class="churn-brand-context">Telecom churn analytics</span></div>'
)
page.run()
