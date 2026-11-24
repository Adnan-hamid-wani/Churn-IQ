import plotly.express as px
import streamlit as st

from src.dashboard import (
    OUTCOME_COLORS, OUTCOME_LABELS, OUTCOME_PATTERNS, chart, churn_rate_chart, outcome_donut, sample_data,
)
from src.schema import CATEGORIES, FIELD_LABELS, ID_COLUMN, NUMERIC_FEATURES

st.title("Data Explorer")
st.markdown("Filter the IBM sample to compare churn across customer groups.")
data = sample_data()

def clear_filters() -> None:
    st.session_state.update(contract_filter=[], internet_filter=[], outcome_filter=[])


with st.container(border=True, key="surface_filters"):
    contract_filter, internet_filter, outcome_filter, reset = st.columns([3, 3, 3, 2], vertical_alignment="bottom")
    contracts = contract_filter.multiselect("Contract", CATEGORIES["Contract"], key="contract_filter")
    internet = internet_filter.multiselect("Internet service", CATEGORIES["InternetService"], key="internet_filter")
    outcomes = outcome_filter.multiselect(
        "Outcome", list(OUTCOME_LABELS), format_func=OUTCOME_LABELS.get, key="outcome_filter",
    )
    reset.button("Clear filters", disabled=not (contracts or internet or outcomes), width="stretch", on_click=clear_filters)

filtered = data
if contracts:
    filtered = filtered[filtered["Contract"].isin(contracts)]
if internet:
    filtered = filtered[filtered["InternetService"].isin(internet)]
if outcomes:
    filtered = filtered[filtered["Churn"].isin(outcomes)]

if filtered.empty:
    st.info("No customers match these filters. Change a selection or clear all filters.")
    st.stop()

churned = filtered["Churn"].eq("Yes")
count, rate = st.columns(2)
count.metric("Matching customers", f"{len(filtered):,}", border=True)
rate.metric("Recorded churn rate", f"{churned.mean():.1%}", border=True)

quantitative, categorical = st.columns(2)
with quantitative.container(border=True, key="surface_numeric"):
    st.subheader("Numerical distribution")
    feature = st.selectbox("Numerical field", NUMERIC_FEATURES, format_func=FIELD_LABELS.get)
    plotted = filtered.assign(Outcome=filtered["Churn"].map(OUTCOME_LABELS))
    figure = px.histogram(
        plotted, x=feature, color="Outcome", barmode="group", nbins=24,
        color_discrete_map=OUTCOME_COLORS, pattern_shape="Outcome", pattern_shape_map=OUTCOME_PATTERNS,
        labels={feature: FIELD_LABELS[feature]}, histnorm="percent",
    )
    figure.update_yaxes(title="Share within each outcome (%)")
    figure.update_layout(bargap=0.12)
    figure.update_traces(marker_pattern_fillmode="overlay", marker_pattern_fgcolor="#FFFFFF", marker_pattern_fgopacity=0.35)
    chart(figure, legend=True)

with categorical.container(border=True, key="surface_category"):
    st.subheader("Categorical breakdown")
    by_category, outcome_mix = st.tabs(["Churn by category", "Outcome mix"])
    with by_category:
        categories = ("InternetService", "Contract", "PaymentMethod", "OnlineSecurity",
                      "TechSupport", "gender", "SeniorCitizen", "Partner", "Dependents")
        category = st.selectbox("Categorical field", categories, format_func=FIELD_LABELS.get)
        groups = filtered
        if category == "SeniorCitizen":
            groups = filtered.assign(SeniorCitizen=filtered["SeniorCitizen"].map({0: "No", 1: "Yes"}))
        churn_rate_chart(groups, category, max_rate=1.1)
    with outcome_mix, st.container(key="outcome_mix"):
        outcome_donut(churned)

with st.expander("Preview matching records"):
    st.caption("First 50 matching rows, with customer identifiers omitted.")
    st.dataframe(
        filtered.drop(columns=ID_COLUMN).head(50),
        hide_index=True, width="stretch",
    )
