import plotly.express as px
import streamlit as st

from src.dashboard import OUTCOME_COLORS, OUTCOME_LABELS, chart, churn_rate_chart, sample_data
from src.schema import CATEGORIES, FIELD_LABELS, ID_COLUMN, NUMERIC_FEATURES

st.title("Data Explorer")
st.markdown("Filter the IBM sample to compare churn across customer groups.")
data = sample_data()

with st.container(border=True, key="surface_filters"):
    contract_filter, internet_filter, outcome_filter = st.columns(3)
    contracts = contract_filter.multiselect("Contract", CATEGORIES["Contract"], key="contract_filter")
    internet = internet_filter.multiselect("Internet service", CATEGORIES["InternetService"])
    outcomes = outcome_filter.multiselect(
        "Outcome", list(OUTCOME_LABELS), format_func=OUTCOME_LABELS.get,
    )

filtered = data
if contracts:
    filtered = filtered[filtered["Contract"].isin(contracts)]
if internet:
    filtered = filtered[filtered["InternetService"].isin(internet)]
if outcomes:
    filtered = filtered[filtered["Churn"].isin(outcomes)]

if filtered.empty:
    st.info("That combination matches no customers. Change or clear a filter to continue.")
    st.stop()

count, rate = st.columns(2)
count.metric("Matching customers", f"{len(filtered):,}", border=True)
rate.metric("Recorded churn rate", f"{filtered['Churn'].eq('Yes').mean():.1%}", border=True)

quantitative, categorical = st.columns(2)
with quantitative.container(border=True, key="surface_numeric"):
    st.subheader("Numerical distribution")
    feature = st.selectbox("Numerical field", NUMERIC_FEATURES, format_func=FIELD_LABELS.get)
    plotted = filtered.assign(Outcome=filtered["Churn"].map(OUTCOME_LABELS))
    figure = px.histogram(
        plotted, x=feature, color="Outcome", barmode="overlay", nbins=24, opacity=0.66,
        color_discrete_map=OUTCOME_COLORS, labels={feature: FIELD_LABELS[feature]}, histnorm="percent",
    )
    figure.update_yaxes(title="Share within each outcome (%)")
    chart(figure, legend=True)

with categorical.container(border=True, key="surface_category"):
    st.subheader("Churn by category")
    categories = ("InternetService", "Contract", "PaymentMethod", "OnlineSecurity",
                  "TechSupport", "gender", "SeniorCitizen", "Partner", "Dependents")
    category = st.selectbox("Categorical field", categories, format_func=FIELD_LABELS.get)
    groups = filtered
    if category == "SeniorCitizen":
        groups = filtered.assign(SeniorCitizen=filtered["SeniorCitizen"].map({0: "No", 1: "Yes"}))
    churn_rate_chart(groups, category, max_rate=1.1)

with st.expander("Preview matching records"):
    st.caption("First 50 matching rows, with customer identifiers omitted.")
    st.dataframe(
        filtered.drop(columns=ID_COLUMN).head(50),
        hide_index=True, width="stretch",
    )
