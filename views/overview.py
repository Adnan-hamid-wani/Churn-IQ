import streamlit as st

from src.dashboard import (
    churn_rate_chart, metric_cards, outcome_donut, sample_data, tenure_churn, tenure_churn_chart,
)
from src.schema import CATEGORIES

with st.container(key="overview_header"):
    heading, action = st.columns([3, 1], vertical_alignment="center")
    with heading:
        st.title("Customer churn overview")
        st.markdown("IBM Telco Customer Churn sample")
    with action.container(key="overview_action", horizontal=True, horizontal_alignment="right"):
        st.page_link(
            "views/predict_churn.py", label="Score a sample customer",
            icon=":material/arrow_forward:", icon_position="right",
        )

data = sample_data()
churned = data["Churn"].eq("Yes")
metric_cards([
    ("Total customers", f"{len(data):,}"),
    ("Recorded churn rate", f"{churned.mean():.1%}"),
    ("Average monthly charge", f"${data['MonthlyCharges'].mean():.2f}"),
    ("Average tenure", f"{data['tenure'].mean():.1f} months"),
], highlight="Recorded churn rate")

with st.container(border=True, key="surface_tenure"):
    title, filter_panel = st.columns([3, 1], vertical_alignment="center")
    with title:
        st.header("Churn by customer tenure")
        st.caption("Churn rate and the share of monthly charges from churned customers in each tenure group.")
    contract = filter_panel.selectbox("Contract", ["All contracts", *CATEGORIES["Contract"]], key="tenure_contract")
    selection = data if contract == "All contracts" else data[data["Contract"].eq(contract)]
    groups = tenure_churn(selection)
    tenure_churn_chart(groups)
    with st.expander("View tenure data"):
        populated = groups[groups["customers"].gt(0)]
        st.table({
            "Tenure (months)": populated.index.tolist(),
            "Customers": populated["customers"].tolist(),
            "Customer churn rate": [f"{value:.1%}" for value in populated["churn_rate"]],
            "Monthly charge share": populated["charge_share"].map("{:.1%}".format, na_action="ignore").fillna("—").tolist(),
        })

outcome_panel, contract_panel = st.columns([2, 3], gap="medium")
with outcome_panel.container(border=True, height="stretch", key="surface_outcomes"):
    st.header("Customer outcomes")
    outcome_donut(churned)

with contract_panel.container(border=True, height="stretch", key="surface_contract"):
    st.header("Churn by contract")
    rates = churn_rate_chart(data, "Contract", max_rate=0.55, height=235)
    high, low = rates.iloc[-1], rates.iloc[0]
    with st.container(key="contract_observation"):
        st.markdown(
            f"**{100 * (high['rate'] - low['rate']):.1f} percentage-point gap**\n\n"
            f"Recorded churn was {high['rate']:.1%} for {high['Contract'].lower()} contracts "
            f"and {low['rate']:.1%} for {low['Contract'].lower()} contracts."
        )

    with st.expander("View contract data"):
        st.table({
            "Contract": rates["Contract"].tolist()[::-1],
            "Customers": rates["customers"].tolist()[::-1],
            "Recorded churn": [f"{rate:.1%}" for rate in rates["rate"][::-1]],
        })
