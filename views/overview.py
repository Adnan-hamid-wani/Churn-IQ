import streamlit as st

from src.dashboard import OUTCOME_COLORS, churn_rate_chart, metric_cards, sample_data

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

contract_panel, outcome_panel = st.columns([3, 2], gap="medium")
with contract_panel.container(border=True, height="stretch", key="surface_contract"):
    st.header("Churn by contract")
    rates = churn_rate_chart(data, "Contract", max_rate=0.55, height=235)
    with st.expander("View contract data"):
        st.table({
            "Contract": rates["Contract"].tolist()[::-1],
            "Customers": rates["customers"].tolist()[::-1],
            "Recorded churn": [f"{rate:.1%}" for rate in rates["rate"][::-1]],
        })

with outcome_panel.container(border=True, height="stretch", key="surface_outcomes"):
    st.header("Customer outcomes")
    counts = {"Retained": int((~churned).sum()), "Churned": int(churned.sum())}
    total = len(data)
    blocks = "".join(
        f'<span class="churn-outcome-block" style="--customers:{count};'
        f'--outcome-color:{OUTCOME_COLORS[label]}"></span>'
        for label, count in counts.items()
    )
    rows = "".join(
        f'<div style="--outcome-color:{OUTCOME_COLORS[label]}"><dt>{label}</dt>'
        f'<dd>{count:,}</dd><dd>{count / total:.1%}</dd></div>'
        for label, count in counts.items()
    )
    st.html(
        f'<div class="churn-outcome-bar" aria-hidden="true">{blocks}</div>'
        f'<dl class="churn-outcomes" aria-label="Customer outcomes">{rows}</dl>'
    )

    high, low = rates.iloc[-1], rates.iloc[0]
    with st.container(key="contract_observation"):
        st.markdown(
            f"**{100 * (high['rate'] - low['rate']):.1f} percentage-point gap**\n\n"
            f"Recorded churn was {high['rate']:.1%} for {high['Contract'].lower()} contracts "
            f"and {low['rate']:.1%} for {low['Contract'].lower()} contracts."
        )
