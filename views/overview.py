import plotly.graph_objects as go
import streamlit as st

from src.dashboard import BLUE, TEAL, chart, churn_rate_chart, metric_cards, sample_data

st.title("Customer churn overview")
data = sample_data()

st.html("""
<div class="churn-hero">
    <div>
        <p><strong class="churn-hero-title">How churn varies across customers</strong></p>
        <p class="churn-hero-copy">Compare recorded churn by contract and service, then score a fictional customer with the trained model.</p>
    </div>
    <div class="churn-hero-meta"><span>Dataset</span><strong>IBM Telco Customer Churn</strong></div>
</div>
""")

churned = data["Churn"].eq("Yes")
metric_cards([
    ("Total customers", f"{len(data):,}"),
    ("Recorded churn rate", f"{churned.mean():.1%}"),
    ("Average monthly charge", f"${data['MonthlyCharges'].mean():.2f}"),
    ("Average tenure", f"{data['tenure'].mean():.1f} months"),
])

contract_panel, outcome_panel = st.columns([3, 2])
with contract_panel.container(border=True, key="surface_contract"):
    st.subheader("Churn by contract")
    rates = churn_rate_chart(data, "Contract", max_rate=0.55)
with outcome_panel.container(border=True, key="surface_outcomes"):
    st.subheader("Customer outcomes")
    retained = int((~churned).sum())
    lost = int(churned.sum())
    distribution = go.Figure(go.Bar(
        x=[retained, lost], y=["Retained", "Churned"], orientation="h",
        marker_color=[BLUE, TEAL], text=[f"{retained:,}", f"{lost:,}"],
        textposition="auto", hovertemplate="%{y} · %{x:,}<extra></extra>",
    ))
    distribution.update_yaxes(autorange="reversed")
    distribution.update_xaxes(title="Customers", range=[0, max(retained, lost) * 1.18], tickformat=",")
    chart(distribution)

with st.container(border=True, key="surface_observation"):
    st.subheader("What stands out")
    high = rates.iloc[-1]
    low = rates.iloc[0]
    st.write(
        f"**{high['Contract']}** customers had the highest recorded churn rate "
        f"(**{high['rate']:.1%}**), while **{str(low['Contract']).lower()}** customers "
        f"had the lowest (**{low['rate']:.1%}**). These are associations within the IBM sample."
    )
    st.page_link("views/predict_churn.py", label="Score a sample customer", icon=":material/arrow_forward:")
