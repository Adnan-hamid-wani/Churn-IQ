import plotly.graph_objects as go
import streamlit as st

from src.dashboard import BLUE, TEAL, chart, model_release
from src.explain import explain_case
from src.predict import predict_one
from src.schema import CATEGORIES, FIELD_LABELS, INTERNET_ADDONS

st.title("Predict Churn")
st.markdown("Score a fictional customer using their account history, billing and services.")
bundle, _ = model_release()

with st.form("customer_profile", border=True):
    st.subheader("Account and billing")
    first, second, third = st.columns(3)
    tenure = first.number_input("Tenure (months)", min_value=0, value=4, step=1)
    monthly = second.number_input("Monthly charges", min_value=0.0, value=81.40, step=0.01)
    total = third.number_input(
        "Total charges", min_value=0.0, value=324.50, step=0.01,
        help="You can clear this value if the cumulative charge is unavailable.",
    )
    contract = first.selectbox("Contract", CATEGORIES["Contract"])
    payment = second.selectbox("Payment method", CATEGORIES["PaymentMethod"])
    paperless = third.selectbox("Paperless billing", ("Yes", "No"))

    st.subheader("Services")
    left, right = st.columns(2)
    phone = left.selectbox("Phone setup", ("Single line", "Multiple lines", "No phone service"))
    internet = right.selectbox(
        "Internet service", CATEGORIES["InternetService"], index=1,
        help="When no internet service is selected, all add-ons are marked unavailable.",
    )
    add_ons = st.pills(
        "Internet add-ons", INTERNET_ADDONS, selection_mode="multi",
        format_func=FIELD_LABELS.get, default=["StreamingTV"],
    )
    submitted = st.form_submit_button("Calculate churn risk", type="primary")

if submitted:
    profile = {
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Contract": contract, "PaymentMethod": payment, "PaperlessBilling": paperless,
        "PhoneService": "No" if phone == "No phone service" else "Yes",
        "MultipleLines": {"Single line": "No", "Multiple lines": "Yes"}.get(phone, phone),
        "InternetService": internet,
        **{
            field: "No internet service" if internet == "No" else ("Yes" if field in add_ons else "No")
            for field in INTERNET_ADDONS
        },
    }
    st.session_state.pop("scored_profile", None)
    try:
        st.session_state["scored_profile"] = (predict_one(bundle, profile), explain_case(bundle, profile))
    except ValueError as error:
        st.error(str(error))

if "scored_profile" in st.session_state:
    prediction, explanations = st.session_state["scored_profile"]
    st.subheader("Latest prediction")
    probability, decision, risk = st.columns(3)
    probability.metric("Churn probability", f"{prediction.probability:.1%}", border=True)
    decision.metric("Predicted churn", prediction.predicted_class, border=True)
    risk.metric("Risk band", prediction.risk, border=True)
    st.caption(
        "This is a statistical estimate for the last submitted profile. "
        "Actual customer outcomes can differ."
    )

    with st.container(border=True, key="surface_local_explanation"):
        st.subheader("What changed this score?")
        st.markdown(
            "We changed one field group at a time to values observed in training, then scored the edited "
            "profile. The bars show how many percentage points the model score moved after each edit."
        )
        changes = [entry for entry in explanations if abs(entry["difference"]) >= 0.0005][:5]
        if changes:
            differences = [100 * entry["difference"] for entry in changes]
            figure = go.Figure(go.Bar(
                x=differences, y=[entry["group"] for entry in changes], orientation="h",
                marker_color=[BLUE if value > 0 else TEAL for value in differences],
                text=[f"{value:+.1f} pp" for value in differences], textposition="outside",
                cliponaxis=False, hovertemplate="%{y} · %{x:+.1f} pp<extra></extra>",
            ))
            extent = max(abs(value) for value in differences) * 1.35
            figure.update_xaxes(title="Change in churn probability (percentage points)", range=[-extent, extent])
            figure.update_yaxes(autorange="reversed")
            chart(figure, height=275)
            with st.expander("See the exact input edits"):
                for entry in changes:
                    edits = ", ".join(
                        f"{FIELD_LABELS[field]} → {value:g}" if isinstance(value, (int, float))
                        else f"{FIELD_LABELS[field]} → {value}"
                        for field, value in entry["changes"].items()
                    )
                    st.markdown(f"- **{entry['group']}**, {entry['difference'] * 100:+.1f} pp · {edits}")
        else:
            st.info("The tested edits had little effect on this prediction.")
        st.caption(
            "The edits can interact, and each result depends on the reference values chosen. "
            "The changes describe how the model responds and cannot establish a real-world cause."
        )

with st.expander("How risk bands are assigned"):
    low, threshold = bundle["risk_low_cutoff"], bundle["threshold"]
    st.markdown(
        f"- **Low** · Below {low:.2%}\n"
        f"- **Medium** · From {low:.2%} up to {threshold:.2%}\n"
        f"- **High** · {threshold:.2%} and above. This is also the cutoff for predicting churn."
    )
    st.caption("The app applies the full-precision cutoffs before rounding scores for display.")
