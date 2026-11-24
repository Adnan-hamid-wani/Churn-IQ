import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import BLUE, MUTED, TEAL, chart, metric_cards, model_release
from src.schema import FIELD_LABELS

st.title("Model Performance")
st.markdown("Compare model selection on the training folds with final performance on held-out data.")
_, metrics = model_release()
test = metrics["test"]
threshold = metrics["threshold"]
selected = metrics["selected_model"]
row_count = metrics["splits"]["test"]["rows"]

with st.container(border=True):
    st.subheader(selected)
    st.markdown(
        f"The model led the {metrics['cv_folds']}-fold average precision comparison. "
        f"The **{threshold:.2%}** classification cutoff maximized F1 on the separate validation split. "
        f"The following test results use **{row_count:,}** held-out customers."
    )
    metric_cards([
        ("Average precision", f"{test['average_precision']:.3f}"),
        ("ROC-AUC", f"{test['roc_auc']:.3f}"),
        ("Precision", f"{test['precision']:.1%}"),
        ("Recall", f"{test['recall']:.1%}"),
    ])
    st.caption(
        f"Held-out accuracy {test['accuracy']:.1%} · F1 {test['f1']:.3f} · "
        f"Brier score {test['brier_score']:.3f}"
    )

st.subheader("Model comparison")
comparison = pd.DataFrame([
    {"Model": row["model"], "Mean CV average precision": row["cv_average_precision"],
     "Fold standard deviation": row["cv_std"]}
    for row in metrics["comparison"]
])
st.dataframe(
    comparison, hide_index=True, width="stretch",
    column_config={
        "Mean CV average precision": st.column_config.NumberColumn(format="%.4f"),
        "Fold standard deviation": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption("Each row uses the best settings found for that model on the same training folds.")

roc_panel, pr_panel = st.columns(2)
curves = metrics["test_curves"]
with roc_panel.container(border=True, key="surface_roc"):
    st.subheader("ROC curve")
    roc = curves["roc"]
    figure = go.Figure([
        go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line={"color": BLUE, "width": 3}),
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": MUTED, "dash": "dot"},
                   hoverinfo="skip"),
    ])
    figure.update_xaxes(title="False positive rate", range=[0, 1], tickformat=".0%")
    figure.update_yaxes(title="True positive rate", range=[0, 1.02], tickformat=".0%")
    chart(figure)

with pr_panel.container(border=True, key="surface_pr"):
    st.subheader("Precision–recall curve")
    pr = curves["precision_recall"]
    prevalence = curves["prevalence"]
    figure = go.Figure([
        go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                   line={"color": TEAL, "width": 3, "shape": "hv"}),
        go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines",
                   line={"color": MUTED, "dash": "dot"}, hoverinfo="skip"),
    ])
    figure.update_xaxes(title="Recall", range=[0, 1], tickformat=".0%")
    figure.update_yaxes(title="Precision", range=[0, 1.02], tickformat=".0%")
    chart(figure)

matrix_panel, threshold_panel = st.columns(2)
with matrix_panel.container(border=True, key="surface_matrix"):
    st.subheader("Confusion matrix")
    counts = test["confusion_matrix"]
    figure = go.Figure(go.Heatmap(
        z=counts, text=counts, texttemplate="%{text}", colorscale="Blues", showscale=False,
        x=["Retained", "Churn"], y=["Retained", "Churn"],
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} customers<extra></extra>",
    ))
    figure.update_yaxes(title="Actual outcome", autorange="reversed")
    figure.update_xaxes(title="Predicted outcome")
    chart(figure, height=245)
    st.caption(f"{counts[0][1]} false alerts and {counts[1][0]} missed churners in the held-out set.")

with threshold_panel.container(border=True, key="surface_threshold"):
    st.subheader("Threshold trade-off")
    st.markdown(
        "A false positive flags a customer who stayed, which can waste outreach. "
        "A false negative misses a customer who left, which can cost a retention opportunity."
    )
    validation = metrics["validation"]
    points = [(f"{threshold:.2%}", validation["chosen"]), ("50.00%", validation["at_0_50"])]
    st.dataframe(pd.DataFrame([
        {
            "Cutoff": label, "Precision": f"{score['precision']:.1%}", "Recall": f"{score['recall']:.1%}",
            "False alerts": score["confusion_matrix"][0][1], "Missed churners": score["confusion_matrix"][1][0],
        }
        for label, score in points
    ]), hide_index=True, width="stretch")
    st.caption("This comparison uses validation data. F1 balances precision and recall.")

with st.container(border=True):
    st.subheader("Global feature importance")
    st.markdown(
        "We shuffled each input in the held-out data eight times and measured the change in average "
        "precision. A larger drop means the model relied more on that field in this sample."
    )
    ordered = metrics["global_importance"][:10][::-1]
    figure = go.Figure(go.Bar(
        x=[row["mean"] for row in ordered], y=[FIELD_LABELS[row["feature"]] for row in ordered],
        orientation="h", marker_color=BLUE, error_x={"type": "data", "array": [row["std"] for row in ordered]},
        hovertemplate="%{y} · %{x:.4f} average precision drop<extra></extra>",
    ))
    figure.update_xaxes(title="Mean drop in average precision")
    chart(figure, height=370)
    st.caption(
        "The bars show the ten highest mean drops. Error bars show the standard deviation across shuffles. "
        "Correlated inputs can share importance, and shuffling may create combinations that rarely occur."
    )
    with st.expander("See importance for every feature"):
        st.dataframe(pd.DataFrame([
            {"Feature": FIELD_LABELS[row["feature"]], "Mean drop": row["mean"], "Standard deviation": row["std"]}
            for row in metrics["global_importance"]
        ]), hide_index=True, width="stretch")

with st.expander("Metric definitions"):
    st.markdown("""
- **Accuracy** is the share of all classifications that were correct.
- **Precision** is the share of predicted churners who actually churned.
- **Recall** is the share of actual churners identified by the model.
- **F1** is the harmonic mean of precision and recall.
- **ROC-AUC** measures how well scores rank churners above retained customers across cutoffs.
- **Average precision** summarizes precision across gains in recall. The dotted PR reference is the churn rate.
- **Brier score** measures squared error of predicted probabilities. Lower is better.
""")
