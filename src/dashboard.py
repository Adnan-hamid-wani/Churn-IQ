"""Shared Streamlit data access and chart styling."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.artifacts import load_release
from src.data_loader import load_dataset
from src.schema import TARGET

NAVY = "#142840"
BLUE = "#315A79"
TEAL = "#116858"
MUTED = "#465D66"
GRID = "#D5E1DD"
OUTCOME_LABELS = {"No": "Retained", "Yes": "Churned"}
OUTCOME_COLORS = {"Retained": BLUE, "Churned": TEAL}


@st.cache_data(show_spinner="Loading the verified IBM sample…")
def _sample() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource(show_spinner="Loading the trained model…")
def _release() -> tuple[dict, dict]:
    return load_release()


def sample_data() -> pd.DataFrame:
    try:
        return _sample()
    except (OSError, ValueError):
        st.error("The IBM sample is unavailable. Try again shortly or run the dataset download command locally.")
        st.stop()


def model_release() -> tuple[dict, dict]:
    try:
        return _release()
    except (OSError, ValueError, KeyError, TypeError):
        st.error("Model files are missing or inconsistent. Use the published artifacts or run training locally.")
        st.stop()


def metric_cards(items: list[tuple[str, str]], *, highlight: str | None = None) -> None:
    entries = "".join(
        f'<div class="churn-kpi{" is-highlighted" if label == highlight else ""}">'
        f'<dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>'
        for label, value in items
    )
    st.html(f'<dl class="churn-kpis" aria-label="Key metrics">{entries}</dl>')


def chart(
    figure: go.Figure, *, height: int = 300, legend: bool = False, legend_below: bool = False,
) -> None:
    legend_layout = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0}
    if legend_below:
        legend_layout.update(x=0.5, xanchor="center", y=0, yanchor="bottom", yref="container")
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"family": "Source Sans, Arial, sans-serif", "color": NAVY, "size": 13},
        margin={"l": 8, "r": 12, "t": 14, "b": 88 if legend_below else 8},
        showlegend=legend,
        legend=legend_layout,
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, fixedrange=True)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, fixedrange=True)
    st.plotly_chart(figure, theme=None, width="stretch", config={"displayModeBar": False})


def tenure_churn(data: pd.DataFrame) -> pd.DataFrame:
    """Measure churn and charge share within the same tenure groups."""
    edges = [*range(0, 72, 6), 73]
    labels = [f"{start}–{end - 1}" for start, end in zip(edges[:-1], edges[1:], strict=True)]
    churned = data[TARGET].eq("Yes")
    charges = data["MonthlyCharges"]
    groups = (
        data.assign(
            tenure_band=pd.cut(data["tenure"], edges, labels=labels, right=False),
            churned=churned,
            churned_charges=charges.where(churned, 0),
        )
        .groupby("tenure_band", observed=False)
        .agg(
            customers=(TARGET, "size"),
            churn_rate=("churned", "mean"),
            charges=("MonthlyCharges", "sum"),
            churned_charges=("churned_charges", "sum"),
        )
    )
    charge_total = groups["charges"].where(groups["charges"].gt(0))
    groups["charge_share"] = groups["churned_charges"].div(charge_total)
    return groups


def tenure_churn_chart(groups: pd.DataFrame) -> None:
    labels = groups.index.tolist()
    figure = go.Figure()
    for column, name, color, dash, symbol in [
        ("churn_rate", "Customer churn rate", TEAL, "solid", "circle"),
        ("charge_share", "Monthly charge share", BLUE, "dash", "diamond"),
    ]:
        figure.add_scatter(
            x=labels, y=groups[column], name=name, mode="lines+markers",
            line={"color": color, "dash": dash, "width": 3},
            marker={"size": 7, "symbol": symbol, "line": {"color": "#F8FAF9", "width": 1}},
            hovertemplate=f"%{{y:.1%}}<extra>{name}</extra>",
        )
    ticks = [labels[index] for index in (0, 2, 4, 6, 8, 11)]
    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(title="Customer tenure (months)", tickmode="array", tickvals=ticks, griddash="dot")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero", griddash="dot")
    chart(figure, height=335, legend=True, legend_below=True)


def outcome_donut(churned: pd.Series) -> None:
    """Plot recorded outcomes and give the counts in an accessible text legend."""
    total = len(churned)
    lost = int(churned.sum())
    counts = {"Retained": total - lost, "Churned": lost}
    figure = go.Figure(go.Pie(
        labels=list(counts), values=list(counts.values()), hole=0.73,
        sort=False, direction="counterclockwise", textinfo="none",
        marker={"colors": [OUTCOME_COLORS[label] for label in counts], "line": {"color": "#F8FAF9", "width": 2}},
        hovertemplate="%{label}<br>%{value:,} customers · %{percent:.1%}<extra></extra>",
    ))
    figure.add_annotation(
        x=0.5, y=0.55, text=f"<b>{lost / total:.1%}</b>", showarrow=False,
        font={"size": 28, "color": NAVY},
    )
    figure.add_annotation(
        x=0.5, y=0.42, text="Churned", showarrow=False, font={"size": 13, "color": MUTED},
    )
    chart(figure, height=185)

    rows = "".join(
        f'<div style="--outcome-color:{OUTCOME_COLORS[label]}"><dt>{label}</dt>'
        f'<dd>{count:,}</dd><dd>{count / total:.1%}</dd></div>'
        for label, count in counts.items()
    )
    st.html(f'<dl class="churn-outcomes" aria-label="Customer outcomes">{rows}</dl>')


def churn_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.assign(churned=data[TARGET].eq("Yes"))
        .groupby(column, as_index=False)["churned"]
        .agg(customers="size", rate="mean")
        .sort_values("rate")
    )


def churn_rate_chart(
    data: pd.DataFrame, column: str, *, max_rate: float = 1.0, height: int = 300,
) -> pd.DataFrame:
    rates = churn_by(data, column)
    figure = go.Figure(go.Bar(
        x=rates["rate"], y=rates[column].astype(str), orientation="h",
        marker_color=TEAL, text=rates["rate"].map(lambda rate: f"{rate:.1%}"),
        textposition="outside", cliponaxis=False, customdata=rates["customers"],
        hovertemplate="%{y}<br>%{x:.1%} churn · %{customdata:,} customers<extra></extra>",
    ))
    figure.update_xaxes(title="Recorded churn rate", tickformat=".0%", range=[0, max_rate])
    figure.update_yaxes(type="category")
    figure.update_layout(bargap=0.42)
    chart(figure, height=height)
    return rates
