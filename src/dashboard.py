"""Shared Streamlit data access and chart styling."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.artifacts import load_release
from src.data_loader import load_dataset
from src.schema import TARGET

NAVY = "#142840"
BLUE = "#196493"
TEAL = "#197C71"
MUTED = "#465D73"
GRID = "#E5EDF5"
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


def metric_cards(items: list[tuple[str, str]]) -> None:
    entries = "".join(
        f'<div class="churn-kpi"><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>'
        for label, value in items
    )
    st.html(f'<dl class="churn-kpis" aria-label="Key metrics">{entries}</dl>')


def chart(figure: go.Figure, *, height: int = 300, legend: bool = False) -> None:
    figure.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Arial, sans-serif", "color": NAVY, "size": 13},
        margin={"l": 8, "r": 12, "t": 14, "b": 8},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, fixedrange=True)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True, fixedrange=True)
    st.plotly_chart(figure, theme=None, width="stretch", config={"displayModeBar": False})


def churn_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.assign(churned=data[TARGET].eq("Yes"))
        .groupby(column, as_index=False)["churned"]
        .agg(customers="size", rate="mean")
        .sort_values("rate")
    )


def churn_rate_chart(data: pd.DataFrame, column: str, *, max_rate: float = 1.0) -> pd.DataFrame:
    rates = churn_by(data, column)
    figure = go.Figure(go.Bar(
        x=rates["rate"], y=rates[column].astype(str), orientation="h",
        marker_color=BLUE, text=rates["rate"].map(lambda rate: f"{rate:.1%}"),
        textposition="outside", cliponaxis=False, customdata=rates["customers"],
        hovertemplate="%{y}<br>%{x:.1%} churn · %{customdata:,} customers<extra></extra>",
    ))
    figure.update_xaxes(title="Share of customers who churned", tickformat=".0%", range=[0, max_rate])
    figure.update_yaxes(type="category")
    chart(figure)
    return rates
