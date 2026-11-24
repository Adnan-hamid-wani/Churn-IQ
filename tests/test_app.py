import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import tenure_churn
from src.data_loader import ROOT


@pytest.mark.parametrize(("path", "title"), [
    (None, "Customer churn overview"),
    ("views/data_explorer.py", "Data Explorer"),
    ("views/predict_churn.py", "Predict Churn"),
    ("views/model_performance.py", "Model Performance"),
    ("views/batch_predictions.py", "Batch Predictions"),
])
def test_page_renders(path, title):
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()
    if path:
        app.switch_page(path).run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == title


def test_prediction_form_returns_probability_and_explanation():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()
    app.switch_page("views/predict_churn.py").run()
    next(button for button in app.button if button.label == "Calculate churn risk").click().run()
    assert not app.exception
    assert not app.error
    assert any(metric.label == "Churn probability" for metric in app.metric)
    assert any(heading.value == "What changed this score?" for heading in app.subheader)


def test_explorer_can_filter_to_monthly_contracts():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()
    app.switch_page("views/data_explorer.py").run()
    app.multiselect(key="contract_filter").set_value(["Month-to-month"]).run()
    assert not app.exception
    assert "3,875" in app.metric[0].value
    traces = json.loads(app.get("plotly_chart")[-1].proto.spec)["data"]
    assert traces[0]["type"] == "pie"
    assert traces[0]["labels"] == ["Retained", "Churned"]
    assert traces[0]["values"] == [2220, 1655]


def test_batch_example_scores_both_rows():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()
    app.switch_page("views/batch_predictions.py").run()
    next(button for button in app.button if button.label == "Try the example").click().run()
    assert not app.exception
    assert not app.error
    assert "2 customers scored" in app.success[0].value
    assert app.dataframe[0].value["customerID"].tolist() == ["DEMO-001", "DEMO-002"]



def test_tenure_chart_compares_customers_and_charges_within_each_group():
    sample = pd.DataFrame({
        "tenure": [0, 5, 6, 72],
        "MonthlyCharges": [90, 10, 20, 25],
        "Churn": ["Yes", "No", "No", "Yes"],
    })
    groups = tenure_churn(sample)
    assert groups.loc["0–5", "churn_rate"] == pytest.approx(0.5)
    assert groups.loc["0–5", "charge_share"] == pytest.approx(0.9)
    assert groups.loc["6–11", "charge_share"] == 0
    assert pd.isna(groups.loc["12–17", "charge_share"])
    assert groups.loc["66–72", "customers"] == 1
