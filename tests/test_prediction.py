import pandas as pd
import pytest

from src.artifacts import load_release
from src.explain import explain_case
from src.predict import example_csv, export_csv, parse_batch, predict_batch, predict_one, risk_band
from src.schema import FEATURES


def test_packaged_model_matches_declared_features():
    bundle, metrics = load_release()
    assert tuple(bundle["pipeline"].feature_names_in_) == FEATURES
    assert tuple(metrics["features"]) == FEATURES
    assert bundle["threshold"] == metrics["threshold"]


def test_model_rejects_mixed_or_corrupt_files(tmp_path):
    from src.artifacts import METRICS_FILE, MODEL_DIR, PIPELINE_FILE

    (tmp_path / METRICS_FILE).write_bytes((MODEL_DIR / METRICS_FILE).read_bytes())
    (tmp_path / PIPELINE_FILE).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_release(tmp_path)


def test_single_and_batch_match_and_preserve_order_and_id():
    bundle, _ = load_release()
    frame = parse_batch(example_csv())
    frame.loc[0, "customerID"] = "001"
    result = predict_batch(bundle, frame)
    single = predict_one(bundle, frame.iloc[0].to_dict())
    assert list(result.columns) == ["customerID", "predicted_class", "churn_probability"]
    assert result["customerID"].tolist() == frame["customerID"].tolist()
    assert result.iloc[0]["predicted_class"] == single.predicted_class
    assert result.iloc[0]["churn_probability"] == pytest.approx(single.probability)
    assert result["churn_probability"].between(0, 1).all()


@pytest.mark.parametrize("header", [b"a,a\n1,2\n", b"a,b\n1\n"])
def test_bad_csv_reports_repeated_headers_or_incorrect_field_count(header):
    with pytest.raises(ValueError, match="Repeated|fields"):
        parse_batch(header)


def test_blank_fields_allowed_but_bad_numeric_and_service_values_rejected():
    bundle, _ = load_release()
    frame = parse_batch(example_csv())
    frame.loc[0, "TotalCharges"] = ""
    assert len(predict_batch(bundle, frame)) == len(frame)
    frame.loc[0, "TotalCharges"] = "n/a"
    with pytest.raises(ValueError, match="TotalCharges"):
        predict_batch(bundle, frame)
    frame.loc[0, "TotalCharges"] = ""
    frame.loc[0, "InternetService"] = "No"
    with pytest.raises(ValueError, match="consistent with InternetService"):
        predict_batch(bundle, frame)


def test_export_escapes_formula_like_identifiers_and_preserves_normal_ones():
    result = pd.DataFrame({"customerID": ["001", "=1+1"], "predicted_class": ["Yes", "No"]})
    assert "001,Yes" in export_csv(result).decode()
    assert "'=1+1,No" in export_csv(result).decode()


def test_risk_boundaries_are_inclusive():
    bundle = {"risk_low_cutoff": 0.2, "threshold": 0.4}
    assert [risk_band(p, bundle) for p in (0.19, 0.20, 0.39, 0.40)] == [
        "Low", "Medium", "Medium", "High",
    ]


def test_case_explanation_matches_actual_edited_model_response():
    bundle, _ = load_release()
    profile = parse_batch(example_csv()).iloc[0].to_dict()
    baseline = predict_one(bundle, profile).probability
    explanations = explain_case(bundle, profile)
    assert explanations
    for explanation in explanations:
        modified = {**profile, **explanation["changes"]}
        edited = predict_one(bundle, modified).probability
        assert explanation["difference"] == pytest.approx(edited - baseline)
