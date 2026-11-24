"""Model inspection using held-out shuffling and training-based reference values."""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.evaluate import churn_probabilities
from src.preprocessing import SEED
from src.schema import INTERNET_ADDONS, normalize_features, validate_service_combinations

REFERENCE_GROUPS = {
    "Account history": ("tenure", "TotalCharges"),
    "Monthly charges": ("MonthlyCharges",),
    "Contract": ("Contract",),
    "Paperless billing": ("PaperlessBilling",),
    "Payment method": ("PaymentMethod",),
    "Phone setup": ("PhoneService", "MultipleLines"),
    "Internet setup": ("InternetService", *INTERNET_ADDONS),
}


def global_importance(pipeline: Pipeline, features: pd.DataFrame, labels: pd.Series) -> list[dict]:
    """Measure the drop in test average precision when each raw feature is shuffled."""
    result = permutation_importance(
        pipeline, features, labels, scoring="average_precision", n_repeats=8, random_state=SEED,
    )
    rows = [
        {"feature": column, "mean": float(mean), "std": float(std)}
        for column, mean, std in zip(
            features.columns, result.importances_mean, result.importances_std, strict=True,
        )
    ]
    return sorted(rows, key=lambda row: row["mean"], reverse=True)


def build_references(training: pd.DataFrame) -> dict[str, list[dict]]:
    """Store common service profiles and representative observed billing or tenure values."""
    references = {}
    for group, columns in REFERENCE_GROUPS.items():
        complete = training.loc[:, list(columns)].dropna()
        if group in {"Account history", "Monthly charges"}:
            ordered = complete.sort_values(list(columns), kind="stable")
            positions = [round((len(ordered) - 1) * fraction) for fraction in (0.5, 0.25, 0.75)]
            selected = ordered.iloc[positions].drop_duplicates()
        else:
            selected = complete.value_counts(sort=False).sort_values(
                ascending=False, kind="stable"
            ).head(3).reset_index().loc[:, list(columns)]
        references[group] = selected.to_dict("records")
    return references


def explain_case(bundle: dict, profile: dict) -> list[dict]:
    """Score one feasible profile edit per group and rank by absolute probability change."""
    original = normalize_features(pd.DataFrame([profile])).reset_index(drop=True)
    validate_service_combinations(original)
    edited_profiles = []
    explanations = []
    for group, candidates in bundle["references"].items():
        for candidate in candidates:
            changes = {
                column: value for column, value in candidate.items()
                if pd.isna(original.at[0, column]) or original.at[0, column] != value
            }
            if changes:
                edited_profiles.append({**original.iloc[0].to_dict(), **changes})
                explanations.append({"group": group, "changes": changes})
                break

    cases = pd.DataFrame([original.iloc[0].to_dict(), *edited_profiles])
    scores = churn_probabilities(bundle["pipeline"], cases)
    for explanation, score in zip(explanations, scores[1:], strict=True):
        explanation["difference"] = float(score - scores[0])
    return sorted(explanations, key=lambda row: abs(row["difference"]), reverse=True)
