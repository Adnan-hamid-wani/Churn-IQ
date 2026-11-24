"""Construct a leakage-safe preprocessing and classifier pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.schema import CATEGORIES, NUMERIC_FEATURES

SEED = 42
MODEL_GRIDS = {
    "Logistic regression": {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "Decision tree": {
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [15, 40],
    },
    "Random forest": {
        "model__max_depth": [6, 12, None],
        "model__min_samples_leaf": [4, 15],
    },
}


def build_pipeline(model_name: str) -> Pipeline:
    """Create an unfitted pipeline so each cross-validation fold fits its own transforms."""
    models = {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision tree": DecisionTreeClassifier(random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=120, random_state=SEED, n_jobs=1),
    }
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if model_name == "Logistic regression":
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORIES)),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", models[model_name])])
