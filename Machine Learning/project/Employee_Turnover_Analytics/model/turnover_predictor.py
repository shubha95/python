"""
Reusable module for predicting employee turnover risk.

Usage:
    from turnover_predictor import predict_turnover
    import pandas as pd

    employees = pd.read_csv("new_employees.csv")   # same raw columns as HR_comma_sep.csv, minus 'left'
    result = predict_turnover(employees)
    print(result)
"""

import os
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "turnover_model.joblib")
COLUMNS_PATH = os.path.join(HERE, "feature_columns.joblib")

_model = None
_feature_columns = None


def _load():
    global _model, _feature_columns
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run train_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
        _feature_columns = joblib.load(COLUMNS_PATH)
    return _model, _feature_columns


def _risk_zone(p: float) -> str:
    if p < 0.20:
        return "Safe Zone (Green)"
    elif p < 0.60:
        return "Low-Risk Zone (Yellow)"
    elif p < 0.90:
        return "Medium-Risk Zone (Orange)"
    else:
        return "High-Risk Zone (Red)"


def predict_turnover(employees: pd.DataFrame) -> pd.DataFrame:
    """
    employees: raw DataFrame with the same columns as HR_comma_sep.csv
               (satisfaction_level, last_evaluation, number_project,
                average_montly_hours, time_spend_company, Work_accident,
                promotion_last_5years, sales, salary) -- 'left' not required.

    Returns a DataFrame with the original rows plus:
        turnover_probability : float, model's predicted probability of leaving
        risk_zone            : one of Safe/Low-Risk/Medium-Risk/High-Risk Zone
    """
    model, feature_columns = _load()

    df = employees.copy()
    df = df.drop(columns=["left"], errors="ignore")

    categorical_vars = df[["sales", "salary"]]
    numeric_vars = df.drop(columns=["sales", "salary"])
    dummies = pd.get_dummies(categorical_vars, drop_first=True).astype(int)
    processed = pd.concat([numeric_vars, dummies], axis=1)

    # Align columns to exactly what the model was trained on
    # (adds any missing dummy columns as 0, drops unexpected ones, fixes order)
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    probabilities = model.predict_proba(processed)[:, 1]

    result = employees.copy()
    result["turnover_probability"] = probabilities
    result["risk_zone"] = [_risk_zone(p) for p in probabilities]
    return result
