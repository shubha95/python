"""
Trains the Random Forest turnover-prediction model on HR_comma_sep.csv
and saves it (plus the feature column list) to disk with joblib.

Run this once (or whenever HR_comma_sep.csv is updated):
    python train_model.py
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "HR_comma_sep.csv")
MODEL_PATH = os.path.join(HERE, "turnover_model.joblib")
COLUMNS_PATH = os.path.join(HERE, "feature_columns.joblib")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Same preprocessing used in the notebook: one-hot encode sales/salary."""
    categorical_vars = data[["sales", "salary"]]
    numeric_vars = data.drop(columns=["sales", "salary"])
    categorical_dummies = pd.get_dummies(categorical_vars, drop_first=True).astype(int)
    return pd.concat([numeric_vars, categorical_dummies], axis=1)


def main():
    data = pd.read_csv(DATA_PATH)
    data_final = build_features(data)

    X = data_final.drop(columns=["left"])
    y = data_final["left"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    smote = SMOTE(random_state=123)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=123)
    model.fit(X_train_resampled, y_train_resampled)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)

    print(f"Model saved to:   {MODEL_PATH}")
    print(f"Columns saved to: {COLUMNS_PATH}")
    print(f"Test accuracy:    {model.score(X_test, y_test):.3f}")


if __name__ == "__main__":
    main()
