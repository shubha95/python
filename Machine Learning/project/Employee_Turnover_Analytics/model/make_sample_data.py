"""
Generates a random sample of employees (not from the training data) and
saves them to an Excel file, for testing the turnover_predictor module.

Usage:
    python3 make_sample_data.py
"""

import numpy as np
import pandas as pd

DEPARTMENTS = [
    "sales", "accounting", "hr", "technical", "support",
    "management", "IT", "product_mng", "marketing", "RandD",
]
SALARY_LEVELS = ["low", "medium", "high"]

rng = np.random.default_rng(seed=42)
N = 25

sample = pd.DataFrame({
    "satisfaction_level": rng.uniform(0.05, 1.0, N).round(2),
    "last_evaluation": rng.uniform(0.35, 1.0, N).round(2),
    "number_project": rng.integers(2, 8, N),
    "average_montly_hours": rng.integers(95, 315, N),
    "time_spend_company": rng.integers(1, 11, N),
    "Work_accident": rng.integers(0, 2, N),
    "promotion_last_5years": rng.choice([0, 0, 0, 0, 1], N),  # promotions are rare
    "sales": rng.choice(DEPARTMENTS, N),
    "salary": rng.choice(SALARY_LEVELS, N),
})

sample.to_excel("sample_employees.xlsx", index=False)
print(f"Saved {N} random employees to sample_employees.xlsx")
print(sample.head())
