"""
Example script: score employees for turnover risk using the trained model.

Usage:
    python3 run_predictions.py
"""

import pandas as pd
from turnover_predictor import predict_turnover

# Using the existing dataset as a stand-in for "new employees" (drop the
# known 'left' label since in real use you wouldn't have it yet).
employees = pd.read_csv("../HR_comma_sep.csv").drop(columns=["left"])

result = predict_turnover(employees)

print(result[["satisfaction_level", "last_evaluation", "turnover_probability", "risk_zone"]].head(10))
print()
print(result["risk_zone"].value_counts())

result.to_csv("predictions_output.csv", index=False)
print("\nFull results saved to predictions_output.csv")
