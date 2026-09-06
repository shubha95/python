========================================================================
 EMPLOYEE TURNOVER PREDICTION MODEL - README
========================================================================

This folder contains a trained machine learning model that predicts the
probability that an employee will leave the company, plus tools to use
it (command line, web UI, and sample test data).


------------------------------------------------------------------------
 FILES IN THIS FOLDER
------------------------------------------------------------------------

train_model.py
    Trains the Random Forest model on ../HR_comma_sep.csv and saves it.
    Run this only once, or again if the training data changes.

turnover_model.joblib
    The trained model itself (created by train_model.py). Do not edit.

feature_columns.joblib
    The exact list/order of columns the model expects (created by
    train_model.py). Do not edit.

turnover_predictor.py
    The reusable module. Exposes one function:
        predict_turnover(employees_df) -> DataFrame with
            turnover_probability  and  risk_zone  columns added.

run_predictions.py
    Example script: scores every employee in ../HR_comma_sep.csv and
    saves the results to predictions_output.csv.

make_sample_data.py
    Generates 25 random (synthetic, not from the training data)
    employees and saves them to sample_employees.xlsx, for testing.

sample_employees.xlsx
    Random test data created by make_sample_data.py.

predictions_output.csv
    Output produced by run_predictions.py (safe to delete/regenerate).

app.py
    A web UI (Streamlit) for non-technical users: a form to check one
    employee, or upload a CSV to score many at once.


------------------------------------------------------------------------
 ONE-TIME SETUP
------------------------------------------------------------------------

1. Open Terminal and go to the project's python folder:

    cd "/Users/shubhamkeshari/Desktop/project/python"

2. Activate the virtual environment:

    source .venv/bin/activate

3. Install required packages (only needed once):

    python3 -m pip install pandas scikit-learn imbalanced-learn joblib openpyxl streamlit

4. Go into the model folder:

    cd "Machine Learning/project/Employee_Turnover_Analytics/model"


------------------------------------------------------------------------
 HOW TO RUN THE MODEL
------------------------------------------------------------------------

Option A - Train the model (only needed once, or if the CSV data changes)
--------------------------------------------------------------------------
    python3 train_model.py

    This creates/updates turnover_model.joblib and feature_columns.joblib.
    You should already have these files, so you can skip this step
    unless you want to retrain.


Option B - Score employees from the command line
--------------------------------------------------------------------------
    python3 run_predictions.py

    Prints predictions for the sample data and saves the full results
    to predictions_output.csv in this same folder.


Option C - Create random test data and check the model works
--------------------------------------------------------------------------
    python3 make_sample_data.py

    This regenerates sample_employees.xlsx with new random employees.
    To score that file specifically, run:

    python3 -c "
    import pandas as pd
    from turnover_predictor import predict_turnover
    data = pd.read_excel('sample_employees.xlsx')
    result = predict_turnover(data)
    print(result[['turnover_probability','risk_zone']])
    "


Option D - Use the web app (recommended for normal / non-technical users)
--------------------------------------------------------------------------
    python3 -m streamlit run app.py

    This opens a webpage at http://localhost:8501 in your browser.

    - Tab "Check One Employee": fill in the form, click "Predict turnover
      risk", and see the probability + color-coded risk zone.

    - Tab "Upload a CSV": upload a CSV of many employees at once and
      download the scored results.

    To stop the app: go back to the Terminal window and press Ctrl+C.


------------------------------------------------------------------------
 RISK ZONES (what the output means)
------------------------------------------------------------------------

    Safe Zone (Green)          probability < 20%   -> no action needed
    Low-Risk Zone (Yellow)     20% - 60%            -> monitor
    Medium-Risk Zone (Orange)  60% - 90%             -> manager check-in
    High-Risk Zone (Red)       probability > 90%    -> urgent retention action


------------------------------------------------------------------------
 TROUBLESHOOTING
------------------------------------------------------------------------

"No such file or directory" when running cd
    Check where you currently are with: pwd
    Then use the full path shown at the top of this file if unsure.

"ModuleNotFoundError" for pandas / sklearn / streamlit / etc.
    Make sure the virtual environment is activated (see step 2 above),
    then re-run the pip install command from step 3.

"No trained model found" error from turnover_predictor.py
    Run: python3 train_model.py
    (this creates the missing turnover_model.joblib file)
