"""
Simple web UI for the employee turnover prediction model.

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from turnover_predictor import predict_turnover

DEPARTMENTS = [
    "sales", "accounting", "hr", "technical", "support",
    "management", "IT", "product_mng", "marketing", "RandD",
]
SALARY_LEVELS = ["low", "medium", "high"]

ZONE_COLORS = {
    "Safe Zone (Green)": "#2e7d32",
    "Low-Risk Zone (Yellow)": "#f9a825",
    "Medium-Risk Zone (Orange)": "#e67e22",
    "High-Risk Zone (Red)": "#c0392b",
}

st.set_page_config(page_title="Employee Turnover Predictor", page_icon="📊", layout="centered")
st.title("📊 Employee Turnover Risk Predictor")
st.caption("Predicts the probability that an employee will leave, using a trained Random Forest model.")

tab_single, tab_batch = st.tabs(["🧑 Check One Employee", "📁 Upload a CSV (Multiple Employees)"])

# ---------------------------------------------------------------------------
# TAB 1: single employee, filled in via a form
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter employee details")

    col1, col2 = st.columns(2)
    with col1:
        satisfaction_level = st.slider("Satisfaction level", 0.0, 1.0, 0.5, 0.01)
        last_evaluation = st.slider("Last evaluation score", 0.0, 1.0, 0.7, 0.01)
        number_project = st.number_input("Number of projects", min_value=1, max_value=10, value=4)
        average_montly_hours = st.number_input("Average monthly hours", min_value=50, max_value=350, value=200)
        time_spend_company = st.number_input("Years at the company", min_value=1, max_value=20, value=3)
    with col2:
        work_accident = st.selectbox("Had a workplace accident?", ["No", "Yes"])
        promotion_last_5years = st.selectbox("Promoted in the last 5 years?", ["No", "Yes"])
        sales = st.selectbox("Department", DEPARTMENTS)
        salary = st.selectbox("Salary level", SALARY_LEVELS)

    if st.button("Predict turnover risk", type="primary"):
        employee = pd.DataFrame([{
            "satisfaction_level": satisfaction_level,
            "last_evaluation": last_evaluation,
            "number_project": number_project,
            "average_montly_hours": average_montly_hours,
            "time_spend_company": time_spend_company,
            "Work_accident": 1 if work_accident == "Yes" else 0,
            "promotion_last_5years": 1 if promotion_last_5years == "Yes" else 0,
            "sales": sales,
            "salary": salary,
        }])

        result = predict_turnover(employee).iloc[0]
        prob = result["turnover_probability"]
        zone = result["risk_zone"]
        color = ZONE_COLORS[zone]

        st.markdown("### Result")
        st.metric("Probability of leaving", f"{prob:.1%}")
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background-color:{color};"
            f"color:white;text-align:center;font-weight:bold;'>{zone}</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))

# ---------------------------------------------------------------------------
# TAB 2: batch upload of a CSV
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV of employees")
    st.write(
        "The file must have these columns (same as `HR_comma_sep.csv`, without `left`): "
        "`satisfaction_level, last_evaluation, number_project, average_montly_hours, "
        "time_spend_company, Work_accident, promotion_last_5years, sales, salary`"
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        employees = pd.read_csv(uploaded)
        result = predict_turnover(employees)

        st.success(f"Scored {len(result)} employees.")
        st.dataframe(result, use_container_width=True)

        st.bar_chart(result["risk_zone"].value_counts())

        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="turnover_predictions.csv",
            mime="text/csv",
        )
