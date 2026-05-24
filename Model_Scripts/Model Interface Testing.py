import os
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def audit_loan_application(applicant_data, model_name="Fair-EBM"):
    """
    Automated credit risk support and audit system using fairness-aware models.
    """
    pkg_path = os.path.normpath(os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.',
                                             'New/loan_model_package.joblib'))
    if not os.path.exists(pkg_path):
        pkg_path = 'loan_model_package.joblib'
        if not os.path.exists(pkg_path):
            return "Error: model package not found."

    pkg = joblib.load(pkg_path)
    base_model_name = model_name.replace("Fair-", "")
    model = pkg['models'][base_model_name]
    features = pkg['features']
    th_doc, th_rest = pkg['thresholds'][base_model_name]

    # Structuring and alignment of columns to prevent standardization misalignments
    df_in = pd.DataFrame([applicant_data])
    for col in features:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[features]

    # Categorical alignment using the pre-fitted LabelEncoders natively
    for col, le in pkg['le_dict'].items():
        if col in df_in.columns:
            raw_val = df_in[col].iloc[0]
            native_type = type(le.classes_[0])
            try:
                typed_val = native_type(raw_val)
                df_in[col] = le.transform([typed_val])[0] if typed_val in le.classes_ else \
                le.transform([le.classes_[0]])[0]
            except Exception:
                df_in[col] = le.transform([le.classes_[0]])[0]

    # Numeric enforcement to secure StandardScaler Z-score processing without NaNs
    for col in df_in.columns:
        df_in[col] = pd.to_numeric(df_in[col], errors='coerce').fillna(0)

    # Sensitive attribute extraction (S=1 for Doctorate group)
    is_privileged = 1 if applicant_data['person_education'] == 'Doctorate' else 0
    threshold = th_doc if is_privileged == 1 else th_rest

    # Feature standardization matrix mapping
    X_scaled = pkg['scaler'].transform(df_in)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # Forward model inference (closer to 100% means higher risk of default)
    if base_model_name == "GAM":
        low, high = pkg['gam_bounds']
        raw_prob = model.predict_mu(np.clip(X_scaled, low, high))[0]
    else:
        raw_prob = model.predict_proba(X_scaled)[0, 1]

    # Inverting risk score into an intuitive 'Financial Safety Confidence Rate'
    prob_safety = 1.0 - raw_prob

    # Risk decisioning policy matching the native Grid-Search optimization kernel
    prediction = 1 if raw_prob < threshold else 0
    decision = "APPROVED (FUNDING GRANTED)" if prediction == 1 else "REJECTED (CREDIT INTERCEPTED)"

    # XAI: Local Feature Attribution extraction with sign inversion alignment
    explain_log = []
    if base_model_name == "EBM":
        try:
            ebm_local = model.explain_local(X_scaled_df)
            case_data = ebm_local.data(0)

            business_scores = {name: -score for name, score in zip(case_data['names'], case_data['scores'])}

            if prediction == 1:
                # Approved cases: Highlight dominant drivers pushing toward class 1 (Funding)
                sorted_feats = sorted(business_scores.items(), key=lambda x: x[1], reverse=True)
                drivers = [f"{n}(+{s:.2f})" for n, s in sorted_feats if s > 0]
                explain_log.append(
                    f"Top Safety Drivers: {', '.join(drivers[:3]) if drivers else 'Broadly stable financial criteria'}")
            else:
                # Rejected cases: Highlight dominant risk factors pulling down safety scores
                sorted_feats = sorted(business_scores.items(), key=lambda x: x[1], reverse=False)
                risks = [f"{n}({s:.2f})" for n, s in sorted_feats if s < 0]
                explain_log.append(
                    f"Top Risk Penalties: {', '.join(risks[:3]) if risks else 'Multiple high-leverage traits over limit'}")
        except Exception as e:
            explain_log.append(f"Local attribution parsing failure: {str(e)}")

    # Construct the automated regulatory audit log stream
    explanation = ["  [XAI Local Explanation Framework]"]
    if explain_log:
        explanation.append(f"    - {explain_log[0]}")

    # Financial domain validation red-lines
    audit_trail = []
    if applicant_data['loan_percent_income'] > 0.19:
        audit_trail.append(f"High Leverage Alert: Debt-to-income ratio ({applicant_data['loan_percent_income']:.2%}) violates target tier-1 ceiling.")
    if applicant_data['credit_score'] < 601:
        audit_trail.append(f"Weak Credit Tier: Rating score ({applicant_data['credit_score']}) falls into the lower underperforming quartile.")
    if applicant_data['previous_loan_defaults_on_file'] == "Yes":
        audit_trail.append("Blacklist Profile Triggered: Historical credit default records detected on active file.")

    explanation.append("  [Compliance Review Protocol]")
    if audit_trail:
        for audit in audit_trail:
            explanation.append(f"    - {audit}")
    else:
        explanation.append("    - No critical risk thresholds or compliance red-lines breached.")

    if is_privileged:
        explanation.append(
            f"    - Fairness Engine: PhD demographic matched (S=1). Dynamic threshold optimization shifted to: {threshold:.4f}")
    else:
        explanation.append(
            f"    - Fairness Engine: Base cohort matched (S=0). Evaluated under standard baseline threshold: {threshold:.4f}")

    return {
        "result": decision,
        "prob": f"{prob_safety:.2%}",
        "th": f"{threshold:.4f}",
        "cohort": "PhD Privileged Cohort (S=1)" if is_privileged else "Regular Cohort Baseline (S=0)",
        "logs": "\n".join(explanation)
    }


# Retained your precise manual case sequencing and adjustments
synthetic_cases = [
    {"desc": "Scenario 1: High-Net-Worth Profile (Standard educational background/high credit score/low debt ratio/no defaults)", "data": {
        "person_age": 35, "person_gender": "male", "person_education": "Bachelor", "person_income": 150000,
        "person_emp_exp": 10, "person_home_ownership": "MORTGAGE", "loan_amnt": 8000, "loan_intent": "VENTURE",
        "loan_int_rate": 6.5, "loan_percent_income": 0.05, "cb_person_cred_hist_length": 8, "credit_score": 810,
        "previous_loan_defaults_on_file": "No"}},

    {"desc": "Scenario 2: Over-Leveraged High DTI Case (No previous default, but loan size hits 85% of annual income)", "data": {
        "person_age": 24, "person_gender": "male", "person_education": "Master", "person_income": 20000,
        "person_emp_exp": 1, "person_home_ownership": "RENT", "loan_amnt": 17000, "loan_intent": "PERSONAL",
        "loan_int_rate": 14.5, "loan_percent_income": 0.85, "cb_person_cred_hist_length": 2, "credit_score": 610,
        "previous_loan_defaults_on_file": "No"}},

    {"desc": "Scenario 3: Solid Middle Class Applicant (Master Degree / High Job Tenure / Moderate FICO / Very Low DTI)", "data": {
        "person_age": 38, "person_gender": "female", "person_education": "Master", "person_income": 95000,
        "person_emp_exp": 12, "person_home_ownership": "OWN", "loan_amnt": 9500, "loan_intent": "HOMEIMPROVEMENT",
        "loan_int_rate": 8.5, "loan_percent_income": 0.10, "cb_person_cred_hist_length": 11, "credit_score": 740,
        "previous_loan_defaults_on_file": "No"}},

    {"desc": "Scenario 4: Toxic Blacklisted Default Case (Subprime Income Tier / Extreme Leverage / FICO Tier at Bottom Bound)", "data": {
        "person_age": 22, "person_gender": "female", "person_education": "High School", "person_income": 22000,
        "person_emp_exp": 0, "person_home_ownership": "RENT", "loan_amnt": 16000, "loan_intent": "MEDICAL",
        "loan_int_rate": 16.5, "loan_percent_income": 0.75, "cb_person_cred_hist_length": 1, "credit_score": 415,
        "previous_loan_defaults_on_file": "Yes"}},

    {"desc": "Scenario 5: Healthy Early-Career Applicant (Regular Bachelor Group / Steady Income Entry / Strong FICO / Low Leverage)", "data": {
        "person_age": 25, "person_gender": "male", "person_education": "Bachelor", "person_income": 65000,
        "person_emp_exp": 3, "person_home_ownership": "RENT", "loan_amnt": 4500, "loan_intent": "EDUCATION",
        "loan_int_rate": 9.2, "loan_percent_income": 0.07, "cb_person_cred_hist_length": 3, "credit_score": 725,
        "previous_loan_defaults_on_file": "No"}},
]

if __name__ == "__main__":
    target_algorithm = "Fair-EBM"
    print(f"\n Model Interface Testing ({target_algorithm})\n")

    for run in synthetic_cases:
        res = audit_loan_application(run['data'], model_name=target_algorithm)
        print(f"[{run['desc']}]")
        print(f"  Credit Recommendation : {res['result']}")
        print(f"  Safety Confidence     : {res['prob']} (Operational Shifting Threshold: {res['th']})")
        print(f"  Demographic Matching  : {res['cohort']}")
        print(f"{res['logs']}")
        print("-" * 95 + "\n")
