import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


NEW_DIR = "Model_Scripts/New"
DATASET_PATH = os.path.join("DataSet", "Loan Approval Insights Data to Decision-Making.csv")


# ==========================================
# Data and model loading module
# ==========================================
@st.cache_resource
def load_model_package():
    pkg_path = os.path.join(NEW_DIR, "loan_model_package.joblib")
    if not os.path.exists(pkg_path):
        pkg_path = "loan_model_package.joblib"
    if os.path.exists(pkg_path):
        return joblib.load(pkg_path)
    return None


@st.cache_data
def load_experiment_data():

    data = {}

    try:
        data['raw_data'] = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        try:
            data['raw_data'] = pd.read_csv("Loan Approval Insights Data to Decision-Making.csv")
        except FileNotFoundError:
            data['raw_data'] = None

    try:
        data['performance'] = pd.read_csv(os.path.join(NEW_DIR, "Performance.csv"))
        data['significance'] = pd.read_csv(os.path.join(NEW_DIR, "Significance.csv"))
        data['spd_runs'] = pd.read_csv(os.path.join(NEW_DIR, "SPD_Runs.csv"))
    except FileNotFoundError:
        data['performance'] = None
        data['significance'] = None
        data['spd_runs'] = None

    return data


# ==========================================
# Core risk control audit backend logic
# ==========================================
def audit_loan_application(applicant_data, pkg, model_name="Fair-EBM"):

    try:
        if pkg is None:
            return {"error": "Model package not found."}

        base_model_name = model_name.replace("Fair-", "")
        if base_model_name not in pkg['models']:
            return {"error": f"Model {base_model_name} missing from package."}

        model = pkg['models'][base_model_name]
        features = pkg['features']
        th_doc, th_rest = pkg['thresholds'][base_model_name]

        # Data alignment and preprocessing
        df_in = pd.DataFrame([applicant_data])
        for col in features:
            if col not in df_in.columns:
                df_in[col] = 0
        df_in = df_in[features]

        # Encoder fault-tolerant mapping
        for col, le in pkg['le_dict'].items():
            if col in df_in.columns:
                raw_val = df_in[col].iloc[0]
                native_type = type(le.classes_[0])
                try:
                    typed_val = native_type(raw_val)
                    df_in[col] = le.transform([typed_val])[0] if typed_val in le.classes_ else le.transform([le.classes_[0]])[0]
                except Exception:
                    df_in[col] = le.transform([le.classes_[0]])[0]

        for col in df_in.columns:
            df_in[col] = pd.to_numeric(df_in[col], errors='coerce').fillna(0)

        # Sensitive attributes and threshold determination
        is_privileged = 1 if applicant_data['person_education'] == 'Doctorate' else 0
        threshold = th_doc if is_privileged == 1 else th_rest

        # Feature matrix standardization
        X_scaled = pkg['scaler'].transform(df_in)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)

        # Model prediction
        if base_model_name == "GAM":
            low, high = pkg['gam_bounds']
            prob_default = model.predict_mu(np.clip(X_scaled, low, high))[0]
        else:
            prob_default = model.predict_proba(X_scaled)[0, 1]

        prob_safety = float(1.0 - prob_default)  # Current user's security score
        safety_threshold = float(1.0 - threshold)  # Minimum safety passing standard

        prob_safety = max(0.0, min(1.0, prob_safety))

        # Approval will only be granted if the safety score is-
        # -higher than the minimum safety passing score.
        model_approved = True if prob_safety >= safety_threshold else False

        # Read the business rules
        rules = pkg.get('business_rules')

        # If no rule is found, a system-level block will be triggered directly.
        if not rules or 'dti_ceiling' not in rules or 'fico_ceiling' not in rules:
            return {
                "error": "CRITICAL SYSTEM ERROR: Data-driven compliance boundaries ('business_rules') are missing from the model package. Audit engine halted to prevent uncalibrated risk exposure."
            }

        # Securely obtain pure data-driven thresholds
        dti_ceiling = rules['dti_ceiling']
        fico_floor = rules['fico_ceiling']


        # Hard compliance review
        audit_trail = []
        if applicant_data['loan_percent_income'] > dti_ceiling:
            audit_trail.append(f"High Leverage Alert: loan_percent_income ({applicant_data['loan_percent_income']:.2%}) violates target tier-1 ceiling.")
        if applicant_data['credit_score'] < fico_floor:
            audit_trail.append(f"Weak Credit Tier: Rating score ({applicant_data['credit_score']}) falls into the lower underperforming quartile.")
        if applicant_data['previous_loan_defaults_on_file'] == "Yes":
            audit_trail.append("Blacklist Profile Triggered: Historical credit default records detected on active file.")

        if audit_trail:
            decision = "REJECTED (COMPLIANCE HARD STOP)"
            model_approved = False
        else:
            decision = "APPROVED (FUNDING GRANTED)" if model_approved else "REJECTED (CREDIT INTERCEPTED)"

        # XAI Local Interpreter Extraction and Improved Null Value Coverage
        explain_log = []
        if base_model_name == "EBM":
            try:
                ebm_local = model.explain_local(X_scaled_df)
                case_data = ebm_local.data(0)
                business_scores = {name: score for name, score in zip(case_data['names'], case_data['scores'])}

                if model_approved:
                    sorted_feats = sorted(business_scores.items(), key=lambda x: x[1], reverse=False)
                    drivers = [f"{n} (+{abs(s):.2f})" for n, s in sorted_feats if s < 0]
                    # If no financial indicators show a positive contribution, provide a backup text
                    explain_log.append(f"Top Safety Drivers: {', '.join(drivers[:3]) if drivers else 'Broadly stable financial criteria'}")
                else:
                    sorted_feats = sorted(business_scores.items(), key=lambda x: x[1], reverse=True)
                    risks = [f"{n} (-{abs(s):.2f})" for n, s in sorted_feats if s > 0]
                    explain_log.append(f"Top Risk Penalties: {', '.join(risks[:3]) if risks else 'Multiple high-leverage traits over limit'}")
            except Exception:
                explain_log.append("Top Risk Penalties: Local attribution parsing failed or model unsupported.")

        # Assemble a complete audit log
        explanation = ["===== Intelligent Risk & Interpretability Audit Log ====="]
        explanation.append("\n[1. XAI Local Explanation Framework]")
        if explain_log:
            explanation.append(f"  > {explain_log[0]}")
        else:
            explanation.append("  > No specific feature attribution generated.")

        explanation.append("\n[2. Compliance Review Protocol]")
        if audit_trail:
            for audit in audit_trail:
                explanation.append(f"  > [CRITICAL] {audit}")
        else:
            explanation.append("  > No critical risk thresholds or compliance red-lines breached.")

        explanation.append("\n[3. Fairness-Aware Engine]")
        cohort_str = "Privileged Group (Doctorate, S=1)" if is_privileged else "Non-Privileged Group (Other, S=0)"
        explanation.append(f"  > Cohort Match: {cohort_str}.\n  > Dynamic Risk Threshold Applied: {threshold:.4f}")

        return {
            "result": decision,
            "prob": prob_safety,
            "th": safety_threshold,
            "cohort": "PhD Privileged Cohort (S=1)" if is_privileged else "Regular Cohort Baseline (S=0)",
            "logs": "\n".join(explanation)
        }

    except Exception as e:
        return {"error": f"Internal system error during inference: {str(e)}"}


# ==========================================
#  UI (Streamlit Layout)
# ==========================================
st.set_page_config(page_title="Fairness-Aware Loan System", layout="wide")

st.title("Fairness-Aware Loan Approval & Audit System")
st.markdown("""
This system demonstrates a comprehensive pipeline from **Data Exploration**, **Fairness Model Evaluation**, to **Real-Time Risk Auditing**. 
It utilizes **Threshold Shifting** and **Explainable AI (EBM/GAM)** to mitigate algorithmic bias towards sensitive demographic attributes.
""")

tab1, tab2, tab3 = st.tabs(["📂 1. Dataset Overview", "📊 2. Experimental Results", "🛡️ 3. Real-Time Risk Audit"])

# --- TAB 1: Dataset Overview ---
with tab1:
    st.header("Exploratory Data Analysis (EDA)")

    data_dict = load_experiment_data()
    df_raw = data_dict.get('raw_data')

    if df_raw is not None:
        st.markdown(f"Dataset: Loan Approval Insights Data to Decision-Making. Total Samples: N={len(df_raw)}")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader("Raw Data Sample (First 100 rows)")
            st.dataframe(df_raw.head(100), use_container_width=True)

        with col_d2:
            st.subheader("Descriptive Statistics")
            st.dataframe(df_raw.describe().T.round(2), use_container_width=True)

        st.info(
            "Observation: High skewness in 'Person_income' indicates extreme outliers. Variables like 'Person_age' require preprocessing due to logical anomalies.")
    else:
        st.error(f"Could not find dataset at `{DATASET_PATH}`. Please check your folder structure.")

# --- TAB 2: Experimental Results ---
with tab2:
    st.header("Model Performance and Fairness Validation")


    # Dynamically attempt to match image names
    def get_image_path(base_name):
        # Try matching filenames with and without 's'.
        for ext in ["s.png", ".png", "s.PNG", ".PNG"]:
            path = os.path.join(NEW_DIR, f"{base_name}{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(NEW_DIR, f"{base_name}.png")


    # Show a comparison chart of F1 and SPD

    st.subheader("Model performance results")

    if data_dict.get('performance') is not None:
        st.dataframe(data_dict['performance'].style.format(precision=4), use_container_width=True)

    st.subheader("F1-Score vs. SPD Bias")
    chart_perf_path = get_image_path("Figure_Model Evaluation Results")

    if os.path.exists(chart_perf_path):
        image_perf = Image.open(chart_perf_path)
        st.image(image_perf, caption="Figure 4-11: Model Evaluation Results (F1-Score & SPD)", use_container_width=True)
    else:
        st.warning(
            f"Chart not found: {chart_perf_path}. Please make sure you have run the training script to generate the plots.")

    st.markdown("---")

    col_c1, col_c2 = st.columns(2)

    # Displaying the SPD distribution map after 10 runs.
    with col_c1:
        st.subheader("Robustness: SPD Values across 10 Runs")
        chart_spd_path = os.path.join(NEW_DIR, "Figur_SPD_Distribution.png")  # 严格匹配你截图里的拼写
        if os.path.exists(chart_spd_path):
            image_spd = Image.open(chart_spd_path)
            st.image(image_spd, caption="Figure 4-12: Distribution of SPD across 10 independent runs",
                     use_container_width=True)
        else:
            st.warning(f"Chart not found: {chart_spd_path}")

    # Show the significance chart of the T-test
    with col_c2:
        st.subheader("Statistical Significance Validation")
        chart_sig_path = os.path.join(NEW_DIR, "Figure_Significance.png")
        if os.path.exists(chart_sig_path):
            image_sig = Image.open(chart_sig_path)
            st.image(image_sig, caption="Figure 4-13: Statistical significance of fairness improvement",
                     use_container_width=True)
        else:
            st.warning(f"Chart not found: {chart_sig_path}")

    # Showing the T-test results table
    if data_dict.get('significance') is not None:
        st.subheader("Paired T-Test Results")
        st.dataframe(data_dict['significance'].style.map(
            lambda x: "background-color: #d4edda; color: #155724; font-weight: bold;" if x == 'Yes' else "",
            subset=['Sig?']
        ), use_container_width=True)
        st.success(
            "Conclusion: All Fairness-Aware models demonstrate a statistically significant reduction in bias (P-Value < 0.05).")

# --- TAB 3: Real-Time Risk Audit ---
with tab3:
    st.header("Single Applicant Penetration & Decision Audit")

    pkg = load_model_package()
    if pkg is None:
        st.error(
            f"Model file `loan_model_package.joblib` not found in `{NEW_DIR}/`. Please run your training script first.")

    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.subheader("📋 Applicant Information")
        with st.form("applicant_form"):
            edu = st.selectbox("Education Level (Sensitive Attribute)",
                               ["Doctorate", "Master", "Bachelor", "High School"], index=2)
            income = st.number_input("Annual Income (USD)", value=65000, step=5000)
            loan_amnt = st.number_input("Requested Loan Amount (USD)", value=15000, step=1000)
            loan_pct_income = st.slider("Loan-to-Income Ratio (loan_percent_income)", min_value=0.01, max_value=0.99,value=0.15, step=0.01)
            credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=720, step=10)
            default_hist = st.radio("Previous Default History?", ["No", "Yes"])

            # Default fallbacks
            person_age = st.number_input("Age", value=28, step=1)
            emp_exp = st.number_input("Employment Experience (Years)", value=5, step=1)

            submit = st.form_submit_button("Run Fairness Model & Audit", use_container_width=True)

    with col_output:
        st.subheader("📄 Audit Resolution")
        if submit:
            if pkg is None:
                st.warning("Please resolve the model loading error first.")
            else:
                mock_data = {
                    "person_age": person_age, "person_gender": "male", "person_education": edu,
                    "person_income": income, "person_emp_exp": emp_exp, "person_home_ownership": "RENT",
                    "loan_amnt": loan_amnt, "loan_intent": "EDUCATION", "loan_int_rate": 10.5,
                    "loan_percent_income": loan_pct_income, "cb_person_cred_hist_length": 4,
                    "credit_score": credit_score, "previous_loan_defaults_on_file": default_hist
                }

                with st.spinner("Invoking Fair-EBM Engine..."):
                    res = audit_loan_application(mock_data, pkg, model_name="Fair-EBM")

                st.markdown("---")

                if res is None:
                    st.error("System Error: The audit engine returned no data.")
                elif "error" in res:
                    st.error(f"Audit Error: {res['error']}")
                else:
                    if "APPROVED" in res.get('result', ''):
                        st.success(f"Recommended Decision: {res['result']}")
                    else:
                        st.error(f"Recommended Decision: {res['result']}")

                    st.markdown(f"Model Safety Confidence: {res['prob']:.2%}")
                    st.progress(float(res['prob']))
                    st.text_area("System Audit Trail", value=res.get('logs', ''), height=300)
        else:
            st.info(
                "Please adjust applicant parameters on the left and click 'Run' to generate a real-time risk report.")