from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_churn_model.joblib"
CONFIG_PATH = ROOT / "models" / "final_model_config.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    background-color: #080b12;
}

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(92, 82, 220, 0.15),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(0, 200, 180, 0.08),
            transparent 25%
        ),
        #080b12;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #0d111b;
    border-right: 1px solid #202635;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #e9edf7 !important;
}

/* Main container */

.block-container {
    max-width: 1450px;
    padding-top: 35px;
    padding-bottom: 50px;
}


/* Hero */

.hero {
    padding: 42px;
    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(25, 31, 53, 0.98),
            rgba(12, 17, 29, 0.98)
        );

    border: 1px solid rgba(255,255,255,0.09);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.35);

    margin-bottom: 30px;
}

.hero-label {
    color: #8d9cff;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    color: #ffffff;
    font-size: 64px;
    font-weight: 900;
    letter-spacing: -3px;
    margin-top: 8px;
}

.hero-description {
    color: #aeb7ca;
    font-size: 17px;
    line-height: 1.6;
    max-width: 850px;
    margin-top: 10px;
}


/* Section */

.section-title {
    color: #ffffff;
    font-size: 25px;
    font-weight: 800;
    margin-top: 30px;
}

.section-description {
    color: #8792a9;
    font-size: 14px;
    margin-bottom: 18px;
}


/* KPI */

.kpi {
    background: rgba(17,22,34,0.90);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 22px;
    min-height: 125px;
}

.kpi-label {
    color: #858fa5;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}

.kpi-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 850;
    margin-top: 10px;
}

.kpi-description {
    color: #737e95;
    font-size: 12px;
    margin-top: 5px;
}


/* Risk */

.risk-box {
    background:
        linear-gradient(
            145deg,
            rgba(24,30,47,0.98),
            rgba(10,14,23,0.98)
        );

    border: 1px solid rgba(255,255,255,0.09);

    border-radius: 22px;

    padding: 35px;

    text-align: center;

    box-shadow:
        0 20px 55px rgba(0,0,0,0.25);
}

.risk-label {
    color: #858fa5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.risk-score {
    font-size: 62px;
    font-weight: 900;
    letter-spacing: -3px;
    margin: 8px 0;
}

.high {
    color: #ff6677;
}

.medium {
    color: #ffc857;
}

.low {
    color: #42d6a3;
}

.risk-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 50px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    color: #ffffff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}


/* Action cards */

.action {
    background: rgba(17,22,34,0.90);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 10px;
}

.action-title {
    color: #8d9cff;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.action-text {
    color: #e9edf7;
    font-size: 15px;
    margin-top: 8px;
}


/* Buttons */

.stButton > button {
    width: 100%;
    min-height: 52px;

    border-radius: 13px;

    background:
        linear-gradient(
            135deg,
            #5966f2,
            #795cff
        );

    color: white;

    border: none;

    font-size: 15px;
    font-weight: 800;

    box-shadow:
        0 12px 30px rgba(89,102,242,0.25);
}

.stButton > button:hover {
    transform: translateY(-2px);
}


/* Footer */

.footer {
    text-align: center;
    color: #566176;
    font-size: 12px;
    padding-top: 45px;
    padding-bottom: 20px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_threshold():

    threshold = 0.55

    if CONFIG_PATH.exists():

        try:

            with open(
                CONFIG_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                config = json.load(file)

            threshold = float(
                config.get(
                    "threshold",
                    0.55
                )
            )

        except Exception:
            pass

    return threshold


def get_tenure_group(tenure):

    if tenure <= 12:
        return "0-12"

    elif tenure <= 24:
        return "13-24"

    elif tenure <= 48:
        return "25-48"

    return "49-72"


def get_risk(probability, threshold):

    if probability >= threshold:

        return (
            "HIGH RISK",
            "high",
            "Immediate retention action recommended."
        )

    elif probability >= 0.40:

        return (
            "MEDIUM RISK",
            "medium",
            "Customer should be monitored."
        )

    else:

        return (
            "LOW RISK",
            "low",
            "Customer currently has relatively low churn risk."
        )


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "Model file not found.\n\n"
        "Please run:\n\n"
        "`python src/optimize_model.py`"
    )

    st.stop()


model = joblib.load(
    MODEL_PATH
)

threshold = load_threshold()


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-label">
            AI-POWERED CUSTOMER INTELLIGENCE
        </div>

        <div class="hero-title">
            ChurnIQ
        </div>

        <div class="hero-description">
            Predict customer churn risk, understand why the model
            made its prediction, and generate targeted retention
            strategies using machine learning and explainable AI.
        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 👤 Customer Profile"
)

st.sidebar.caption(
    "Configure the customer information below."
)


gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)


senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)


partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)


dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)


tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)


phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


multiple = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "Yes",
        "No",
        "No phone service"
    ]
)


internet = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


security = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


device = st.sidebar.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


support = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


paperless = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)


payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=1.0
)


total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=800.0,
    step=10.0
)


# ============================================================
# CREATE CUSTOMER DATA
# ============================================================

tenure_group = get_tenure_group(
    tenure
)


avg_monthly_spend = (
    total / tenure
    if tenure > 0
    else monthly
)


high_monthly_charge = int(
    monthly >= 70
)


customer = {

    "gender": gender,

    "SeniorCitizen": senior,

    "Partner": partner,

    "Dependents": dependents,

    "tenure": tenure,

    "PhoneService": phone,

    "MultipleLines": multiple,

    "InternetService": internet,

    "OnlineSecurity": security,

    "OnlineBackup": backup,

    "DeviceProtection": device,

    "TechSupport": support,

    "StreamingTV": tv,

    "StreamingMovies": movies,

    "Contract": contract,

    "PaperlessBilling": paperless,

    "PaymentMethod": payment,

    "MonthlyCharges": monthly,

    "TotalCharges": total,

    "AvgMonthlySpend": avg_monthly_spend,

    "TenureGroup": tenure_group,

    "HighMonthlyCharge": high_monthly_charge
}


# ============================================================
# START SCREEN
# ============================================================

st.html(
    """
    <div class="section-title">
        Risk Assessment
    </div>

    <div class="section-description">
        Run the machine-learning model to calculate this customer's
        probability of churn.
    </div>
    """
)


if st.button(
    "🚀  RUN AI CHURN ASSESSMENT"
):

    st.session_state[
        "prediction_complete"
    ] = True


# ============================================================
# BEFORE PREDICTION
# ============================================================

if not st.session_state.get(
    "prediction_complete",
    False
):

    col1, col2, col3 = st.columns(3)


    with col1:

        st.html(
            """
            <div class="kpi">

                <div class="kpi-label">
                    STEP 01
                </div>

                <div class="kpi-value">
                    Predict
                </div>

                <div class="kpi-description">
                    Calculate customer churn probability.
                </div>

            </div>
            """
        )


    with col2:

        st.html(
            """
            <div class="kpi">

                <div class="kpi-label">
                    STEP 02
                </div>

                <div class="kpi-value">
                    Explain
                </div>

                <div class="kpi-description">
                    Understand the strongest churn drivers.
                </div>

            </div>
            """
        )


    with col3:

        st.html(
            """
            <div class="kpi">

                <div class="kpi-label">
                    STEP 03
                </div>

                <div class="kpi-value">
                    Act
                </div>

                <div class="kpi-description">
                    Generate targeted retention actions.
                </div>

            </div>
            """
        )


    st.html(
        """
        <div class="footer">
            ChurnIQ · Python · SQL · Machine Learning · SHAP · Streamlit
        </div>
        """
    )

    st.stop()


# ============================================================
# MODEL PREDICTION
# ============================================================

X = pd.DataFrame(
    [customer]
)


probability = float(
    model.predict_proba(X)[0, 1]
)


risk, risk_class, risk_description = get_risk(
    probability,
    threshold
)


# ============================================================
# KPI SECTION
# ============================================================

st.html(
    """
    <div class="section-title">
        Prediction Overview
    </div>

    <div class="section-description">
        AI-generated assessment of the selected customer.
    </div>
    """
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.html(
        f"""
        <div class="kpi">

            <div class="kpi-label">
                Churn Probability
            </div>

            <div class="kpi-value">
                {probability:.1%}
            </div>

            <div class="kpi-description">
                Estimated likelihood of churn
            </div>

        </div>
        """
    )


with c2:

    st.html(
        f"""
        <div class="kpi">

            <div class="kpi-label">
                Risk Level
            </div>

            <div class="kpi-value">
                {risk}
            </div>

            <div class="kpi-description">
                {risk_description}
            </div>

        </div>
        """
    )


with c3:

    st.html(
        f"""
        <div class="kpi">

            <div class="kpi-label">
                Threshold
            </div>

            <div class="kpi-value">
                {threshold:.0%}
            </div>

            <div class="kpi-description">
                Optimized classification threshold
            </div>

        </div>
        """
    )


with c4:

    st.html(
        f"""
        <div class="kpi">

            <div class="kpi-label">
                Customer Tenure
            </div>

            <div class="kpi-value">
                {tenure} Months
            </div>

            <div class="kpi-description">
                Length of customer relationship
            </div>

        </div>
        """
    )


# ============================================================
# RISK SCORE
# ============================================================

st.html(
    """
    <div class="section-title">
        Customer Risk Score
    </div>
    """
)


risk_col, logic_col = st.columns(
    [1.2, 1]
)


with risk_col:

    st.html(
        f"""
        <div class="risk-box">

            <div class="risk-label">
                AI CHURN SCORE
            </div>

            <div class="risk-score {risk_class}">
                {probability:.1%}
            </div>

            <div class="risk-badge">
                {risk}
            </div>

        </div>
        """
    )

    st.write("")

    st.progress(
        probability
    )


with logic_col:

    st.html(
        """
        <div class="card">

            <div class="card-title">
                Decision Logic
            </div>

        </div>
        """
    )

    st.write(
        f"**Model probability:** {probability:.1%}"
    )

    st.write(
        f"**Decision threshold:** {threshold:.0%}"
    )


    if probability >= threshold:

        st.error(
            "HIGH PRIORITY — customer is above the churn threshold."
        )

    elif probability >= 0.40:

        st.warning(
            "MEDIUM PRIORITY — customer should be monitored."
        )

    else:

        st.success(
            "LOW PRIORITY — customer is currently below the risk threshold."
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

st.html(
    """
    <div class="section-title">
        🧠 Explainable AI
    </div>

    <div class="section-description">
        Understand which customer attributes influence the prediction.
    </div>
    """
)


try:

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "model"
    ]


    X_transformed = preprocessor.transform(
        X
    )


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    classifier_name = (
        classifier.__class__.__name__
    )


    # --------------------------------------------------------
    # TREE MODELS
    # --------------------------------------------------------

    if classifier_name in {

        "XGBClassifier",

        "RandomForestClassifier",

        "GradientBoostingClassifier",

        "HistGradientBoostingClassifier"

    }:

        explainer = shap.TreeExplainer(
            classifier
        )

        shap_values = explainer.shap_values(
            X_transformed
        )


        if isinstance(
            shap_values,
            list
        ):

            values = np.asarray(
                shap_values[-1]
            )[0]

        else:

            values = np.asarray(
                shap_values
            )[0]


    # --------------------------------------------------------
    # LINEAR MODELS
    # --------------------------------------------------------

    elif hasattr(
        classifier,
        "coef_"
    ):

        coefficients = np.asarray(
            classifier.coef_
        )[0]


        row = (

            X_transformed.toarray()[0]

            if hasattr(
                X_transformed,
                "toarray"
            )

            else np.asarray(
                X_transformed
            )[0]

        )


        values = coefficients * row


    else:

        raise ValueError(
            "Unsupported model for SHAP explanation."
        )


    explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "Impact": values
        }
    )


    explanation["Feature"] = (
        explanation["Feature"]
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
        .str.replace(
            "numeric__",
            "",
            regex=False
        )
        .str.replace(
            "_",
            " ",
            regex=False
        )
    )


    positive = (
        explanation[
            explanation["Impact"] > 0
        ]
        .sort_values(
            "Impact",
            ascending=False
        )
        .head(5)
    )


    negative = (
        explanation[
            explanation["Impact"] < 0
        ]
        .sort_values(
            "Impact"
        )
        .head(5)
    )


    explain1, explain2 = st.columns(2)


    with explain1:

        st.html(
            """
            <div class="action">

                <div class="action-title">
                    🔴 Increasing Churn Risk
                </div>

            </div>
            """
        )


        if positive.empty:

            st.info(
                "No strong positive drivers detected."
            )

        else:

            for _, row in positive.iterrows():

                st.write(
                    f"🔴 **{row['Feature']}**"
                )


    with explain2:

        st.html(
            """
            <div class="action">

                <div class="action-title">
                    🟢 Reducing Churn Risk
                </div>

            </div>
            """
        )


        if negative.empty:

            st.info(
                "No strong protective factors detected."
            )

        else:

            for _, row in negative.iterrows():

                st.write(
                    f"🟢 **{row['Feature']}**"
                )


except Exception as error:

    st.warning(
        "Prediction completed, but SHAP explanation could not be generated."
    )

    st.caption(
        str(error)
    )


# ============================================================
# RETENTION STRATEGY
# ============================================================

st.html(
    """
    <div class="section-title">
        🎯 Retention Action Plan
    </div>

    <div class="section-description">
        Recommended actions based on the customer's profile and risk signals.
    </div>
    """
)


actions = []


if contract == "Month-to-month":

    actions.append(
        (
            "Contract Conversion",
            "Offer an annual or two-year contract with a loyalty incentive."
        )
    )


if monthly >= 70:

    actions.append(
        (
            "Pricing Strategy",
            "Consider a personalized discount or loyalty offer."
        )
    )


if support == "No":

    actions.append(
        (
            "Technical Support",
            "Offer proactive technical assistance or a support trial."
        )
    )


if payment == "Electronic check":

    actions.append(
        (
            "Payment Optimization",
            "Encourage automatic payment with a small incentive."
        )
    )


if tenure <= 12:

    actions.append(
        (
            "Early Lifecycle",
            "Trigger an onboarding and early-stage retention campaign."
        )
    )


if internet == "Fiber optic":

    actions.append(
        (
            "Premium Service",
            "Review service quality and satisfaction for high-value users."
        )
    )


if not actions:

    actions.append(
        (
            "Customer Engagement",
            "Continue standard loyalty and engagement programs."
        )
    )


for title, description in actions:

    st.html(
        f"""
        <div class="action">

            <div class="action-title">
                {title}
            </div>

            <div class="action-text">
                {description}
            </div>

        </div>
        """
    )


# ============================================================
# CUSTOMER SNAPSHOT
# ============================================================

st.html(
    """
    <div class="section-title">
        👤 Customer Snapshot
    </div>

    <div class="section-description">
        Key attributes used by the prediction model.
    </div>
    """
)


snapshot = pd.DataFrame(
    {
        "Attribute": [

            "Gender",

            "Senior Citizen",

            "Partner",

            "Dependents",

            "Tenure",

            "Contract",

            "Internet Service",

            "Monthly Charges",

            "Total Charges",

            "Payment Method",

            "Technical Support",

            "Paperless Billing"

        ],

        "Value": [

            gender,

            senior,

            partner,

            dependents,

            f"{tenure} months",

            contract,

            internet,

            f"₹{monthly:,.2f}",

            f"₹{total:,.2f}",

            payment,

            support,

            paperless

        ]
    }
)


st.dataframe(
    snapshot,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <b>ChurnIQ</b>
        <br>

        Customer Churn Prediction & Retention Intelligence

        <br><br>

        Python · SQL · Machine Learning · XGBoost · SHAP · Streamlit

    </div>
    """
)