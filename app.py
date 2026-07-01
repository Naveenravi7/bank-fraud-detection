"""
Bank Fraud Detection — Streamlit App
=====================================
Loads the trained GradientBoosting pipeline (model1.pkl) and provides an
interactive UI to score a single transaction for fraud risk.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo (app.py, model1.pkl, requirements.txt) to GitHub.
    2. Go to https://share.streamlit.io -> "New app" -> pick the repo/branch.
    3. Set "Main file path" to app.py -> Deploy.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Overall background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    /* Headings */
    h1, h2, h3 { color: #f8fafc; }
    p, label, span { color: #cbd5e1; }

    /* Card container */
    .metric-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    .hero {
        background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(14,165,233,0.25);
    }
    .hero h1 { margin: 0; color: white; font-size: 2.1rem; }
    .hero p { margin-top: .4rem; color: #e0f2fe; font-size: 1.02rem; }

    .result-fraud {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        border: 1px solid #ef4444;
    }
    .result-safe {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        border: 1px solid #22c55e;
    }
    .result-fraud h2, .result-safe h2 { color: white; margin-bottom: .2rem;}
    .result-fraud p, .result-safe p { color: #f1f5f9; font-size: 1.05rem; }

    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1f2937;
    }

    div[data-testid="stForm"] {
        background: #1e293b;
        padding: 1.6rem;
        border-radius: 16px;
        border: 1px solid #334155;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #0284c7 100%);
        color: white;
    }

    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model1.pkl"


class CompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map old scikit-learn internal module names to the new ones
        if module == "closs":
            module = "sklearn._loss._loss"
        elif module == "sklearn._loss.link":
            module = "sklearn._loss.loss"
        elif module == "_loss":
            module = "sklearn._loss._loss"
            
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError as e:
            # Try searching in sklearn._loss or sklearn._loss.loss
            if "sklearn._loss" in module or module == "_loss":
                try:
                    import sklearn._loss.loss as loss_mod
                    return getattr(loss_mod, name)
                except AttributeError:
                    pass
                try:
                    import sklearn._loss._loss as closs_mod
                    return getattr(closs_mod, name)
                except AttributeError:
                    pass
            raise e


@st.cache_resource(show_spinner="Loading fraud detection model...")
def load_model(path: Path):
    # Compatibility shim: some scikit-learn versions renamed an internal
    # helper class used inside fitted ColumnTransformer objects. This keeps
    # the app working even if the deployment environment's sklearn version
    # differs slightly from the one the model was trained with.
    try:
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass

    with open(path, "rb") as f:
        return CompatibilityUnpickler(f).load()



model = load_model(MODEL_PATH)

# ----------------------------------------------------------------------------
# Schema (extracted from the trained pipeline's ColumnTransformer)
# ----------------------------------------------------------------------------
TRANSACTION_TYPES = ["ATM", "Online", "POS"]
MERCHANT_CATEGORIES = ["ATM", "Clothing", "Electronics", "Fuel", "Grocery", "Restaurant"]
LOCATIONS = [
    "Bangkok", "Dubai", "Faisalabad", "Islamabad", "Karachi",
    "Kuala Lumpur", "Lahore", "London", "Multan", "Singapore",
]
HOME_LOCATIONS = ["Faisalabad", "Islamabad", "Karachi", "Lahore", "Multan"]
CARD_TYPES = ["Credit", "Debit"]
YES_NO = ["No", "Yes"]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Bank Fraud Detection</h1>
        <p>Score a transaction in real time using a Gradient Boosting model
        trained on historical banking transaction data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — about / info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "This app uses a **Gradient Boosting Classifier** trained on a "
        "banking transaction dataset to flag potentially fraudulent "
        "transactions in real time."
    )
    st.markdown("### 🧠 Model")
    st.write("Algorithm: `GradientBoostingClassifier`")
    st.write("Pipeline: `StandardScaler + OneHotEncoder + GBM`")
    st.markdown("### 📊 How to use")
    st.write(
        "1. Fill in the transaction details on the right.\n"
        "2. Click **Analyze Transaction**.\n"
        "3. Review the fraud risk score and verdict."
    )
    st.markdown("---")
    st.caption("Built with Streamlit • scikit-learn")

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("transaction_form"):
    st.subheader("Transaction Details")

    tab1, tab2, tab3 = st.tabs(["💳 Transaction", "👤 Customer", "🚩 Risk Signals"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            transaction_amount = st.number_input(
                "Transaction Amount (in Million)", min_value=0.0, value=1.0, step=0.1
            )
            transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
        with c2:
            merchant_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
            card_type = st.selectbox("Card Type", CARD_TYPES)
        with c3:
            transaction_location = st.selectbox("Transaction Location", LOCATIONS)
            is_international = st.selectbox("International Transaction?", YES_NO)

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_home_location = st.selectbox("Customer Home Location", HOME_LOCATIONS)
            account_balance = st.number_input(
                "Account Balance (in Million)", min_value=0.0, value=5.0, step=0.1
            )
        with c2:
            avg_transaction_amount = st.number_input(
                "Avg Transaction Amount (in Million)", min_value=0.0, value=1.0, step=0.1
            )
            weekly_transaction_count = st.number_input(
                "Weekly Transaction Count", min_value=0, value=10, step=1
            )
        with c3:
            distance_from_home = st.number_input(
                "Distance From Home (km)", min_value=0.0, value=5.0, step=1.0
            )
            previous_fraud_count = st.number_input(
                "Previous Fraud Count", min_value=0, value=0, step=1
            )

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            max_transaction_24h = st.number_input(
                "Max Transaction Last 24h (in Million)", min_value=0.0, value=1.0, step=0.1
            )
        with c2:
            is_new_merchant = st.selectbox("New Merchant?", YES_NO)
        with c3:
            unusual_time = st.selectbox("Unusual Time Transaction?", YES_NO)

    submitted = st.form_submit_button("🔍 Analyze Transaction")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    high_value_flag = transaction_amount > 5.0

    input_df = pd.DataFrame(
        [
            {
                "Transaction_Amount (in Million)": transaction_amount,
                "Distance_From_Home": distance_from_home,
                "Account_Balance (in Million)": account_balance,
                "Weekly_Transaction_Count": weekly_transaction_count,
                "Avg_Transaction_Amount (in Million)": avg_transaction_amount,
                "Max_Transaction_Last_24h (in Million)": max_transaction_24h,
                "Previous_Fraud_Count": previous_fraud_count,
                "Transaction_Type": transaction_type,
                "Merchant_Category": merchant_category,
                "Transaction_Location": transaction_location,
                "Customer_Home_Location": customer_home_location,
                "Card_Type": card_type,
                "Is_International_Transaction": is_international,
                "Is_New_Merchant": is_new_merchant,
                "Unusual_Time_Transaction": unusual_time,
                "High_Value_Flag": high_value_flag,
            }
        ]
    )

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        fraud_probability = float(proba[1]) * 100
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown("---")
    result_col, gauge_col = st.columns([1, 1.3])

    with result_col:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-fraud">
                    <h2>⚠️ Fraud Detected</h2>
                    <p>This transaction shows a high likelihood of fraud.</p>
                    <h1 style="color:white; font-size:2.6rem;">{fraud_probability:.1f}%</h1>
                    <p>estimated fraud probability</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-safe">
                    <h2>✅ Transaction Looks Safe</h2>
                    <p>No strong indicators of fraud were found.</p>
                    <h1 style="color:white; font-size:2.6rem;">{fraud_probability:.1f}%</h1>
                    <p>estimated fraud probability</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""<div class="metric-card"><p>Verdict</p>
                <h3 style="color:{'#f87171' if prediction==1 else '#4ade80'}">
                {'FRAUD' if prediction==1 else 'NORMAL'}</h3></div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card"><p>Confidence</p>
                <h3 style="color:#38bdf8">{max(proba)*100:.1f}%</h3></div>""",
                unsafe_allow_html=True,
            )

    with gauge_col:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=fraud_probability,
                number={"suffix": "%", "font": {"color": "white", "size": 40}},
                title={"text": "Fraud Risk Score", "font": {"color": "#cbd5e1", "size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#0ea5e9"},
                    "steps": [
                        {"range": [0, 40], "color": "#064e3b"},
                        {"range": [40, 70], "color": "#78350f"},
                        {"range": [70, 100], "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": fraud_probability,
                    },
                },
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            height=320,
            margin=dict(l=20, r=20, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (if available on the underlying classifier)
    try:
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        imp_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(10)
        )
        with st.expander("📈 Top factors driving the model's decisions (global importance)"):
            fig2 = go.Figure(
                go.Bar(
                    x=imp_df["importance"][::-1],
                    y=imp_df["feature"][::-1],
                    orientation="h",
                    marker_color="#0ea5e9",
                )
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass

    with st.expander("🔎 Raw input sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

else:
    st.info("Fill in the transaction details above and click **Analyze Transaction** to get a fraud risk score.")