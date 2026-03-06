import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
from dotenv import load_dotenv

# =============================
# Load ENV properly
# =============================
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# =============================
# OpenAI Client (NEW SDK)
# =============================
client = None
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    client = None

# =============================
# MODEL PATHS
# =============================
PHISHING_MODEL_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\best_phishing_model.joblib"
PHISHING_VECTORIZER_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\vectorizer_phishing.joblib"
THREAT_MODEL_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\random_forest_model.pkl"
PROTO_LE_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\proto_encoder.pkl"
SERVICE_LE_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\service_encoder.pkl"
STATE_LE_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\state_encoder.pkl"
LGBM_MODEL_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\lightgbm_email_model.txt"

INSIDER_FEATURES = 10002

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    m = {}
    try:
        m["phishing_model"] = joblib.load(PHISHING_MODEL_PATH)
        m["vectorizer"] = joblib.load(PHISHING_VECTORIZER_PATH)
    except:
        m["phishing_model"] = None
        m["vectorizer"] = None

    try:
        m["threat_model"] = joblib.load(THREAT_MODEL_PATH)
        m["proto_le"] = joblib.load(PROTO_LE_PATH)
        m["service_le"] = joblib.load(SERVICE_LE_PATH)
        m["state_le"] = joblib.load(STATE_LE_PATH)
    except:
        m["threat_model"] = None

    try:
        import lightgbm as lgb
        m["lgbm_model"] = lgb.Booster(model_file=LGBM_MODEL_PATH)
    except:
        m["lgbm_model"] = None

    return m

models = load_models()

# =============================
# HELPERS
# =============================
def generate_alert(t, c):
    return f"⚠️ {t} detected (Confidence: {c:.2f})"

def get_text_column(df):
    for col in ["email", "message", "body", "content", "subject", "text"]:
        if col in df.columns:
            return col
    obj = df.select_dtypes(include=["object"]).columns
    return obj[0] if len(obj) else None

def auto_detect_type(df):
    if df.select_dtypes(include=["object"]).shape[1] > 0:
        return "Phishing"
    if df.select_dtypes(include=[np.number]).shape[1] / max(1, df.shape[1]) > 0.7:
        return "Data Breach"
    return "Network"

# =============================
# LLM SUMMARY (SAFE)
# =============================
def llm_summary(threat, df):
    if client is None:
        return "LLM disabled (API key missing)."

    try:
        counts = df["Prediction"].value_counts().to_dict()
        prompt = f"Summarize {threat} detection results: {counts}. Suggest 2 actions."

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=150
        )
        return resp.output_text
    except Exception as e:
        return f"LLM error: {e}"

# =============================
# STREAMLIT UI
# =============================
st.set_page_config("Cyber Shield AI v2", "🛡️", layout="wide")
st.title("🛡️ Cyber Shield AI v2")

# Login
st.sidebar.header("Login")
if "auth" not in st.session_state:
    st.session_state.auth = False

u = st.sidebar.text_input("Username")
p = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    if u in ["admin", "user"] and p in ["admin123", "user123"]:
        st.session_state.auth = True
        st.sidebar.success("Logged in")
    else:
        st.sidebar.error("Invalid")

if not st.session_state.auth:
    st.stop()

# Upload
file = st.file_uploader("Upload CSV", type=["csv"])
if not file:
    st.stop()

df = pd.read_csv(file)
st.dataframe(df.head())

detected = auto_detect_type(df)
st.subheader(f"Detected Type: {detected}")

# =============================
# PHISHING
# =============================
if detected == "Phishing":
    col = get_text_column(df)
    if not col:
        st.error("No text column found")
    else:
        X = models["vectorizer"].transform(df[col].astype(str))
        probs = models["phishing_model"].predict_proba(X)[:, 1]
        df["Prediction"] = np.where(probs > 0.5, "Phishing", "Safe")
        st.success(generate_alert("Phishing", probs.mean()))
        st.dataframe(df.head())
        st.info(llm_summary("Phishing", df))

# =============================
# INSIDER THREAT
# =============================
elif detected == "Data Breach":
    if models.get("lgbm_model") is None:
        st.error("Insider Threat model not loaded.")
        st.stop()

    try:
        # Remove label column
        label_cols = ["threat_label", "label", "target"]
        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in label_cols if c in df.columns],
            errors="ignore"
        )

        if numeric_df.shape[1] == 0:
            st.error("No numeric features for Insider Threat detection.")
            st.stop()

        # Standardize
        X = numeric_df.values.astype(np.float32)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        X = (X - mean) / std

        # Match training dimension
        REQUIRED = 10002
        if X.shape[1] < REQUIRED:
            X = np.hstack([X, np.zeros((X.shape[0], REQUIRED - X.shape[1]))])
        elif X.shape[1] > REQUIRED:
            X = X[:, :REQUIRED]

        # Predict
        probs = models["lgbm_model"].predict(X)

        # Risk boost for demo clarity
        risk_boost = (
            (df.get("usb_insert_count", 0) > 2).astype(int) +
            (df.get("failed_login_attempts", 0) > 2).astype(int) +
            (df.get("access_outside_work_hours", 0) == 1).astype(int)
        ) * 0.2

        probs = np.clip(probs + risk_boost.values, 0, 1)

        THRESHOLD = 0.25
        df["Threat_Probability"] = np.round(probs, 4)
        df["Prediction"] = np.where(probs > THRESHOLD, "Insider Threat", "Safe")

        st.success(generate_alert("Data Breach", float(np.mean(probs))))
        st.dataframe(df.head(15))

        if use_llm:
            st.markdown("### 🧠 LLM Summary")
            st.info(generate_llm_summary("Insider Threat", df))

    except Exception as e:
        st.error(f"Insider Threat detection failed: {e}")



st.caption("LLM optional • App runs even without OpenAI")
