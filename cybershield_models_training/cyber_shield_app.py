# cyber_shield_ai_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
from dotenv import load_dotenv

# Optional OpenAI client (newer SDK)
try:
    from openai import OpenAI
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

# -------------------------
# Config: update model paths
# -------------------------
PHISHING_MODEL_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\best_phishing_model.joblib"
PHISHING_VECTORIZER_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\vectorizer_phishing.joblib"

THREAT_MODEL_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\random_forest_model.pkl"
PROTO_LE_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\proto_encoder.pkl"
SERVICE_LE_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\service_encoder.pkl"
STATE_LE_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\state_encoder.pkl"

LGBM_MODEL_PATH = r"C:\Users\91636\Desktop\major_project\New folder\cyber\lightgbm_email_model.txt"

EXPECTED_FEATURES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
    'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
    'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
    'is_sm_ips_ports'
]

# -------------------------
# Load environment & LLM
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OPENAI_INSTALLED and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        client = None
        st.warning(f"OpenAI init error: {e}")
else:
    # no OpenAI installed or key
    client = None

# -------------------------
# Model loading (cached)
# -------------------------
@st.cache_resource
def load_models():
    models = {}
    # phishing
    try:
        models['phishing_model'] = joblib.load(PHISHING_MODEL_PATH)
        models['vectorizer'] = joblib.load(PHISHING_VECTORIZER_PATH)
    except Exception as e:
        models['phishing_model'] = None
        models['vectorizer'] = None
        st.info("Phishing model/vectorizer not loaded — check paths.")

    # network threat model + encoders
    try:
        models['threat_model'] = joblib.load(THREAT_MODEL_PATH)
        models['proto_le'] = joblib.load(PROTO_LE_PATH)
        models['service_le'] = joblib.load(SERVICE_LE_PATH)
        models['state_le'] = joblib.load(STATE_LE_PATH)
    except Exception as e:
        models['threat_model'] = None
        models['proto_le'] = None
        models['service_le'] = None
        models['state_le'] = None
        st.info("Network threat model/encoders not loaded — check paths.")

    # lightgbm
    try:
        import lightgbm as lgb
        models['lgbm_model'] = lgb.Booster(model_file=LGBM_MODEL_PATH)
    except Exception as e:
        models['lgbm_model'] = None
        st.info("LightGBM model not loaded — check path.")

    return models

models = load_models()

# -------------------------
# Helpers
# -------------------------
def generate_alert(threat_type, confidence):
    alerts = {
        "Phishing": ["⚠️ Suspicious email detected.", "🚨 Potential phishing — avoid links."],
        "Network": ["🌐 Unusual network activity.", "🚨 Network anomaly detected."],
        "Data Breach": ["🔐 Possible insider/data leak.", "⚠️ Abnormal data access pattern."]
    }
    return random.choice(alerts.get(threat_type, ["⚠️ Threat detected"])) + f" (Confidence: {confidence:.2f})"

def safe_label_encode(le, series):
    if le is None:
        return pd.Series([-1]*len(series), index=series.index)
    known = set(getattr(le, "classes_", []))
    def enc(x):
        try:
            return int(le.transform([x])[0]) if str(x) in known else -1
        except Exception:
            return -1
    return series.apply(enc)

def preprocess_network_input(df):
    df = df.copy()
    df['proto'] = safe_label_encode(models.get('proto_le'), df['proto'].astype(str))
    df['service'] = safe_label_encode(models.get('service_le'), df['service'].astype(str))
    df['state'] = safe_label_encode(models.get('state_le'), df['state'].astype(str))
    return df

# Auto-detect heuristics
def auto_detect_type(df):
    """
    Return: ('Phishing'|'Network'|'Data Breach'|None, reason_text)
    """
    cols_lower = [c.lower() for c in df.columns]
    # quick checks for email-like
    email_like = any(x in cols_lower for x in ['email', 'subject', 'body', 'message', 'from', 'to'])
    # count overlap with expected network features
    overlap = sum(1 for f in EXPECTED_FEATURES if f in df.columns)
    # numeric ratio
    num_cols = df.select_dtypes(include=[np.number]).shape[1]
    total_cols = df.shape[1] or 1
    numeric_ratio = num_cols / total_cols

    if email_like:
        return "Phishing", "Detected email-like columns (email/subject/body/etc.)"
    if overlap >= max(6, len(EXPECTED_FEATURES)//6):  # flexible threshold
        return "Network", f"Detected many network/flow features (overlap={overlap})"
    if numeric_ratio >= 0.6 and num_cols >= 3:
        return "Data Breach", f"Mostly numeric features (numeric_ratio={numeric_ratio:.2f})"
    # fallback to LLM if available
    if client is not None:
        try:
            label, reason = llm_classify_file_with_client(df)
            if label:
                return label, f"LLM classified as {label}: {reason}"
        except Exception:
            pass
    # else unknown
    return None, "Unable to determine automatically (try enabling LLM or check CSV format)"

# LLM-based file classifier (uses client if present)
def llm_classify_file_with_client(df, sample_rows=3, model_name="gpt-4o-mini"):
    """
    Returns (label, reason) or (None, error_message)
    """
    if client is None:
        return None, "LLM client not available"
    sample_csv = df.head(sample_rows).to_csv(index=False)
    cols = ", ".join(list(df.columns)[:40])
    prompt = f"""
You are a cybersecurity assistant. Classify the following CSV sample into exactly one of:
- Phishing
- Network
- Data Breach

Return JSON: {{ "label": "<one of the three>", "reason": "<1-2 sentence reason referencing columns or sample values>" }}

Columns: {cols}

Sample rows:
{sample_csv}
"""
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=200
        )
        text = resp.choices[0].message.content.strip()
        # try to parse JSON-like response
        import json
        parsed = json.loads(text)
        return parsed.get("label"), parsed.get("reason")
    except Exception as e:
        return None, f"LLM error: {e}"

# LLM summary explanation (one-paragraph)
def generate_llm_summary(threat_type, df, model_stats=None, model_name="gpt-4o-mini"):
    if client is None:
        return "⚠️ LLM not available — enable your OPENAI_API_KEY in .env"
    # build short summary of predictions
    try:
        if 'Prediction' in df.columns:
            counts = df['Prediction'].value_counts().to_dict()
        elif 'Phishing_Prob' in df.columns:
            counts = {"Phishing": int((df['Phishing_Prob'] >= 0.5).sum()), "Safe": int((df['Phishing_Prob'] < 0.5).sum())}
        elif 'Threat_Probability' in df.columns:
            counts = {"HighRisk": int((df['Threat_Probability'] > 0.5).sum()), "LowRisk": int((df['Threat_Probability'] <= 0.5).sum())}
        else:
            counts = {"rows": df.shape[0]}
        prompt = f"""
You are an expert cybersecurity analyst. Provide a concise (2-4 sentences) summary of these detection results for a {threat_type} run:
Summary counts: {counts}
If there are notable risks, suggest top 3 immediate actions.
Keep it concise and actionable.
"""
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":"You are a cybersecurity analyst."},
                      {"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM summary failed: {e}"

# Chat assistant helper (sidebar)
def chat_with_assistant(user_message, model_name="gpt-4o-mini"):
    if client is None:
        return "⚠️ LLM not available — cannot answer."
    # maintain conversation in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"system","content":"You are an expert cybersecurity assistant called Cyber Shield."}]
    st.session_state.chat_history.append({"role":"user","content":user_message})
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=st.session_state.chat_history,
            temperature=0.3,
            max_tokens=400
        )
        assistant_msg = resp.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role":"assistant","content":assistant_msg})
        return assistant_msg
    except Exception as e:
        return f"⚠️ Chat assistant error: {e}"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Cyber Shield AI v2", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyber Shield AI v2 — Auto-detection + LLM Chat Assistant")
st.markdown("Upload a CSV and the system will automatically detect which model to run and provide an LLM-backed summary. Use the chat assistant in the sidebar to ask follow-ups.")

# Sidebar: login + chat
st.sidebar.header("🔑 Login")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

username = st.sidebar.text_input("Username", value="")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    if username and password and username in {"admin","user"} and (password in {"admin123","user123"}):
        st.session_state.authenticated = True
        st.sidebar.success("Logged in")
    else:
        st.sidebar.error("Invalid credentials")

st.sidebar.markdown("---")
st.sidebar.header("💬 Chat Assistant")
chat_input = st.sidebar.text_input("Ask Cyber Shield:", "")
if st.sidebar.button("Send") and chat_input:
    answer = chat_with_assistant(chat_input)
    st.sidebar.markdown("**Assistant:**")
    st.sidebar.write(answer)

# Option: enable LLM classification/explanations
use_llm = st.sidebar.checkbox("Use LLM for classification & summaries", value=True)

# Main upload
if not st.session_state.get("authenticated", False):
    st.info("Please login (sidebar) to use the app.")
    st.stop()

uploaded = st.file_uploader("Upload CSV (email / network / user activity data)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin auto-detection and analysis.")
    st.stop()

# Read file
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.markdown("**Preview (first 5 rows)**")
st.dataframe(df.head(5))

# Auto-detect
detected_type, reason = auto_detect_type(df)
st.markdown(f"**Auto-detected Type:** {detected_type or 'Unknown'}")
st.caption(f"Reason: {reason}")

if detected_type is None:
    st.error("Could not auto-detect dataset type. Try enabling LLM or ensure CSV contains recognizable columns.")
    st.stop()

# Route to corresponding model
if detected_type == "Phishing":
    if models.get('phishing_model') is None or models.get('vectorizer') is None:
        st.error("Phishing model/vectorizer not loaded on server.")
    else:
        try:
            X = models['vectorizer'].transform(df['email'].astype(str))
            probs = models['phishing_model'].predict_proba(X)[:, 1]
            df['Phishing_Prob'] = np.round(probs, 4)
            df['Prediction'] = ["Phishing" if p >= 0.5 else "Safe" for p in probs]
            st.success(generate_alert("Phishing", float(np.mean(probs))))
            st.markdown("**Results (sample)**")
            st.dataframe(df.head(15))
            # LLM summary
            if use_llm:
                st.markdown("### 🧠 LLM Summary")
                st.info(generate_llm_summary("Phishing", df))
        except Exception as e:
            st.error(f"Failed to run phishing pipeline: {e}")

elif detected_type == "Network":
    if models.get('threat_model') is None:
        st.error("Network threat model not loaded on server.")
    else:
        # Ensure required features exist (allow extra columns)
        missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
        if len(missing) > 0 and len(missing) < len(EXPECTED_FEATURES):
            st.warning(f"Some expected network features missing (sample): {missing[:8]}{'...' if len(missing)>8 else ''}")
        if len(missing) == len(EXPECTED_FEATURES):
            st.error("No expected network features present; cannot run network model.")
        else:
            try:
                df_proc = preprocess_network_input(df)
                # Keep only expected_features that are present; fill missing with 0
                X = pd.DataFrame(index=df_proc.index)
                for feat in EXPECTED_FEATURES:
                    if feat in df_proc.columns:
                        X[feat] = df_proc[feat]
                    else:
                        X[feat] = 0
                preds = models['threat_model'].predict(X.values)
                # If preds numeric or label-encoded; create readable label
                df['Prediction'] = ["Attack" if int(p) == 1 else "Safe" for p in preds]
                st.success(generate_alert("Network", float(np.mean([float(p) for p in preds]))))
                st.markdown("**Results (sample)**")
                st.dataframe(df.head(15))
                if use_llm:
                    st.markdown("### 🧠 LLM Summary")
                    st.info(generate_llm_summary("Network", df))
            except Exception as e:
                st.error(f"Failed to run network model: {e}")

elif detected_type == "Data Breach":
    if models.get('lgbm_model') is None:
        st.error("Insider-threat LightGBM model not loaded.")
    else:
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                st.error("No numeric features found for Data Breach model.")
            else:
                # Align to model input: use all numeric columns as-is
                preds = models['lgbm_model'].predict(numeric_df.values)
                df['Threat_Probability'] = np.round(preds, 4)
                df['Prediction'] = ["Insider Threat" if p > 0.5 else "Safe" for p in preds]
                st.success(generate_alert("Data Breach", float(np.mean(preds))))
                st.markdown("**Results (sample)**")
                st.dataframe(df.head(15))
                if use_llm:
                    st.markdown("### 🧠 LLM Summary")
                    st.info(generate_llm_summary("Insider Threat", df))
        except Exception as e:
            st.error(f"Failed to run data breach model: {e}")

# Footer
st.markdown("---")
st.caption("Notes: Keep sensitive data anonymized. LLM usage consumes API credits. If LLM isn't working, ensure OPENAI_API_KEY is set in your .env and you have the 'openai' package installed.")

