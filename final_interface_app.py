# cyber_shield_ai_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
import time
from datetime import datetime
from dotenv import load_dotenv

# Optional OpenAI client
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
        st.warning(f"OpenAI init error: {e}")

# -------------------------
# Model loading (cached)
# -------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models['phishing_model'] = joblib.load(PHISHING_MODEL_PATH)
        models['vectorizer'] = joblib.load(PHISHING_VECTORIZER_PATH)
    except Exception:
        models['phishing_model'] = None
        models['vectorizer'] = None

    try:
        models['threat_model'] = joblib.load(THREAT_MODEL_PATH)
        models['proto_le'] = joblib.load(PROTO_LE_PATH)
        models['service_le'] = joblib.load(SERVICE_LE_PATH)
        models['state_le'] = joblib.load(STATE_LE_PATH)
    except Exception:
        models['threat_model'] = None
    
    try:
        import lightgbm as lgb
        models['lgbm_model'] = lgb.Booster(model_file=LGBM_MODEL_PATH)
    except Exception:
        models['lgbm_model'] = None

    return models

models = load_models()

# -------------------------
# State Management
# -------------------------
def init_state():
    if "devices" not in st.session_state:
        st.session_state.devices = [
            {"id": "d1", "name": "Workstation Alpha", "type": "desktop", "os": "Windows 11", "threat_status": "secure", "threat_count": 0, "last_scan": "2024-03-10 09:00:00"},
            {"id": "d2", "name": "MacBook Pro", "type": "laptop", "os": "macOS Sonoma", "threat_status": "threat", "threat_count": 2, "last_scan": "2024-03-10 10:30:00"},
            {"id": "d3", "name": "Server-01", "type": "server", "os": "Ubuntu 22.04", "threat_status": "warning", "threat_count": 1, "last_scan": "2024-03-09 18:45:00"},
        ]
    if "threats" not in st.session_state:
        st.session_state.threats = [
            {"id": "t1", "name": "Suspicious Outbound Traffic", "severity": "high", "status": "detected", "description": "Unusual data upload to unknown IP"},
            {"id": "t2", "name": "Failed Login Attempt", "severity": "medium", "status": "detected", "description": "Multiple failed attempts from 192.168.1.100"},
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are CyberShield AI, an expert cybersecurity assistant."}]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "overview"

init_state()

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
    return series.apply(lambda x: int(le.transform([x])[0]) if str(x) in known else -1)

def preprocess_network_input(df):
    df_c = df.copy()
    df_c['proto'] = safe_label_encode(models.get('proto_le'), df_c['proto'].astype(str))
    df_c['service'] = safe_label_encode(models.get('service_le'), df_c['service'].astype(str))
    df_c['state'] = safe_label_encode(models.get('state_le'), df_c['state'].astype(str))
    return df_c

def auto_detect_type(df):
    cols_lower = [c.lower() for c in df.columns]
    text_cols = ["email", "message", "body", "text", "subject"]
    has_text = any(c in cols_lower for c in text_cols)
    overlap = sum(1 for f in EXPECTED_FEATURES if f in df.columns)
    numeric_ratio = df.select_dtypes(include=[np.number]).shape[1] / (df.shape[1] or 1)

    if has_text: return "Phishing", "Detected email text column"
    if overlap >= 6: return "Network", f"Detected network features ({overlap})"
    if numeric_ratio >= 0.6: return "Data Breach", "Detected numeric behavioral features"
    return None, "Unable to classify dataset"

def generate_llm_summary(threat_type, df, model_name="gpt-4o-mini"):
    if client is None: return "⚠️ LLM not available"
    try:
        if 'Prediction' in df.columns:
            counts = df['Prediction'].value_counts().to_dict()
        else:
            counts = {"rows": df.shape[0]}
        prompt = f"Summarize results for {threat_type}. Counts: {counts}. Suggest 3 actions."
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2, max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {e}"

# -------------------------
# View Components
# -------------------------
def sidebar_nav():
    st.sidebar.title("🛡️ Cyber Shield")
    
    # Navigation
    menu = ["Overview", "Devices", "AI Assistant", "File Scanner"]
    choice = st.sidebar.radio("Navigation", menu)
    
    st.sidebar.markdown("---")
    
    # User Profile (Mock)
    st.sidebar.markdown("### 👤 User Profile")
    st.sidebar.text("admin@cybershield.ai")
    st.sidebar.caption("Status: Protected")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
        
    return choice

def view_overview():
    st.title("Security Overview")
    st.markdown("Real-time threat monitoring")
    
    # Metrics
    devs = st.session_state.devices
    threats = st.session_state.threats
    
    total = len(devs)
    secure = sum(1 for d in devs if d['threat_status'] == 'secure')
    at_risk = sum(1 for d in devs if d['threat_status'] in ['threat', 'warning'])
    active_threats = len([t for t in threats if t['status'] == 'detected'])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Devices", total, delta_color="off")
    c2.metric("Secure", secure, delta_color="normal") # Green usually
    c3.metric("At Risk", at_risk, delta_color="inverse") # Red usually
    c4.metric("Active Threats", active_threats, delta_color="inverse")
    
    st.markdown("---")
    
    # Two columns: Device Status & Recent Threats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Device Status")
        for d in devs[:5]:
            status_color = "🟢" if d['threat_status'] == 'secure' else "🔴" if d['threat_status'] == 'threat' else "🟡"
            st.markdown(f"**{status_color} {d['name']}**")
            st.caption(f"{d['type'].title()} • {d['os']}")
            
    with col2:
        st.subheader("Recent Threats")
        if not threats:
            st.success("No active threats.")
        for t in threats[:5]:
            severity_icon = "🔥" if t['severity'] == 'high' else "⚠️"
            st.warning(f"**{severity_icon} {t['name']}**\n\n{t['description']}")

def view_devices():
    st.title("Protected Devices")
    st.markdown("Manage and monitor your devices")
    
    # Add Device Form (Expander)
    with st.expander("➕ Add New Device"):
        with st.form("add_device_form"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name")
            dtype = c2.selectbox("Type", ["desktop", "laptop", "mobile", "server"])
            os_name = c3.text_input("OS (e.g. Windows)")
            submitted = st.form_submit_button("Add Device")
            if submitted and name:
                new_dev = {
                    "id": f"d{len(st.session_state.devices)+1}",
                    "name": name,
                    "type": dtype,
                    "os": os_name or "Unknown",
                    "threat_status": "secure",
                    "threat_count": 0,
                    "last_scan": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.devices.append(new_dev)
                st.success(f"Added {name}")
                st.rerun()

    # Device Grid
    if not st.session_state.devices:
        st.info("No devices found.")
    else:
        for dev in st.session_state.devices:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    icon = "🖥️" if dev['type']=='desktop' else "💻" if dev['type']=='laptop' else "📱" if dev['type']=='mobile' else "🗄️"
                    st.subheader(f"{icon} {dev['name']}")
                    st.caption(f"{dev['os']} • {dev['type']}")
                
                with c2:
                    status = dev['threat_status']
                    color = "green" if status == 'secure' else "red" if status == 'threat' else "orange"
                    st.markdown(f"Status: :{color}[{status.upper()}]")
                    if status != 'secure':
                        st.markdown(f"**{dev['threat_count']} Threats Detected**")
                    st.caption(f"Last Scan: {dev.get('last_scan', 'Never')}")
                
                with c3:
                    if st.button("Scan Now", key=f"scan_{dev['id']}"):
                        with st.spinner("Scanning..."):
                            time.sleep(1.5) # Simulate scan
                            # Randomly update status
                            new_status = random.choice(["secure", "warning", "threat"])
                            dev['threat_status'] = new_status
                            dev['threat_count'] = 0 if new_status == "secure" else random.randint(1, 5)
                            dev['last_scan'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.rerun()
                    
                    if st.button("Remove", key=f"del_{dev['id']}"):
                        st.session_state.devices = [d for d in st.session_state.devices if d['id'] != dev['id']]
                        st.rerun()

def view_chat():
    st.title("AI Security Assistant")
    st.markdown("Ask questions about your device security")
    
    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "system": continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Input
    if prompt := st.chat_input("Ask CyberShield..."):
        # User message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Assistant Response
        with st.chat_message("assistant"):
            if client:
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.chat_history,
                        temperature=0.3
                    )
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"Error: {e}"
            else:
                reply = "I am a demo bot. Configure OPENAI_API_KEY to enable real AI."
            
            st.write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

def view_file_scanner():
    st.title("File Scanner & Analysis")
    st.markdown("Upload CSV logs for automated threat analysis (Phishing, Network, Insider).")
    
    uploaded = st.file_uploader("Upload Log CSV", type=["csv"])
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            
            detected_type, reason = auto_detect_type(df)
            st.info(f"Detected Type: **{detected_type}** ({reason})")
            
            if st.button("Run Analysis"):
                if detected_type == "Phishing":
                    run_phishing_analysis(df)
                elif detected_type == "Network":
                    run_network_analysis(df)
                elif detected_type == "Data Breach":
                    run_insider_analysis(df)
                else:
                    st.error("Unknown data type.")
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")

# -------------------------
# Analysis Logic (Refactored from original)
# -------------------------
def run_phishing_analysis(df):
    if not models['phishing_model']: st.error("Model not loaded"); return
    
    # Find text col
    text_col = next((c for c in df.columns if c.lower() in ["email", "text", "body", "content"]), None)
    if not text_col:
        st.error("No text column found.")
        return
        
    X = models["vectorizer"].transform(df[text_col].astype(str))
    probs = models["phishing_model"].predict_proba(X)[:, 1]
    df["Phishing_Prob"] = np.round(probs, 4)
    df["Prediction"] = np.where(probs >= 0.5, "Phishing", "Safe")
    
    st.success(generate_alert("Phishing", float(np.mean(probs))))
    st.dataframe(df[["Prediction", "Phishing_Prob", text_col]].head(10))
    
    st.markdown("### AI Summary")
    st.write(generate_llm_summary("Phishing", df))

def run_network_analysis(df):
    if not models['threat_model']: st.error("Model not loaded"); return
    
    try:
        df_proc = preprocess_network_input(df)
        X = pd.DataFrame(0, index=df_proc.index, columns=EXPECTED_FEATURES)
        for f in EXPECTED_FEATURES:
            if f in df_proc.columns: X[f] = df_proc[f]
            
        preds = models['threat_model'].predict(X.values)
        df['Prediction'] = ["Attack" if p==1 else "Safe" for p in preds]
        
        st.success(generate_alert("Network", 0.9 if any(preds) else 0.1))
        st.dataframe(df.head(10))
        st.write(generate_llm_summary("Network", df))
    except Exception as e:
        st.error(f"Analysis failed: {e}")

def run_insider_analysis(df):
    if not models['lgbm_model']: st.error("Model not loaded"); return
    
    try:
        # Preprocessing: drop labels if present to avoid leakage (though unused)
        label_cols = ["threat_label", "label", "target"]
        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in label_cols if c in df.columns], 
            errors="ignore"
        )
        
        # Standardize features (heuristic boost expects this scale implicitly or robustly)
        X = numeric_df.values.astype(np.float32)
        
        REQUIRED = 10002 # Matches training shape
        if X.shape[1] < REQUIRED:
            X = np.hstack([X, np.zeros((X.shape[0], REQUIRED - X.shape[1]))])
        elif X.shape[1] > REQUIRED:
            X = X[:, :REQUIRED]
            
        probs = models['lgbm_model'].predict(X)
        
        # --- MISSING LOGIC (PORTED FROM AI_CYBER_APP) ---
        # Heuristic Risk Boost: Add probability for suspicious behavioral flags
        risk_boost = (
            (df.get("usb_insert_count", 0) > 2).astype(int) +
            (df.get("failed_login_attempts", 0) > 2).astype(int) +
            (df.get("access_outside_work_hours", 0) == 1).astype(int)
        ) * 0.2
        
        probs = np.clip(probs + risk_boost.values, 0, 1)
        # Lower threshold to catch behavioral anomalies
        THRESHOLD = 0.25
        # -----------------------------------------------
        
        df['Threat_Score'] = probs
        df['Prediction'] = np.where(probs > THRESHOLD, "Insider Threat", "Safe")
        
        st.success(generate_alert("Data Breach", float(np.mean(probs))))
        st.dataframe(df.head(10))
        st.write(generate_llm_summary("Insider Threat", df))
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# -------------------------
# Main App Entry
# -------------------------
def main():
    st.set_page_config(page_title="Cyber Shield v2", page_icon="🛡️", layout="wide")
    
    # Auth Check
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        st.title("🛡️ Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in ["admin", "user"] and p in ["admin123", "user123"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        return

    # Logged In Layout
    page = sidebar_nav()
    
    if page == "Overview":
        view_overview()
    elif page == "Devices":
        view_devices()
    elif page == "AI Assistant":
        view_chat()
    elif page == "File Scanner":
        view_file_scanner()

if __name__ == "__main__":
    main()
