import streamlit as st
import numpy as np
import joblib

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield | AI Transaction Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background-color: #f1f5f9; }
#MainMenu, footer { visibility: hidden; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
    padding: 2.6rem 3rem 2.2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "🛡️";
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.12;
}
.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(4px);
    padding: 0.25rem 0.9rem;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.hero h1  { font-size: 2.3rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
.hero p   { font-size: 1rem; opacity: 0.78; margin: 0.45rem 0 0; font-weight: 400; }

/* ── Section Headers ── */
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #64748b;
    margin: 0 0 0.8rem;
}

/* ── Sample Cards ── */
.sample-card {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1rem 0.9rem;
    text-align: center;
    transition: all 0.18s;
}
.sample-card:hover { border-color: #3b82f6; box-shadow: 0 6px 20px rgba(59,130,246,0.14); transform: translateY(-2px); }
.sc-icon  { font-size: 1.9rem; margin-bottom: 0.3rem; }
.sc-label { font-weight: 700; font-size: 0.92rem; margin: 0; color: #0f172a; }
.sc-desc  { font-size: 0.74rem; color: #64748b; margin: 0.15rem 0 0; }

/* ── Input Card ── */
.input-card {
    background: white;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

/* ── Verdict Banners ── */
.verdict {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
}
.verdict-safe   { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border: 2px solid #86efac; }
.verdict-fraud  { background: linear-gradient(135deg,#fff1f2,#fee2e2); border: 2px solid #fca5a5; }
.verdict-icon   { font-size: 2.8rem; flex-shrink: 0; }
.verdict-title  { font-size: 1.5rem; font-weight: 800; margin: 0; }
.verdict-sub    { font-size: 0.88rem; margin: 0.3rem 0 0; opacity: 0.68; }

/* ── Metric Tiles ── */
.metric-tile {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: center;
}
.mt-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 0.4rem; }
.mt-value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
.mt-sub   { font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── Score Bars ── */
.bar-wrap { margin-bottom: 0.9rem; }
.bar-row  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.bar-name { font-size: 0.82rem; font-weight: 600; color: #334155; }
.bar-pct  { font-size: 0.82rem; font-weight: 700; }
.bar-bg   { background: #f1f5f9; border-radius: 999px; height: 9px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }

/* ── Risk Badge ── */
.risk-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.45rem 1.3rem; border-radius: 50px;
    font-weight: 700; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.risk-low      { background:#dcfce7; color:#15803d; }
.risk-medium   { background:#fef9c3; color:#854d0e; }
.risk-high     { background:#fee2e2; color:#b91c1c; }
.risk-critical { background:#1e1b4b; color:#c7d2fe; }

/* ── Interpretation Card ── */
.interp {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.3rem;
    margin-top: 0.5rem;
}
.interp p { margin: 0; font-size: 0.88rem; color: #374151; line-height: 1.65; }

/* ── Sidebar ── */
[data-testid="stSidebar"] > div:first-child { background: #0f172a; padding-top: 1.5rem; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #e2e8f0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: white !important; }
.sidebar-card {
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}
.sidebar-card p { margin: 0.2rem 0; font-size: 0.82rem; color: #94a3b8 !important; }
.sidebar-card strong { color: #e2e8f0 !important; }

/* ── Analyze Button ── */
div[data-testid="stVerticalBlock"] .stButton > button[kind="primary"] {
    height: 3.2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Artifacts ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("fraud_detection_model.pkl")
    scaler   = joblib.load("fraud_scaler.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, scaler, features

model, scaler, feature_columns = load_artifacts()

# ─── Sample Transactions ──────────────────────────────────────────────────────
SAMPLES = {
    "normal": {
        "Time": 406.0, "Amount": 149.62,
        "V1": -1.3598, "V2": -0.0728, "V3":  2.5363, "V4":  1.3782,
        "V5": -0.3383, "V6":  0.4624, "V7":  0.2396, "V8":  0.0987,
        "V9":  0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
        "V13": -0.9913,"V14": -0.3112,"V15":  1.4682, "V16": -0.4704,
        "V17":  0.2080,"V18":  0.0258,"V19":  0.4040, "V20":  0.2514,
        "V21": -0.0183,"V22":  0.2778,"V23": -0.1105, "V24":  0.0669,
        "V25":  0.1285,"V26": -0.1891,"V27":  0.1336, "V28": -0.0211,
    },
    "suspicious": {
        "Time": 75432.0, "Amount": 498.00,
        "V1": -1.836,  "V2":  0.939,  "V3":  0.463,  "V4":  2.688,
        "V5": -0.430,  "V6": -0.482,  "V7": -1.149,  "V8":  0.745,
        "V9": -1.567,  "V10": -1.431, "V11":  1.376,  "V12": -1.759,
        "V13": -0.793, "V14": -2.300, "V15":  0.899,  "V16": -0.805,
        "V17": -1.519, "V18":  0.004, "V19":  0.409,  "V20":  0.189,
        "V21":  0.346, "V22": -0.148, "V23": -0.341,  "V24": -0.017,
        "V25": -0.003, "V26": -0.122, "V27":  0.037,  "V28": -0.039,
    },
    "fraud": {
        "Time": 406.0, "Amount": 2.69,
        "V1": -2.3122, "V2":  1.9519, "V3": -1.6098, "V4":  3.9979,
        "V5": -0.5220, "V6": -1.4265, "V7": -2.5374, "V8":  1.3914,
        "V9": -2.7700, "V10": -2.7722,"V11":  3.2020, "V12": -2.8992,
        "V13": -0.5955,"V14": -4.2895,"V15":  0.3898, "V16": -1.1407,
        "V17": -2.8300,"V18": -0.0168,"V19":  0.4163, "V20":  0.3269,
        "V21":  0.7101,"V22": -0.5732,"V23": -0.5714, "V24": -0.1001,
        "V25": -0.1371,"V26": -0.0550,"V27": -0.0597, "V28": -0.0526,
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def risk_level(prob: float):
    if prob < 0.25:
        return "LOW",      "#15803d", "risk-low",      "✅"
    elif prob < 0.50:
        return "MEDIUM",   "#854d0e", "risk-medium",   "⚠️"
    elif prob < 0.75:
        return "HIGH",     "#b91c1c", "risk-high",     "🚨"
    else:
        return "CRITICAL", "#c7d2fe", "risk-critical", "🔴"

def anomaly_score(fraud_prob: float, amount: float, v: dict) -> int:
    prob_part   = fraud_prob * 60
    # Amount deviation from dataset mean (88.35, std 250.12)
    amt_z       = abs((amount - 88.35) / 250.12)
    amt_part    = min(amt_z * 8.0, 20)
    # V14, V4, V12 are top fraud indicators in this dataset
    feat_part   = min((abs(v.get("V14", 0)) * 1.6 +
                       abs(v.get("V4",  0)) * 0.5 +
                       abs(v.get("V12", 0)) * 0.5) * 1.1, 20)
    return min(int(prob_part + amt_part + feat_part), 100)

def make_prediction(time_v, amount_v, v_dict):
    scaled       = scaler.transform([[amount_v, time_v]])[0]
    s_amount, s_time = scaled[0], scaled[1]
    row = []
    for col in feature_columns:
        if   col == "Time":   row.append(s_time)
        elif col == "Amount": row.append(s_amount)
        else:                 row.append(v_dict.get(col, 0.0))
    X    = np.array(row).reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return int(pred), float(prob)

# ─── Session-state defaults ────────────────────────────────────────────────────
_all_cols = feature_columns
for _c in _all_cols:
    if f"inp_{_c}" not in st.session_state:
        st.session_state[f"inp_{_c}"] = 0.0

def load_sample(key: str):
    s = SAMPLES[key]
    for c in _all_cols:
        st.session_state[f"inp_{c}"] = float(s.get(c, 0.0))

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("*AI-Powered Transaction Monitoring*")
    st.divider()

    st.markdown("### 🤖 Model Details")
    st.markdown("""
    <div class="sidebar-card">
        <p><strong>Algorithm</strong><br>Random Forest Classifier</p>
        <p><strong>Features</strong><br>30 (Time · V1–V28 · Amount)</p>
        <p><strong>Training data</strong><br>284,807 transactions</p>
        <p><strong>Sklearn version</strong><br>1.5.1</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Performance")
    st.markdown("""
    <div class="sidebar-card">
        <p><strong>Precision</strong> — 95.2 %</p>
        <p><strong>Recall</strong>    — 79.4 %</p>
        <p><strong>F1 Score</strong>  — 86.6 %</p>
        <p><strong>AUC-ROC</strong>   — 97.8 %</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚦 Risk Legend")
    st.markdown("""
    <div class="sidebar-card">
        <p>🟢 <strong>LOW</strong> — 0 – 25 %</p>
        <p>🟡 <strong>MEDIUM</strong> — 25 – 50 %</p>
        <p>🔴 <strong>HIGH</strong> — 50 – 75 %</p>
        <p>⚫ <strong>CRITICAL</strong> — 75 – 100 %</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Dataset · Kaggle Credit Card Fraud Detection")
    st.caption("Built with Streamlit + scikit-learn")

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-pill">🛡️ Enterprise Fraud Intelligence Platform</div>
    <h1>FraudShield</h1>
    <p>Real-time AI-powered credit card transaction analysis &amp; fraud detection</p>
</div>
""", unsafe_allow_html=True)

# ─── Quick-Test Samples ────────────────────────────────────────────────────────
st.markdown('<p class="section-title">⚡ Quick Test — Load a Sample Transaction</p>', unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3)

with sc1:
    st.markdown("""
    <div class="sample-card">
        <div class="sc-icon">✅</div>
        <p class="sc-label">Normal Transaction</p>
        <p class="sc-desc">Typical retail purchase · Low risk</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load Normal", use_container_width=True, key="btn_normal"):
        load_sample("normal")
        st.rerun()

with sc2:
    st.markdown("""
    <div class="sample-card">
        <div class="sc-icon">⚠️</div>
        <p class="sc-label">Suspicious Transaction</p>
        <p class="sc-desc">Unusual patterns detected · Medium risk</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load Suspicious", use_container_width=True, key="btn_susp"):
        load_sample("suspicious")
        st.rerun()

with sc3:
    st.markdown("""
    <div class="sample-card">
        <div class="sc-icon">🚨</div>
        <p class="sc-label">Fraudulent Transaction</p>
        <p class="sc-desc">Known fraud signature · Critical risk</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load Fraudulent", use_container_width=True, key="btn_fraud"):
        load_sample("fraud")
        st.rerun()

st.divider()

# ─── Transaction Input Form ────────────────────────────────────────────────────
st.markdown('<p class="section-title">📋 Transaction Details</p>', unsafe_allow_html=True)

col_t, col_a = st.columns(2)
with col_t:
    time_val = st.number_input(
        "⏱️ Time (seconds since first transaction)",
        key="inp_Time", format="%.2f", min_value=0.0,
        help="Seconds elapsed from the first transaction in the dataset.",
    )
with col_a:
    amount_val = st.number_input(
        "💰 Transaction Amount ($)",
        key="inp_Amount", format="%.2f", min_value=0.0,
        help="Dollar value of the transaction.",
    )

with st.expander("🔬 PCA Features (V1 – V28) — click to expand", expanded=False):
    st.caption(
        "These 28 features are the result of PCA transformation applied to the "
        "original transaction data to protect cardholder privacy."
    )
    v_grid = st.columns(4)
    v_values: dict = {}
    v_feature_cols = [c for c in feature_columns if c.startswith("V")]
    for i, col in enumerate(v_feature_cols):
        with v_grid[i % 4]:
            v_values[col] = st.number_input(col, key=f"inp_{col}", format="%.4f")

st.markdown("")
run = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)

# ─── Results ──────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running fraud analysis…"):
        pred, fraud_prob = make_prediction(time_val, amount_val, v_values)

    a_score              = anomaly_score(fraud_prob, amount_val, v_values)
    label, color, cls, icon = risk_level(fraud_prob)
    fraud_pct            = fraud_prob * 100
    legit_pct            = (1 - fraud_prob) * 100

    st.divider()
    st.markdown('<p class="section-title">📊 Analysis Results</p>', unsafe_allow_html=True)

    # Verdict banner
    if pred == 0:
        st.markdown(f"""
        <div class="verdict verdict-safe">
            <div class="verdict-icon">✅</div>
            <div>
                <p class="verdict-title" style="color:#15803d;">Transaction Approved</p>
                <p class="verdict-sub" style="color:#166534;">
                    No fraudulent patterns detected. This transaction appears legitimate
                    and has been cleared for processing.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict verdict-fraud">
            <div class="verdict-icon">🚨</div>
            <div>
                <p class="verdict-title" style="color:#dc2626;">Fraud Alert — Transaction Blocked</p>
                <p class="verdict-sub" style="color:#991b1b;">
                    This transaction has been flagged as potentially fraudulent.
                    Immediate review and cardholder notification is recommended.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Metric tiles
    m1, m2, m3 = st.columns(3)
    with m1:
        c = "#dc2626" if fraud_prob >= 0.5 else "#d97706" if fraud_prob >= 0.25 else "#15803d"
        st.markdown(f"""
        <div class="metric-tile">
            <div class="mt-label">Fraud Probability</div>
            <div class="mt-value" style="color:{c};">{fraud_pct:.1f}%</div>
            <div class="mt-sub">Model confidence in fraud</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        bar_c = "#dc2626" if a_score >= 75 else "#d97706" if a_score >= 50 else "#15803d"
        st.markdown(f"""
        <div class="metric-tile">
            <div class="mt-label">Anomaly Risk Score</div>
            <div class="mt-value" style="color:{bar_c};">{a_score}<span style="font-size:1rem;font-weight:500;color:#94a3b8;">/100</span></div>
            <div class="mt-sub">Composite risk index</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="mt-label">Risk Level</div>
            <div class="mt-value" style="font-size:1.6rem;">{icon} {label}</div>
            <div class="mt-sub">{"Block & review" if pred == 1 else "Clear to proceed"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Score bars + risk badge
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<p class="section-title">Score Breakdown</p>', unsafe_allow_html=True)

        fraud_bar_color  = "#dc2626" if fraud_pct >= 50 else "#f59e0b" if fraud_pct >= 25 else "#22c55e"
        legit_bar_color  = "#22c55e"
        risk_bar_color   = "#dc2626" if a_score >= 75 else "#f59e0b" if a_score >= 50 else "#22c55e"

        def bar(label, pct, color, suffix="%"):
            st.markdown(f"""
            <div class="bar-wrap">
                <div class="bar-row">
                    <span class="bar-name">{label}</span>
                    <span class="bar-pct" style="color:{color};">{pct:.1f}{suffix}</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{pct}%;background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        bar("Fraud Probability",      fraud_pct,  fraud_bar_color)
        bar("Legitimate Probability", legit_pct,  legit_bar_color)
        bar("Anomaly Risk Score",     a_score,    risk_bar_color, suffix="/100")

    with right:
        st.markdown('<p class="section-title">Risk Classification</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom:1.2rem;">
            <span class="risk-badge {cls}">{icon} {label} RISK</span>
        </div>
        """, unsafe_allow_html=True)

        action_map = {
            "LOW":      ("APPROVE",   "#15803d"),
            "MEDIUM":   ("REVIEW",    "#854d0e"),
            "HIGH":     ("BLOCK",     "#b91c1c"),
            "CRITICAL": ("BLOCK NOW", "#c7d2fe"),
        }
        action, a_color = action_map[label]
        bg = "#1e1b4b" if label == "CRITICAL" else "#f8fafc"
        st.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:1rem 1.2rem;margin-top:0.4rem;">
            <div class="mt-label" style="color:#94a3b8;">Recommended Action</div>
            <div style="font-size:1.3rem;font-weight:800;color:{a_color};margin-top:0.2rem;">{action}</div>
        </div>
        """, unsafe_allow_html=True)

    # Interpretation
    interp_map = {
        "LOW": (
            "The transaction exhibits characteristics consistent with legitimate cardholder behavior. "
            "Amount, timing, and PCA feature distributions are within expected ranges. "
            "No further action is required."
        ),
        "MEDIUM": (
            "The transaction shows some deviations from typical patterns. While the model does not "
            "classify this as definitively fraudulent, secondary verification (e.g., OTP or call-back) "
            "is recommended before final processing."
        ),
        "HIGH": (
            "Multiple fraud indicators have been detected. The transaction significantly deviates from "
            "normal behavioral profiles. Immediate manual review and temporary card hold are recommended "
            "until the cardholder is verified."
        ),
        "CRITICAL": (
            "This transaction matches known fraud signatures with very high model confidence. "
            "Strongly recommend blocking the transaction immediately, freezing the card, and initiating "
            "a fraud investigation. Cardholder should be notified without delay."
        ),
    }

    st.markdown(f"""
    <div class="interp">
        <p><strong>Risk Interpretation:</strong> {interp_map[label]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Transaction summary
    st.markdown("")
    with st.expander("📄 Full Transaction Summary", expanded=False):
        sa, sb = st.columns(2)
        with sa:
            st.markdown(f"**Time:** `{time_val:.0f}` seconds")
            st.markdown(f"**Amount:** `${amount_val:.2f}`")
            st.markdown(f"**Fraud Probability:** `{fraud_pct:.2f}%`")
            st.markdown(f"**Legitimate Probability:** `{legit_pct:.2f}%`")
        with sb:
            st.markdown(f"**Anomaly Risk Score:** `{a_score} / 100`")
            st.markdown(f"**Risk Level:** `{label}`")
            st.markdown(f"**Model Decision:** `{'FRAUD' if pred == 1 else 'LEGITIMATE'}`")
            st.markdown(f"**Action:** `{action}`")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("🛡️ FraudShield  ·  Random Forest Classifier  ·  Kaggle Credit Card Fraud Detection Dataset")
