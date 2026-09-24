import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hotel Booking Cancellation", page_icon="🏨", layout="centered")

# ---------------- โหลดโมเดล ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "hotel_tree_.joblib"

# ค่า MinMax ที่ใช้สเกล lead_time และ adr ตอนเทรน
LEAD_TIME_MIN, LEAD_TIME_MAX = 0.0, 737.0
ADR_MIN, ADR_MAX = 0.0, 510.0


def find_model_path():
    exact = os.path.join(APP_DIR, MODEL_NAME)
    if os.path.isfile(exact):
        return exact
    files = [f for f in os.listdir(APP_DIR) if ".joblib" in f.lower()]
    files.sort(key=lambda f: "hotel_tree" not in f.lower())
    return os.path.join(APP_DIR, files[0]) if files else None


@st.cache_resource
def load_model(path):
    return joblib.load(path)


MODEL_PATH = find_model_path()
if MODEL_PATH is None:
    st.error(f"ไม่พบไฟล์โมเดล {MODEL_NAME} — วางไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

model = load_model(MODEL_PATH)
FEATURES = list(model.feature_names_in_)
ROOM_TYPES = [f.removeprefix("reserved_room_type_") for f in FEATURES if f.startswith("reserved_room_type_")]
COUNTRIES = [f.removeprefix("country_") for f in FEATURES if f.startswith("country_")]


def minmax(x, lo, hi):
    return (x - lo) / (hi - lo)


# ---------------- สไตล์ ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, label, input {
    font-family: 'Prompt', sans-serif !important;
}
.block-container { padding-top: 3rem; max-width: 760px; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
    border-radius: 20px; padding: 28px 32px; color: #fff; margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(37, 99, 235, .25);
}
.hero h1 { color: #fff; font-size: 1.9rem; font-weight: 700; margin: 0; padding: 0; }
.hero p  { color: #dbeafe; margin: 6px 0 0; font-weight: 300; }

.section { font-weight: 600; font-size: 1.05rem; margin: 6px 0 2px; }

[data-testid="stForm"] {
    border: 1px solid rgba(148, 163, 184, .35); border-radius: 18px;
    padding: 22px 24px 14px; box-shadow: 0 4px 18px rgba(15, 23, 42, .06);
}
.stFormSubmitButton button {
    background: linear-gradient(135deg, #2563eb, #0ea5e9); color: #fff; border: 0;
    border-radius: 12px; padding: .7rem 0; font-size: 1.05rem; font-weight: 600;
}
.stFormSubmitButton button:hover { filter: brightness(1.08); color: #fff; }

.result {
    border-radius: 18px; padding: 24px 28px; margin-top: 24px; color: #fff;
    display: flex; align-items: center; gap: 22px;
}
.result.cancel { background: linear-gradient(135deg, #b91c1c, #ef4444); }
.result.keep   { background: linear-gradient(135deg, #047857, #10b981); }
.result .icon  { font-size: 3rem; line-height: 1; }
.result .label { font-size: .95rem; opacity: .9; }
.result .title { font-size: 1.6rem; font-weight: 700; }
.bar { background: rgba(255,255,255,.3); border-radius: 99px; height: 10px; margin-top: 10px; }
.bar > div { background: #fff; height: 100%; border-radius: 99px; }
.pct { font-size: .95rem; margin-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🏨 ทำนายการยกเลิกการจองโรงแรม</h1>
  <p>กรอกข้อมูลการจอง แล้วให้โมเดลประเมินโอกาสที่ลูกค้าจะยกเลิก</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- ฟอร์ม ----------------
with st.form("booking"):
    st.markdown('<div class="section">📅 ข้อมูลการจอง</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    lead_time = c1.number_input("จองล่วงหน้า (วัน)", min_value=0, max_value=int(LEAD_TIME_MAX), value=30)
    adr = c2.number_input("ราคาเฉลี่ยต่อคืน (ADR)", min_value=0.0, max_value=ADR_MAX, value=100.0, step=5.0)

    st.markdown('<div class="section">👨‍👩‍👧 จำนวนผู้เข้าพัก</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    adults = c3.number_input("ผู้ใหญ่", 0, 60, 2)
    children = c4.number_input("เด็ก", 0, 10, 0)
    babies = c5.number_input("ทารก", 0, 10, 0)

    st.markdown('<div class="section">🛏️ ห้องพักและสัญชาติ</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    room = c6.selectbox("ประเภทห้อง", ROOM_TYPES)
    country = c7.selectbox("ประเทศ", COUNTRIES, index=COUNTRIES.index("PRT") if "PRT" in COUNTRIES else 0)

    submitted = st.form_submit_button("🔮 ทำนายผล", use_container_width=True)

# ---------------- ทำนาย ----------------
if submitted:
    row = dict.fromkeys(FEATURES, 0.0)
    row["lead_time"] = minmax(lead_time, LEAD_TIME_MIN, LEAD_TIME_MAX)
    row["adr"] = minmax(adr, ADR_MIN, ADR_MAX)
    row["FamilySize"] = float(adults + children + babies)
    row[f"reserved_room_type_{room}"] = 1.0
    row[f"country_{country}"] = 1.0

    X = pd.DataFrame([row], columns=FEATURES)
    pred = int(model.predict(X)[0])
    p_cancel = float(model.predict_proba(X)[0][list(model.classes_).index(1)])

    if pred == 1:
        cls, icon, title = "cancel", "❌", "มีแนวโน้มยกเลิกการจอง"
    else:
        cls, icon, title = "keep", "✅", "มีแนวโน้มเข้าพักตามจอง"

    st.markdown(
        f"""
<div class="result {cls}">
  <div class="icon">{icon}</div>
  <div style="flex:1">
    <div class="label">ผลการทำนาย</div>
    <div class="title">{title}</div>
    <div class="bar"><div style="width:{p_cancel*100:.1f}%"></div></div>
    <div class="pct">โอกาสยกเลิก {p_cancel:.1%}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
