import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hotel Booking Cancellation", page_icon="🏨", layout="centered")

# ---------------- โหลดโมเดล ----------------
# โมเดลเป็น scikit-learn Pipeline (เติมค่าหาย + สเกล + One-hot + Gradient Boosting)
# ได้จาก Experiment_002_030.ipynb จึงรับข้อมูลดิบได้โดยตรง ไม่ต้องสเกลเอง
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "hotel_cancel_model.joblib"

# ขอบบน ADR (Q3 + 1.5 IQR) ที่คำนวณจาก training set (summary.json: adr_upper)
ADR_UPPER = 227.388
LEAD_TIME_MAX = 737

# เกณฑ์ 3 ระดับ เลือกจาก out-of-fold probability บน training set (ตาราง T4)
LOW_T, HIGH_T = 0.40, 0.60


@st.cache_resource
def load_model(path):
    return joblib.load(path)


MODEL_PATH = os.path.join(APP_DIR, MODEL_NAME)
if not os.path.isfile(MODEL_PATH):
    st.error(f"ไม่พบไฟล์โมเดล {MODEL_NAME} — วางไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

model = load_model(MODEL_PATH)

# ประเภทห้อง (ชื่อไทย -> รหัสห้องในข้อมูล) เรียงตามระดับราคาเฉลี่ยจากข้อมูล
ROOM_TYPES = {
    "ห้องสแตนดาร์ด": "A",
    "ห้องซูพีเรีย": "D",
    "ห้องดีลักซ์": "F",
    "ห้องสวีท": "H",
}

# 10 ประเทศที่พบมากที่สุดใน training set (ตรงกับที่โมเดลเก็บไว้) ที่เหลือโมเดลรวมเป็น "อื่น ๆ"
OTHER_LABEL = "อื่น ๆ (Other)"
COUNTRIES = {
    "โปรตุเกส": "PRT",
    "สหราชอาณาจักร": "GBR",
    "ฝรั่งเศส": "FRA",
    "สเปน": "ESP",
    "เยอรมนี": "DEU",
    "อิตาลี": "ITA",
    "ไอร์แลนด์": "IRL",
    "เบลเยียม": "BEL",
    "บราซิล": "BRA",
    "เนเธอร์แลนด์": "NLD",
    OTHER_LABEL: "OTHER",  # ค่าที่ไม่อยู่ใน 10 ประเทศ -> หมวด infrequent ของ One-hot
}


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
.result.warn   { background: linear-gradient(135deg, #b45309, #f59e0b); }
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
    lead_time = c1.number_input("จองล่วงหน้า (วัน)", min_value=0, max_value=LEAD_TIME_MAX, value=30)
    adr = c2.number_input("ราคาเฉลี่ยต่อคืน (ADR)", min_value=0.0, max_value=1000.0, value=100.0, step=5.0)

    st.markdown('<div class="section">👨‍👩‍👧 จำนวนผู้เข้าพัก</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    adults = c3.number_input("ผู้ใหญ่", 0, 60, 2)
    children = c4.number_input("เด็ก", 0, 10, 0)
    babies = c5.number_input("ทารก", 0, 10, 0)

    st.markdown('<div class="section">🛏️ ห้องพักและสัญชาติ</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    room_label = c6.selectbox("ประเภทห้อง", list(ROOM_TYPES))
    country_label = c7.selectbox("ประเทศ", list(COUNTRIES), index=0)

    submitted = st.form_submit_button("🔮 ทำนายผล", use_container_width=True)

# ---------------- ทำนาย ----------------
if submitted:
    X = pd.DataFrame([{
        "lead_time": float(lead_time),
        "adr": min(float(adr), ADR_UPPER),          # ตัดค่าเกินขอบบนเหมือนตอนเทรน
        "FamilySize": float(adults + children + babies),
        "reserved_room_type": ROOM_TYPES[room_label],
        "country": COUNTRIES[country_label],
    }])
    p_cancel = float(model.predict_proba(X)[0][list(model.classes_).index(1)])

    if p_cancel >= HIGH_T:
        cls, icon, title = "cancel", "❌", "ความเสี่ยงสูง: มีแนวโน้มยกเลิกการจอง"
    elif p_cancel <= LOW_T:
        cls, icon, title = "keep", "✅", "ความเสี่ยงต่ำ: มีแนวโน้มเข้าพักตามจอง"
    else:
        cls, icon, title = "warn", "⚠️", "ความเสี่ยงปานกลาง: ควรติดตามใกล้ชิด"

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