import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="Pistachio Classifier", page_icon="🌰", layout="centered")

# ===== CSS 커스텀 =====
st.markdown(
    """
    <style>
    .divider {
        margin-top: 25px;
        margin-bottom: 25px;
        height: 1px;
        background-color: #e1e4e8;
    }
    .pred-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0px 3px 18px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 22px;
    }
    .footer {
        text-align:center;
        margin-top:45px;
        color:#999;
        font-size:12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== 제목 =====
st.markdown(
    """
    <div style="text-align:center;">
        <span style="font-size:60px;">🌰</span>
        <h1 style="font-weight:700; margin-bottom: 5px;">Pistachio Classifier</h1>
        <p style="margin-top:-10px; font-size:17px;">Kirmizi vs Siirt Pistachio</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📁 Upload your pistachio image", type=["jpg", "jpeg", "png"])

# ===== 모델 로드 =====
@st.cache_resource
def load_cnn():
    return load_model("pistachio_cnn_A.h5")

model = load_cnn()

if uploaded_file:

    IMG_SIZE = (120, 120)

    img_raw = Image.open(uploaded_file).convert("RGB")
    img_model = img_raw.resize(IMG_SIZE)

    x = np.array(img_model) / 255.0
    x = np.expand_dims(x, axis=0)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(img_raw, caption="", use_container_width=True)

    prob = float(model.predict(x)[0][0])

    if prob >= 0.5:
        label = "Siirt Pistachio"
        confidence = prob * 100
    else:
        label = "Kirmizi Pistachio"
        confidence = (1 - prob) * 100

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.pie(
        [confidence, 100 - confidence],
        colors=["#34A853", "#E8E8E8"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.35},
    )
    ax.text(
        0, 0, f"{confidence:.1f}%",
        ha="center", va="center",
        fontsize=13, fontweight="bold"
    )
    ax.set(aspect="equal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", transparent=True)
    buf.seek(0)
    gauge_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    st.markdown(
        f"""
        <div class="pred-card">
            <div style="font-size:16px; margin-bottom:8px;">
                <b>Prediction:</b> <span style="color:#2E7D32;">{label}</span>
            </div>
            <img src="data:image/png;base64,{gauge_b64}" style="width:130px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            DKU_DeepLearning • 32215002 • Yunha Hwang
        </div>
        """,
        unsafe_allow_html=True
)
