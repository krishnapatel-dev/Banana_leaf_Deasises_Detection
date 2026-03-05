import streamlit as st
st.set_page_config(page_title="Banana Leaf Disease Detector", page_icon="🍌", layout="wide")

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Hide Streamlit default menu/footer for a cleaner UI
hide = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide, unsafe_allow_html=True)

# --------------------------
# CONFIGURATION
# --------------------------
MODEL_PATH = "custom_cnn_best_aug.h5"
CLASS_NAMES = ['Cordana', 'Healthy', 'Pestalotiopsis', 'Sigatoka']
IMG_SIZE = (300, 300)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# --------------------------
# HEADER UI
# --------------------------
st.markdown("""
<div style="text-align:center;">
    <h1>🍌 Banana Leaf Disease Detection</h1>
    <h4 style="color:gray;">Upload a banana leaf image to diagnose plant health</h4>
</div>
""", unsafe_allow_html=True)

st.write("")  # spacing

# --------------------------
# MAIN LAYOUT
# --------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file:

        # Preprocess
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_idx = np.argmax(preds)
        pred_label = CLASS_NAMES[pred_idx]
        confidence = np.max(preds) * 100

        st.markdown("---")
        st.subheader("🩺 Diagnosis Result")

        # Card-style result box
        st.markdown(f"""
        <div style="
            background-color:#f7f7f7;
            padding:20px;
            border-radius:15px;
            border:1px solid #e0e0e0;
            text-align:center;">
            <h2 style="margin-bottom:5px;">{pred_label}</h2>
            <p style="font-size:18px;color:gray;margin-top:0;">Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Class Probabilities")

        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(CLASS_NAMES, preds[0])
        ax.set_ylabel("Probability")
        ax.set_title("")
        st.pyplot(fig)

    else:
        st.info("Upload a banana leaf image on the left panel to begin analysis.")
