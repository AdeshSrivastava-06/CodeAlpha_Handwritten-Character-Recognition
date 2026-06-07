import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Character Recognition")
st.markdown("Draw a character or upload an image to recognize it.")

@st.cache_resource
def load_resources():
    model = load_model('model/best_model.h5')
    with open('model/class_labels.json') as f:
        labels = json.load(f)
    return model, labels

model, labels = load_resources()

def preprocess_drawn(img_array):
    img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Fix mirror issue
    img = np.fliplr(img)
    img = img.astype('float32') / 255.0
    return img.reshape(1, 28, 28, 1)

def preprocess_uploaded(pil_img):
    img = ImageOps.grayscale(pil_img)
    img = ImageOps.invert(img)
    img = np.array(img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Fix mirror issue
    img = np.fliplr(img)
    img = img.astype('float32') / 255.0
    return img.reshape(1, 28, 28, 1)

def show_predictions(preds):
    top5_idx = preds.argsort()[-5:][::-1]
    top_char = labels[str(top5_idx[0])]
    top_conf = preds[top5_idx[0]] * 100
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border: 2px solid #48bb78;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <div style="font-size: 4rem; font-weight: 900;
                    color: #276749; line-height: 1;">
            {top_char}
        </div>
        <div style="color: #48bb78; font-size: 1.1rem;
                    font-weight: 600; margin-top: 0.4rem;">
            {top_conf:.1f}% confidence
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Top 5 predictions:**")
    for idx in top5_idx:
        char = labels[str(idx)]
        conf = float(preds[idx]) * 100
        st.progress(int(conf), text=f"`{char}`  —  {conf:.1f}%")

tab1, tab2 = st.tabs(["✏️ Draw", "📁 Upload"])

with tab1:
    st.markdown("Draw a **single character** in the box below:")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=8,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if st.button("🔍 Predict", use_container_width=True, key="draw_btn"):
        if canvas.image_data is not None:
            gray_check = cv2.cvtColor(
                canvas.image_data.astype(np.uint8),
                cv2.COLOR_RGBA2GRAY
            )
            if gray_check.min() < 200:
                result = preprocess_drawn(canvas.image_data)
                preds = model.predict(result, verbose=0)[0]
                show_predictions(preds)
            else:
                st.warning("Canvas is empty — please draw a character first.")
        else:
            st.warning("Please draw something first.")

with tab2:
    st.markdown("Upload a clear image of a **single handwritten character:**")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    if uploaded:
        pil_img = Image.open(uploaded)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(pil_img, caption="Uploaded", width=150)
        if st.button("🔍 Predict", use_container_width=True, key="upload_btn"):
            result = preprocess_uploaded(pil_img)
            preds = model.predict(result, verbose=0)[0]
            show_predictions(preds)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#a0aec0; font-size:0.8rem;'>"
    "Built with TensorFlow + Streamlit · "
    "CNN trained on EMNIST Balanced · "
    "47 classes · 90.28% accuracy"
    "</div>",
    unsafe_allow_html=True
)