from __future__ import annotations

import json
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.cancer_detection.gradcam import generate_gradcam

st.set_page_config(page_title="Skin Cancer Detection", layout="wide")
st.title("Skin Cancer Detection App (HAM10000)")
st.caption("EfficientNetB0 inference + Grad-CAM visualization")


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def preprocess(image: Image.Image, img_size: int = 224) -> np.ndarray:
    image = image.convert("RGB").resize((img_size, img_size))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(image.size)

    base = image.convert("RGB")
    return Image.blend(base, jet_img, alpha=alpha)


def load_label_map(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


st.sidebar.header("Model settings")
model_path = st.sidebar.text_input("Model file", "models/binary/best_model.keras")
label_map_path = st.sidebar.text_input("Label map JSON", "models/binary/label_map.json")
img_size = st.sidebar.number_input("Input size", min_value=128, max_value=512, value=224, step=32)

uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to start prediction.")
    st.stop()

image = Image.open(uploaded)
col1, col2 = st.columns(2)
col1.image(image, caption="Input image", use_container_width=True)

if not Path(model_path).exists():
    st.error(f"Model not found at: {model_path}")
    st.stop()

model = load_model(model_path)
tensor = preprocess(image, img_size=int(img_size))
pred = model.predict(tensor, verbose=0)

label_map = load_label_map(Path(label_map_path))

if pred.shape[-1] == 1:
    score = float(pred[0][0])
    label = "malignant" if score >= 0.5 else "benign"
    col2.metric("Prediction", label)
    col2.metric("Malignancy probability", f"{score:.4f}")
    class_index = 0
else:
    class_index = int(np.argmax(pred[0]))
    confidence = float(pred[0][class_index])
    label = label_map.get(class_index, str(class_index)) if label_map else str(class_index)
    col2.metric("Prediction", label)
    col2.metric("Confidence", f"{confidence:.4f}")

with st.expander("Raw model output"):
    st.write(pred.tolist())

try:
    heatmap = generate_gradcam(model, tensor, class_index=class_index)
    overlay = overlay_heatmap(image.resize((int(img_size), int(img_size))), heatmap)
    st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
except Exception as exc:
    st.warning(f"Could not generate Grad-CAM: {exc}")
