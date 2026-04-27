from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from data_preprocessing import DataPreprocessor

DEFAULT_MODEL_PATH = Path("saved_model/monkeypox_vit.keras")
DEFAULT_METADATA_PATH = Path("saved_model/metadata.json")

st.set_page_config(page_title="Monkeypox Skin Disease Classification", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)


@st.cache_data
def load_metadata(path: Path):
    if not path.exists():
        return {"class_names": ["monkeypox", "chickenpox", "measles", "normal", "other"], "image_size": 224, "val_accuracy": None}
    return json.loads(path.read_text(encoding="utf-8"))


def predict(model, image: Image.Image, image_size: int, class_names):
    preprocessor = DataPreprocessor(image_size=(image_size, image_size))
    x = preprocessor.preprocess_for_model(image)
    x = np.expand_dims(x, axis=0)
    probabilities = model.predict(x, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index]), probabilities


def main():
    st.title("🩺 Monkeypox Skin Disease Classification")
    st.write("Upload a skin image to run Vision Transformer inference.")

    metadata = load_metadata(DEFAULT_METADATA_PATH)
    class_names = metadata.get("class_names", ["monkeypox", "chickenpox", "measles", "normal", "other"])
    image_size = int(metadata.get("image_size", 224))

    with st.sidebar:
        st.subheader("Model")
        st.write(f"Path: {DEFAULT_MODEL_PATH}")
        val_acc = metadata.get("val_accuracy")
        if val_acc is not None:
            st.metric("Validation accuracy", f"{val_acc * 100:.2f}%")
        st.caption("Educational research tool. Not a medical diagnosis.")

    model = load_model(DEFAULT_MODEL_PATH)
    if model is None:
        st.error("Model not found. Train first with train_model.py.")
        return

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)

    with left:
        st.image(image, caption="Input", use_container_width=True)

    with right:
        if st.button("Predict", use_container_width=True, type="primary"):
            predicted_class, confidence, probabilities = predict(model, image, image_size, class_names)
            st.success(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.subheader("Probabilities")
            for label, prob in sorted(zip(class_names, probabilities), key=lambda item: item[1], reverse=True):
                st.write(f"{label}: {prob * 100:.2f}%")
                st.progress(float(prob))


if __name__ == "__main__":
    main()
