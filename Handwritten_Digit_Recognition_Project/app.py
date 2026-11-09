import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "mnist_cnn.keras"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit from 0 to 9 or upload an image. The CNN model will predict the digit.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.warning("The trained model is not found.")
    st.info("First run `python train_model.py` in the VS Code terminal.")
    st.stop()


def preprocess_image(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    arr = np.array(img)

    if arr.mean() > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


def predict_image(img):
    x = preprocess_image(img)
    probabilities = model.predict(x, verbose=0)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    return prediction, confidence


def show_prediction(prediction, confidence):
    st.success(f"Predicted Digit: {prediction}")
    st.metric("Confidence", f"{confidence * 100:.2f}%")
    if confidence < 0.60:
        st.warning("The model is not very confident. Try drawing the digit more clearly.")

# Optional drawing canvas
try:
    from streamlit_drawable_canvas import st_canvas

    st.subheader("1. Draw a digit")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )

    if st.button("Predict Drawn Digit"):
        if canvas_result.image_data is None:
            st.error("Please draw a digit first.")
        else:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            prediction, confidence = predict_image(img)
            show_prediction(prediction, confidence)

except ImportError:
    st.info("Drawing feature needs streamlit-drawable-canvas. You can still use image upload below.")

st.subheader("2. Upload a digit image")
uploaded = st.file_uploader("Upload PNG/JPG image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=220)

    if st.button("Predict Uploaded Image"):
        prediction, confidence = predict_image(img)
        show_prediction(prediction, confidence)


st.subheader("3. How it works")
st.write("The CNN receives a 28×28 grayscale image, extracts visual features using convolution layers, and returns probabilities for digits 0–9.")
