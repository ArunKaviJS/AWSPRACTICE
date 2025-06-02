import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('..\Models\my_model_cnn_cifar10.h5')

# CIFAR-10 Labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# --- Page Config ---
st.set_page_config(page_title="Image Classifier", layout="centered")

# --- CSS Styling ---
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        color: #FF6F61;
        text-align: center;
        font-weight: bold;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #6c757d;
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    .prediction-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
        font-weight: bold;
    }
    .label-list {
        font-size: 16px;
        color: #555;
        margin-top: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of one of the following categories:</div>', unsafe_allow_html=True)

# --- Labels Display ---
st.markdown('<div class="label-list">' + ', '.join(labels) + '</div>', unsafe_allow_html=True)

# --- File Upload Box ---
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Upload an image (jpg/png/jpeg):", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((32, 32))
    img_array = np.array(image)

    # Show image preview
    st.image(image, caption='🔍 Preview', width=200)

    # Model expects (1, 32, 32, 3)
    img_batch = img_array.reshape(1, 32, 32, 3) / 255.0

    # Prediction
    predictions = model.predict(img_batch)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    confidence = float(np.max(predictions)) * 100

    # Show prediction
    st.markdown(f'<div class="prediction-box">Predicted: {predicted_label} <br> Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
