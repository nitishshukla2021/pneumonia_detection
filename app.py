import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# -----------------------------
# Model Download & Load
# -----------------------------
MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?id=<FILE_ID>"  # Replace with your actual Google Drive file ID

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_pneumonia_model():
    model = load_model(MODEL_PATH)
    return model

model = load_pneumonia_model()

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image
    image = image.convert("L")        # Convert to grayscale
    image = np.array(image)
    image = image / 255.0             # Normalize
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)  # Convert to 3 channels
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return image

# -----------------------------
# Initialize Session State
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.pred_score = None

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Pneumonia Detection using CNN")
st.write("Upload a Chest X-ray image to check whether pneumonia is present or not.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Resize for display to avoid huge image
    display_image = image.copy()
    display_image.thumbnail((400, 400))
    st.image(display_image, caption="Uploaded X-ray", use_column_width=False)

    # Preprocess and predict
    img = preprocess_image(image)
    pred = model.predict(img)[0][0]

    # Store in session state
    st.session_state.pred_score = pred
    st.session_state.result = "PNEUMONIA DETECTED" if pred > 0.5 else "NORMAL"

# Display prediction if available
if st.session_state.result:
    st.subheader("🔍 Prediction Result:")
    st.write(f"*Model Output Score:* {st.session_state.pred_score:.4f}")

    if st.session_state.result == "PNEUMONIA DETECTED":
        st.error("⚠ *PNEUMONIA DETECTED*")
        st.markdown(
            """
- The model suggests the presence of pneumonia in the chest X-ray.
- Please consult a medical professional for accurate diagnosis.
- Early detection can help in better treatment and management.
            """
        )
    else:
        st.success("✅ *NORMAL – No Pneumonia Detected*")
        st.markdown(
            """
- The model indicates that the X-ray appears normal.
- No signs of pneumonia were detected.
- If symptoms persist, a medical checkup is still recommended.
            """
        )
