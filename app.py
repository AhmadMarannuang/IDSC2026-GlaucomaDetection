import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

# Page config
st.set_page_config(
    page_title="Glaucoma Detection Dashboard",
    page_icon="👁️",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    dummy = np.zeros((1, 224, 224, 3))
    model(dummy)
    model.load_weights('glaucoma_model.h5')
    return model

model = load_model()

# Title
st.title("👁️ Glaucoma Detection Dashboard")
st.markdown("### AI-Based Retinal Fundus Image Screening")
st.write("Upload a retinal fundus image to predict whether it is **Glaucoma** or **Normal**.")

# Preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# File uploader
uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Retinal Image", 
                use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0][0]
            
            if prediction > 0.5:
                label = "Glaucoma (GON+)"
                confidence = prediction
                st.error(f"⚠️ Predicted: **{label}**")
            else:
                label = "Normal (GON-)"
                confidence = 1 - prediction
                st.success(f"✅ Predicted: **{label}**")
            
            st.metric("Confidence Score", f"{confidence:.2%}")
            st.progress(float(confidence))
            
            st.write(f"Raw Model Output: `{prediction:.4f}`")
            st.write("Threshold: `0.5`")
    
    st.warning("⚠️ For educational and research use only. Not a clinical diagnostic tool.")

# Footer
st.markdown("---")
st.markdown("**IDSC 2026** | Mathematics for Hope in Healthcare | UPM × UNAIR × UNMUL × UB")
