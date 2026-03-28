import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

# Page configuration
st.set_page_config(
    page_title="Glaucoma AI Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .stTitle {
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        padding-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #4B5563;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
    }

    /* Card-like containers */
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: white;
        border-radius: 10px;
        width: 100%;
        border: none;
        padding: 0.5rem;
        font-weight: 600;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1E3A8A;
    }

    /* Prediction Result Box */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin-top: 10px;
        text-align: center;
    }
    
    .glaucoma-result {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
    
    .normal-result {
        background-color: #DCFCE7;
        border: 2px solid #22C55E;
        color: #166534;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #6B7280;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #E5E7EB;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062312.png", width=100)
    st.title("About Project")
    st.info("""
    **Glaucoma Detection Dashboard** is an AI-powered screening tool designed to assist in early detection of Glaucoma using Retinal Fundus Images.
    """)
    st.divider()
    st.markdown("### How to use:")
    st.markdown("1. Upload a clear fundus image.\n2. Wait for AI analysis.\n3. Review the prediction result.")
    st.divider()
    st.warning("⚠️ **Disclaimer:** This tool is for research purposes and not a substitute for professional medical advice.")

# Load model (Logic remains unchanged)
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
    try:
        model.load_weights('glaucoma_model.h5')
    except:
        # Fallback if file doesn't exist during preview
        pass
    return model

model = load_model()

# Header Section
st.markdown('<h1 class="stTitle">👁️ Glaucoma Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Based Retinal Fundus Image Screening</p>', unsafe_allow_html=True)

# Main Content Layout
col_up, col_res = st.columns([1, 1.2], gap="large")

with col_up:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)

with col_res:
    st.markdown("### 🔍 Analysis Result")
    
    if uploaded_file is not None:
        with st.spinner("🧠 AI is analyzing the image..."):
            # Preprocessing
            img_processed = image.convert("RGB").resize((224, 224))
            img_array = np.array(img_processed) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            prediction = model.predict(img_array)[0][0]
            
            if prediction > 0.5:
                label = "Glaucoma Detected (GON+)"
                confidence = prediction
                color_class = "glaucoma-result"
                icon = "⚠️"
            else:
                label = "Normal Condition (GON-)"
                confidence = 1 - prediction
                color_class = "normal-result"
                icon = "✅"
            
            # Display result in a styled box
            st.markdown(f"""
                <div class="result-box {color_class}">
                    <h2 style='margin:0;'>{icon} {label}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{confidence:.2%}")
            with m2:
                st.metric("Model Score", f"{prediction:.4f}")
            
            st.progress(float(confidence))
            
            with st.expander("Technical Details"):

                st.write(f"Raw Output: `{prediction:.4f}`")
                st.write(f"Threshold Used: `0.5` (Sigmoid)")
                st.write("Image Dimensions: `224x224 px`")
    else:
        st.info("Please upload an image on the left to see the analysis.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; padding: 20px;'>"
    "<b>IDSC 2026</b> | ROAD TO IMMO TEAM | UNESA"
    "</div>", 
    unsafe_allow_html=True
)