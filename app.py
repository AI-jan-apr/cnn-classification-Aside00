import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="AI Pet Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
    }
    
    h1 {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    [data-testid="stFileUploader"] {
        background-color: #1E293B;
        border: 2px dashed #6366f1;
        border-radius: 12px;
        padding: 20px;
    }

    [data-testid="stImage"] img {
        border-radius: 20px;
        border: 4px solid #1E293B;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
    }

    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }

    .result-container {
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .result-dog {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid #6366f1;
    }

    .result-cat {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid #a855f7;
    }
    
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>AI Pet Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>State-of-the-art Cat vs Dog identification.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('best_model.h5')

with st.spinner('Waking up the AI...'):
    model = load_my_model()

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    thumb_size = (400, 400) 
    display_image = raw_image.copy()
    display_image.thumbnail(thumb_size)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(ImageOps.exif_transpose(display_image), use_column_width=True)
    
    if st.button('IDENTIFY NOW ✨'):
        with st.spinner('Neural networks processing...'):
            img = raw_image.resize((150, 150)) 
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array /= 255.0

            prediction = model.predict(img_array)
            
            if prediction[0] > 0.5:
                st.markdown(f"""
                <div class='result-container result-dog'>
                    <h2 style='color:#818cf8;'>🐾 It's a DOG!</h2>
                    <p style='color:#94a3b8;'>Confidence: <b>{prediction[0][0]*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-container result-cat'>
                    <h2 style='color:#c084fc;'>🐾 It's a CAT!</h2>
                    <p style='color:#94a3b8;'>Confidence: <b>{(1-prediction[0][0])*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

st.sidebar.title("System Info")
st.sidebar.markdown("---")
st.sidebar.write("Architecture: **CNN**")
st.sidebar.write("Environment: **Production**")
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Amjad | Riyadh, 2026")