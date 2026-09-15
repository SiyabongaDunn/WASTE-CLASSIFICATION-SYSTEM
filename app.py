# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from inference import WasteClassifier
from utils import get_top_classes, class_names

# Page configuration
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="",
    layout="wide"
)

st.title("Waste Classification System")
st.markdown("Upload an image to classify waste materials using Vision Transformer.")

# Initialize classifier
@st.cache_resource
def load_classifier():
    return WasteClassifier()

classifier = load_classifier()

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model selection (ViT only)
    selected_model = "vit"
    st.info("Model: ViT (50% Unfreeze)")
    
    # XAI toggle
    show_xai = st.checkbox("Show XAI Visualization", value=True)
    
    st.divider()
    st.caption("Model Information")
    st.caption("Vision Transformer with two-stage fine-tuning (50% unfreeze)")

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Classify", type="primary", use_container_width=True):
            with st.spinner("Classifying..."):
                start_time = time.time()
                result = classifier.predict(image, selected_model, show_xai)
                elapsed_time = (time.time() - start_time) * 1000
                result['latency_ms'] = elapsed_time
                st.session_state['result'] = result

with col2:
    st.subheader("Prediction Results")
    
    # In app.py, find this section:
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        pred_class = result['class']
        confidence = result['confidence']
        
        # FIX: Clamp confidence for progress bar
        confidence_clamped = max(0.0, min(1.0, confidence))
        
        if pred_class == 'trash':
            st.error(f"**{pred_class.upper()}**")
        else:
            st.success(f"**{pred_class.upper()}**")
        
        # Use clamped confidence for progress bar
        st.progress(confidence_clamped)
        st.caption(f"Confidence: {confidence_clamped:.2%}")
        st.caption(f"Latency: {result['latency_ms']:.1f} ms")
        
        st.divider()
        st.caption("Top Predictions")
        top_classes = get_top_classes(np.array([result['predictions']]))
        for item in top_classes:
            st.caption(f"  {item['class']}: {item['confidence']:.2%}")
        
        if show_xai and 'xai_image' in result and result['xai_image'] is not None:
            st.divider()
            st.subheader("XAI Visualization")
            st.caption("Type: ViT Attention Map")
            
            heatmap = result['xai_image']
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            img_array = np.array(image.resize((224, 224)))
            axes[0].imshow(img_array)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            from matplotlib import cm
            heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
            heatmap_colored = cm.jet(heatmap_norm)[:, :, :3]
            overlay = 0.6 * img_array / 255.0 + 0.4 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            
            axes[1].imshow(overlay)
            axes[1].set_title("ViT Attention Map")
            axes[1].axis('off')
            
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Upload an image and click 'Classify' to see results here.")

st.divider()
st.caption("Waste Classification System | Powered by TensorFlow + Streamlit")