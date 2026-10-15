import base64
import io
import time

import numpy as np
import streamlit as st
from PIL import Image
from matplotlib import cm

from inference import WasteClassifier
from utils import get_top_classes, class_names
from sortlight_component import sortlight_ui

st.set_page_config(
    page_title="Waste Classification System",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_classifier():
    return WasteClassifier()

classifier = load_classifier()

_DEFAULT_PALETTE = ["#c99b6a", "#5eead4", "#a9b4c6", "#e9e4d8", "#60a5fa", "#fb7185"]
CLASS_COLORS = {}
for i, name in enumerate(class_names):
    if name.lower() == "trash":
        CLASS_COLORS[name] = "#fb7185"
    else:
        CLASS_COLORS[name] = _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]


def heatmap_to_png_b64(heatmap: np.ndarray) -> str:
    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # heatmap is already normalized 0-1
    heatmap_uint8 = (heatmap_colored * 255).astype(np.uint8)
    img = Image.fromarray(heatmap_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_result_for_component(result: dict) -> dict:
    # FIXED - handles your inference.py return format
    # result is like {"label": "trash", "confidence": 0.94, "top3": [...]}
    
    # Get predictions safely
    if "predictions" in result:
        preds = result["predictions"]
    elif "top3" in result:
        # Convert top3 to predictions format
        preds = result["top3"]
    elif "label" in result:
        # Single prediction
        preds = [result]
    else:
        preds = []

    predictions = np.array([preds]) if len(preds) > 0 else np.array([[]])
    
    # Get top classes safely
    try:
        top_classes = get_top_classes(predictions)
    except:
        # Fallback - use top3 directly if available
        if "top3" in result:
            top_classes = [{"class": x.get("label", x.get("class")), "confidence": x.get("confidence", 0)} for x in result["top3"]]
        else:
            top_classes = [{"class": result.get("label", "unknown"), "confidence": result.get("confidence", 0)}]

    top3 = []
    for c in top_classes[:3] if isinstance(top_classes, list) else []:
        if isinstance(c, dict):
            cls = c.get("class") or c.get("label") or "unknown"
            conf = c.get("confidence") or c.get("prob") or 0
            top3.append({"class": cls, "confidence": float(conf)})

    if not top3 and "label" in result:
        top3 = [{"class": result["label"], "confidence": float(result.get("confidence", 0))}]
    
    return {
        "predicted_class": top3[0]["class"] if top3 else "unknown",
        "confidence": top3[0]["confidence"] if top3 else 0,
        "top3": top3,
        "predictions": top3,
        "latency_ms": result.get("latency_ms", 0)
    }


if "sortlight_result" not in st.session_state:
    st.session_state.sortlight_result = None
if "sortlight_last_action_id" not in st.session_state:
    st.session_state.sortlight_last_action_id = None

value = sortlight_ui(
    class_names=class_names,
    class_colors=CLASS_COLORS,
    model_version="ViT (50% unfreeze)",
    result=st.session_state.sortlight_result,
    key="sortlight",
)

if value and value.get("action") == "classify":
    action_id = value.get("action_id")

    if action_id != st.session_state.sortlight_last_action_id:
        st.session_state.sortlight_last_action_id = action_id

        image_bytes = base64.b64decode(value["image_b64"])
        image = Image.open(io.BytesIO(image_bytes))
        show_xai = value.get("show_xai", True)

        with st.spinner("Classifying..."):
            start_time = time.time()
            result = classifier.predict(image, "vit", show_xai)
            result["latency_ms"] = (time.time() - start_time) * 1000

        component_result = build_result_for_component(result)
        component_result["action_id"] = action_id
        st.session_state.sortlight_result = component_result
        st.rerun()
