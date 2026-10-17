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
    predictions = np.array([result["predictions"]])
    top_classes = get_top_classes(predictions)  # [{'class': ..., 'confidence': 0-1}, ...]

    top3 = [
        {"key": item["class"], "conf": item["confidence"] * 100}
        for item in top_classes
    ]

    payload = {
        "primary": result["class"],
        "confidence": result["confidence"] * 100,
        "top3": top3,
        "interpretation": (
            f"Predicted {result['class']} with "
            f"{result['confidence'] * 100:.1f}% confidence."
        ),
        "latency_ms": result.get("latency_ms"),
    }

    xai_image = result.get("xai_image")
    if xai_image is not None:
        payload["attention_image_b64"] = heatmap_to_png_b64(xai_image)

    return payload

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