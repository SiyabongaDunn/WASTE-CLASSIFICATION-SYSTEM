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
    """
    FINAL FIXED VERSION - works with new inference.py
    result contains: label, confidence, predictions/top3, latency_ms, attention_map
    """
    # result["predictions"] now exists!
    if "predictions" in result:
        top3_raw = result["predictions"]
    elif "top3" in result:
        top3_raw = result["top3"]
    else:
        top3_raw = [{"class": result.get("label", "unknown"), "confidence": result.get("confidence", 0)}]

    # Normalize to [{"class": ..., "confidence": ...}]
    top3 = []
    for item in top3_raw[:3]:
        if isinstance(item, dict):
            cls = item.get("class") or item.get("label") or "unknown"
            conf = item.get("confidence") or item.get("prob") or 0.0
            top3.append({"class": str(cls), "confidence": float(conf)})

    if not top3:
        top3 = [{"class": "unknown", "confidence": 0.0}]

    return {
        "predicted_class": top3[0]["class"],
        "confidence": top3[0]["confidence"],
        "top3": top3,
        "predictions": top3,
        "latency_ms": result.get("latency_ms", 0),
        "attention_map": result.get("attention_map"),
        "original": result.get("original")
    }
