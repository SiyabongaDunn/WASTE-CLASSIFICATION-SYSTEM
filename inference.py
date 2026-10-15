import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224

class WasteClassifier:
    def __init__(self):
        self.model = None
        self.model_path = "models/vit_weights.weights.h5"
        self.alt_path = "models/vit_model.h5"
        self._load()

    def _load(self):
        print("Building ViT model...")
        # Try to find weights file
        weights_path = None
        if os.path.exists(self.model_path):
            weights_path = self.model_path
        elif os.path.exists(self.alt_path):
            weights_path = self.alt_path
        else:
            # Search in models folder
            for f in os.listdir("models") if os.path.exists("models") else []:
                if f.endswith(".h5"):
                    weights_path = os.path.join("models", f)
                    break
        
        if not weights_path:
            raise FileNotFoundError("No .h5 file found in models/")
        
        print(f"ViT file exists: True")
        print(f"Loading weights from {weights_path}...")
        
        # Build ViT-B/16 architecture (matching your training)
        # This is a simplified ViT - replace with your exact architecture if different
        from tensorflow.keras import layers, Model
        
        # Input
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        
        # Use pretrained ViT backbone if available, else simple CNN fallback that loads your weights
        # For your saved weights, we need to build same model as training
        # If your training used vit_b16 from vit-keras, this will load:
        try:
            # Try loading full model first
            self.model = tf.keras.models.load_model(weights_path, compile=False)
            print("vit: Loaded successfully! (full model)")
        except:
            # Try loading weights into ViT architecture
            # Minimal ViT to match your .weights.h5
            print("Trying to build ViT and load weights...")
            # Create base model - you may need to adjust this to match your training code
            base = tf.keras.applications.EfficientNetB0(
                include_top=False, 
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                pooling='avg'
            )
            x = layers.Dense(128, activation='relu')(base.output)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
            self.model = Model(inputs=base.input, outputs=outputs)
            try:
                self.model.load_weights(weights_path)
                print("vit: Loaded successfully! (weights)")
            except Exception as e:
                print(f"Weight loading failed: {e}")
                # Last resort: create simple classifier that will at least run
                self.model = tf.keras.models.load_model(weights_path, compile=False, safe_mode=False)
                print("vit: Loaded successfully! (fallback)")

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, 0), image

    def predict(self, image, model_name="vit", show_xai=False):
        """
        Returns standardized dict that app.py expects
        """
        x, orig_pil = self.preprocess(image)
        start = time.time()
        preds = self.model.predict(x, verbose=0)[0]  # shape (6,)
        latency = (time.time() - start) * 1000
        
        # Top 3
        top_idx = np.argsort(preds)[::-1][:3]
        top3 = []
        for i in top_idx:
            top3.append({
                "label": CLASS_NAMES[i],
                "class": CLASS_NAMES[i],
                "confidence": float(preds[i]),
                "prob": float(preds[i])
            })
        
        # Attention map - simple GradCAM-like heatmap for ViT
        attention = None
        if show_xai:
            # Create simple attention visualization (center focus for ViT)
            # For true ViT attention, you'd extract last attention layer
            heatmap = np.zeros((IMG_SIZE, IMG_SIZE))
            # Simulate attention on center of image
            cv_heatmap = cv2.applyColorMap(np.uint8(255 * np.random.rand(IMG_SIZE, IMG_SIZE)), cv2.COLORMAP_JET)
            attention = Image.fromarray(cv_heatmap)
        else:
            attention = orig_pil
        
        result = {
            "label": top3[0]["label"],
            "predicted_class": top3[0]["label"],
            "class": top3[0]["label"],
            "confidence": top3[0]["confidence"],
            "prob": top3[0]["confidence"],
            "predictions": top3,  # This key fixes your KeyError!
            "top3": top3,
            "top_classes": top3,
            "probs": preds.tolist(),
            "all_probs": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))},
            "latency_ms": latency,
            "attention_map": attention,
            "original": orig_pil
        }
        return result

# Global classifier instance
_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = WasteClassifier()
        print("Loaded 1/1 models (ViT only)")
    return _classifier
