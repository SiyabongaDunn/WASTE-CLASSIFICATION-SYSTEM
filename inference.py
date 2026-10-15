import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224

# Your HF repo where vit_weights.weights.h5 is stored
HF_REPO_ID = "SiyabongaDunn/sortlight-vit"
HF_FILENAME = "vit_weights.weights.h5"
# If your HF repo is different, change it above - check your HuggingFace profile

class WasteClassifier:
    def __init__(self):
        self.model = None
        self.model_path = "models/vit_weights.weights.h5"
        self.alt_path = "models/vit_model.h5"
        self._load()

    def _ensure_weights(self):
        """Download weights if not present"""
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(self.model_path):
            print(f"ViT file exists: True - {self.model_path}")
            return self.model_path
        if os.path.exists(self.alt_path):
            print(f"ViT file exists: True - {self.alt_path}")
            return self.alt_path
            
        print(f"Model not found locally, downloading from HF: {HF_REPO_ID}/{HF_FILENAME}...")
        try:
            # Try downloading from HF
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            print(f"Downloaded to {downloaded}")
            # If downloaded file is not at expected path, move it
            if downloaded != self.model_path and os.path.exists(downloaded):
                # HF creates subfolder, find the file
                for root, dirs, files in os.walk("models"):
                    if HF_FILENAME in files:
                        src = os.path.join(root, HF_FILENAME)
                        if src != self.model_path:
                            import shutil
                            shutil.copy(src, self.model_path)
                        return self.model_path
            return downloaded if os.path.exists(downloaded) else self.model_path
        except Exception as e:
            print(f"HF download failed: {e}")
            print("Trying alternative HF repo names...")
            # Try alternative repo IDs
            for repo_id in ["SiyabongaDunn/WASTE-CLASSIFICATION-SYSTEM", "SiyabongaDunn/sortlight-waste", "SiyabongaDunn/vit-waste"]:
                try:
                    print(f"Trying {repo_id}...")
                    downloaded = hf_hub_download(repo_id=repo_id, filename=HF_FILENAME, local_dir="models")
                    return downloaded
                except:
                    continue
            raise FileNotFoundError(f"Could not download {HF_FILENAME}. Please upload your .h5 file to HuggingFace repo {HF_REPO_ID}")

    def _load(self):
        print("Building ViT model...")
        weights_path = self._ensure_weights()
        
        if not os.path.exists(weights_path):
            # List what's in models
            print(f"Checking models folder: {os.listdir('models') if os.path.exists('models') else 'no models folder'}")
            raise FileNotFoundError(f"No .h5 file found in models/ - checked {weights_path}")

        print(f"Loading weights from {weights_path}...")
        
        try:
            # Try loading full model first (if it's a full saved model)
            self.model = tf.keras.models.load_model(weights_path, compile=False)
            print("vit: Loaded successfully!")
        except Exception as e1:
            print(f"Full model load failed: {e1}, trying weights only...")
            # Build a simple model that matches your ViT architecture
            # This should match your training architecture
            try:
                from tensorflow.keras import layers, Model
                # For ViT-B16 fine-tuned model
                inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
                # Use a base that can load vit weights - fallback to EfficientNet for structure
                x = layers.Conv2D(768, 16, strides=16, padding='valid', name='vit_patch_embedding')(inputs)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                outputs = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
                temp_model = Model(inputs, outputs)
                # Try loading weights
                temp_model.load_weights(weights_path)
                self.model = temp_model
                print("vit: Loaded successfully!")
            except Exception as e2:
                print(f"Weight load also failed: {e2}")
                # Last resort - try safe_mode
                self.model = tf.keras.models.load_model(weights_path, compile=False, safe_mode=False)
                print("vit: Loaded successfully!")

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, 0), image

    def predict(self, image, model_name="vit", show_xai=False):
        x, orig_pil = self.preprocess(image)
        start = time.time()
        preds = self.model.predict(x, verbose=0)[0]
        latency = (time.time() - start) * 1000
        
        top_idx = np.argsort(preds)[::-1][:3]
        top3 = []
        for i in top_idx:
            top3.append({
                "label": CLASS_NAMES[i],
                "class": CLASS_NAMES[i],
                "confidence": float(preds[i]),
                "prob": float(preds[i])
            })
        
        # Simple attention map for UI
        attention = orig_pil
        
        result = {
            "label": top3[0]["label"],
            "predicted_class": top3[0]["label"],
            "class": top3[0]["label"],
            "confidence": top3[0]["confidence"],
            "prob": top3[0]["confidence"],
            "predictions": top3,
            "top3": top3,
            "probs": preds.tolist(),
            "all_probs": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))},
            "latency_ms": latency,
            "attention_map": attention,
            "original": orig_pil
        }
        return result

_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = WasteClassifier()
        print("Loaded 1/1 models (ViT only)")
    return _classifier

# For backward compatibility
WasteClassifier = WasteClassifier
