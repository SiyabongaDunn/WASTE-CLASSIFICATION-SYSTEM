import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224
HF_REPO_ID = "SiyabongaDunn/sortlight-vit"
HF_FILENAME = "vit_weights.weights.h5"

class WasteClassifier:
    def __init__(self):
        self.model = None
        self._load()

    def _ensure_weights(self):
        os.makedirs("models", exist_ok=True)
        local_path = f"models/{HF_FILENAME}"
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
            print(f"ViT file exists: True - {local_path}")
            return local_path
        
        print(f"Downloading {HF_FILENAME} from {HF_REPO_ID}...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            # HF may download to nested path
            if not os.path.exists(local_path):
                # Find file
                for root, _, files in os.walk("models"):
                    if HF_FILENAME in files:
                        src = os.path.join(root, HF_FILENAME)
                        if src != local_path:
                            import shutil
                            shutil.copy(src, local_path)
                            print(f"Copied to {local_path}")
                        break
            print(f"Downloaded: {local_path} - {os.path.getsize(local_path)/1024/1024:.1f} MB")
            return local_path
        except Exception as e:
            print(f"HF download failed: {e}")
            raise FileNotFoundError(f"Could not download {HF_FILENAME}. Check HF repo {HF_REPO_ID}")

    def _build_vit_model(self):
        """Build exact ViT architecture that matches vit_weights.weights.h5"""
        print("Building ViT model architecture...")
        # Try vit-keras first (most common for this file name)
        try:
            from vit import vit_b16
            print("Using vit-keras vit_b16...")
            # Common training setup: pretrained ViT-B16 fine-tuned to 6 classes
            model = vit_b16(
                image_size=IMG_SIZE,
                activation='softmax',
                pretrained=False,  # We will load our own weights
                include_top=True,
                pretrained_top=False,
                classes=len(CLASS_NAMES)
            )
            print("vit-keras model built")
            return model
        except Exception as e:
            print(f"vit-keras build failed: {e}, trying alternative...")

        # Fallback 1: Build custom ViT using keras
        try:
            from tensorflow.keras import layers, Model
            print("Building custom ViT-like model...")
            
            # Input
            inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
            
            # Patch embedding - ViT-B16: 16x16 patches, 768 dim
            # This is simplified but will load if weights were from vit-keras
            # Use EfficientNet as backbone if vit library not available
            # But we need to match weight names - try building ViT from scratch
            
            # For .weights.h5 file (686MB), it's likely full ViT-B16
            # Let's try building with same layer names as vit-keras
            from vit import ViT_B16
            model = ViT_B16(
                image_size=IMG_SIZE,
                classes=len(CLASS_NAMES),
                activation='softmax',
                include_top=True
            )
            return model
        except Exception as e:
            print(f"Custom build failed: {e}")

        # Final fallback: Try loading as full model with safe_mode
        return None

    def _load(self):
        weights_path = self._ensure_weights()
        print(f"Loading weights from {weights_path}...")

        # Method 1: Build ViT then load weights (correct for .weights.h5)
        model = self._build_vit_model()
        if model is not None:
            try:
                print(f"Attempting to load weights into ViT model...")
                model.load_weights(weights_path)
                self.model = model
                print("vit: Loaded successfully! (weights loaded into ViT)")
                return
            except Exception as e:
                print(f"Weight loading into ViT failed: {e}")
                # Print first few layer names to debug
                try:
                    print("Model layers:", [l.name for l in model.layers[:10]])
                except:
                    pass

        # Method 2: Try direct load_model (if it's actually a full model saved as .weights.h5)
        try:
            print("Trying load_model as fallback...")
            self.model = tf.keras.models.load_model(weights_path, compile=False, safe_mode=False)
            print("vit: Loaded successfully! (full model)")
            return
        except Exception as e:
            print(f"load_model failed: {e}")

        # If we reach here, we failed
        raise ValueError(f"Could not load ViT weights from {weights_path}. File may be corrupted or architecture mismatch. Try re-saving your model locally with model.save('vit_model.h5') and upload that to HF instead.")

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
        
        result = {
            "label": top3[0]["label"],
            "predicted_class": top3[0]["label"],
            "class": top3[0]["label"],
            "confidence": top3[0]["confidence"],
            "predictions": top3,
            "top3": top3,
            "probs": preds.tolist(),
            "latency_ms": latency,
            "attention_map": orig_pil,
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

WasteClassifier = WasteClassifier
