import os
import shutil
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
import cv2
from utils import preprocess_image, class_names
from vit_model import build_vit_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'vit_weights.weights.h5')
HF_REPO = "SiyabongaDunn/sortlight-vit"

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        try:
            size = os.path.getsize(MODEL_PATH)
            if size > 1000000:
                print(f"Model already exists: {size} bytes")
                return
        except:
            pass
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading model from {HF_REPO}...")
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="vit_weights.weights.h5",
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )
    if downloaded_path != MODEL_PATH:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        shutil.move(downloaded_path, MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        print(f"Model ready: {os.path.getsize(MODEL_PATH)} bytes")
    else:
        print("ERROR: Download failed")

download_model_if_needed()

class WasteClassifier:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        print("=" * 50)
        print("Loading models...")
        print(f"Models directory: {MODELS_DIR}")
        if os.path.exists(MODELS_DIR):
            try:
                print(f"Files found: {os.listdir(MODELS_DIR)}")
            except:
                pass
        vit_weights_path = MODEL_PATH
        print(f"Checking ViT weights path: {vit_weights_path}")
        print(f"ViT file exists: {os.path.exists(vit_weights_path)}")
        if not os.path.exists(vit_weights_path):
            print(f"ViT model file not found at: {vit_weights_path}")
            print("Loaded 0/1 models (ViT only)")
            return
        try:
            print("Building ViT model...")
            model = build_vit_model()
            print(f"Loading weights from {vit_weights_path}...")
            model.load_weights(vit_weights_path)
            self.models['vit'] = model
            print("vit: Loaded successfully!")
            print(f"Loaded {len(self.models)}/1 models (ViT only)")
        except Exception as e:
            print(f"vit: Failed - {e}")
            import traceback
            traceback.print_exc()
        print("=" * 50)

    def predict(self, image, model_name='vit', show_xai=True):
        processed_image = preprocess_image(image)
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")
        preds = model.predict(processed_image, verbose=0)[0]
        top_indices = np.argsort(preds)[::-1][:3]
        top_classes = [(class_names[i], float(preds[i])) for i in top_indices]
        result = {
            "prediction": top_classes[0][0],
            "confidence": float(top_classes[0][1]),
            "top_3": top_classes,
            "probs": preds
        }
        return result
