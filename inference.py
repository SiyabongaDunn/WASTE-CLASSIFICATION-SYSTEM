import os
try:
    import gdown
except ImportError:
    gdown = None

import tensorflow as tf
import numpy as np
import cv2
from utils import preprocess_image, class_names
from vit_model import build_vit_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'vit_weights.weights.h5')
FILE_ID = "1G6fg3M7USlyPf8ACqxkcoRHv7g8r8Erl"

def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        # If file is < 50MB, it's the HTML virus warning page, delete it
        size = os.path.getsize(MODEL_PATH)
        if size < 50_000_000:
            print(f"Bad file detected (size {size} bytes), deleting and re-downloading...")
            os.remove(MODEL_PATH)
    
    if not os.path.exists(MODEL_PATH):
        if gdown:
            os.makedirs(MODELS_DIR, exist_ok=True)
            print(f"Downloading real model to {MODEL_PATH}...")
            # This format with id= + fuzzy works for 330MB files
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
            print(f"Downloaded size: {os.path.getsize(MODEL_PATH)} bytes")
        else:
            raise FileNotFoundError(f"Model not found")
class WasteClassifier:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all trained models."""
        download_model_if_needed()

        print("=" * 50)
        print("Loading models...")
        print("=" * 50)

        if not os.path.exists(MODELS_DIR):
            print(f"ERROR: Models directory not found at {MODELS_DIR}")
            return

        print(f"Models directory: {MODELS_DIR}")
        print(f"Files found: {os.listdir(MODELS_DIR)}")

        vit_weights_path = MODEL_PATH
        print(f"\nChecking ViT weights path: {vit_weights_path}")
        print(f"ViT file exists: {os.path.exists(vit_weights_path)}")

        if os.path.exists(vit_weights_path):
            try:
                print("\nBuilding ViT architecture...")
                self.models['vit'] = build_vit_model(num_classes=6)

                print("Loading ViT weights...")
                self.models['vit'].load_weights(vit_weights_path)
                print("ViT loaded successfully!")

                dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
                test_output = self.models['vit'].predict(dummy_input, verbose=0)
                print(f"ViT test output shape: {test_output.shape}")

            except Exception as e:
                print(f"ViT load failed: {e}")
                import traceback
                traceback.print_exc()
                self.models['vit'] = None
        else:
            print(f"ViT model file not found at: {vit_weights_path}")
            self.models['vit'] = None

        print("\n" + "-" * 50)
        loaded_count = sum(1 for m in self.models.values() if m is not None)
        print(f"Loaded {loaded_count}/1 models (ViT only)")
        for name, model in self.models.items():
            status = "Loaded" if model is not None else "Failed"
            print(f" {name}: {status}")
        print("=" * 50)

    def predict(self, image, model_name='vit', show_xai=True):
        processed_image = preprocess_image(image)

        model = self.models.get(model_name)
        if model is None:
            available = [k for k, v in self.models.items() if v is not None]
            raise ValueError(
                f"Model '{model_name}' not loaded. "
                f"Available models: {available}"
            )

        logits = model.predict(processed_image, verbose=0)[0]

        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        predictions = exp / exp.sum()

        pred_class = int(np.argmax(predictions))
        confidence = float(predictions[pred_class])

        result = {
            'class': class_names[pred_class],
            'class_idx': pred_class,
            'confidence': confidence,
            'predictions': predictions.tolist(),
            'model': model_name
        }

        if show_xai:
            if model_name == 'vit':
                xai_image = self.get_vit_attention(model, processed_image)
                result['xai_type'] = 'attention'
            else:
                xai_image = self.get_grad_cam(model, processed_image, pred_class)
                result['xai_type'] = 'gradcam'
            result['xai_image'] = xai_image

        return result

    def _get_attn_extractor(self, model):
        """Build (once) a model that outputs all 12 attention maps plus the logits."""
        extractor = getattr(self, "_attn_extractor", None)
        if extractor is None:
            attn_tensors = [l.output for l in model.layers
                            if l.name.endswith("_attn_scores")]
            extractor = tf.keras.Model(model.input, attn_tensors + [model.output])
            self._attn_extractor = extractor
        return extractor

    def get_vit_attention(self, model, image, method="chefer"):
        try:
            extractor = self._get_attn_extractor(model)
            image = tf.convert_to_tensor(image, dtype=tf.float32)

            with tf.GradientTape() as tape:
                outputs = extractor(image, training=False)
                attns, logits = outputs[:-1], outputs[-1]
                class_idx = tf.argmax(logits[0])
                score = logits[0][class_idx]

            N = attns[0].shape[-1]
            R = tf.eye(N)

            if method == "rollout":
                for a in attns:
                    a = tf.reduce_mean(a[0], axis=0)
                    a = 0.5 * a + 0.5 * tf.eye(N)
                    a = a / tf.reduce_sum(a, axis=-1, keepdims=True)
                    R = tf.matmul(a, R)
            else:
                grads = tape.gradient(score, attns)
                for a, g in zip(attns, grads):
                    cam = tf.reduce_mean(tf.nn.relu(a[0] * g[0]), axis=0)
                    R = R + tf.matmul(cam, R)

            cls_to_patches = R[0, 1:].numpy()
            g = int(np.sqrt(cls_to_patches.shape[0]))
            heatmap = cls_to_patches.reshape(g, g)

            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            heatmap = np.maximum(heatmap, 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap.astype(np.float32)

        except Exception as e:
            print(f"Attention extraction failed: {e}")
            return None

    def get_grad_cam(self, model, image, class_idx):
        # kept for compatibility if you add CNN models later
        try:
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break
            if last_conv is None:
                return None
            grad_model = tf.keras.Model(
                [model.inputs], [model.get_layer(last_conv).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, class_idx]
            grads = tape.gradient(loss, conv_outputs)[0]
            weights = tf.reduce_mean(grads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1).numpy()
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.astype(np.float32)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            return None
