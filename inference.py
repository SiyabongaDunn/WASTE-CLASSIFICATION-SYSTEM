import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import preprocess_image, class_names
from vit_model import build_vit_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


class WasteClassifier:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all trained models."""
        print("=" * 50)
        print("Loading models...")
        print("=" * 50)

        if not os.path.exists(MODELS_DIR):
            print(f"ERROR: Models directory not found at {MODELS_DIR}")
            return

        print(f"Models directory: {MODELS_DIR}")
        print(f"Files found: {os.listdir(MODELS_DIR)}")

        vit_weights_path = os.path.join(MODELS_DIR, 'vit_weights.weights.h5')
        print(f"\nChecking ViT weights path: {vit_weights_path}")
        print(f"ViT file exists: {os.path.exists(vit_weights_path)}")
        if not vit_weights_path.endswith('.weights.h5'):
            print("WARNING: filename doesn't end in '.weights.h5' -- if this was "
                  "saved with Keras 3's model.save_weights(), rename it to match "
                  "exactly, or loading can silently take a different code path "
                  "than the one that wrote it. See debug_predict.py.")

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
                self.models['vit'] = None
        else:
            print(f"ViT model file not found at: {vit_weights_path}")
            self.models['vit'] = None

        print("\n" + "-" * 50)
        loaded_count = sum(1 for m in self.models.values() if m is not None)
        print(f"Loaded {loaded_count}/1 models (ViT only)")
        for name, model in self.models.items():
            status = "Loaded" if model is not None else "Failed"
            print(f"  {name}: {status}")
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
                            if l.name.endswith("_attn_scores")]      # 12 layers, in block order
            extractor = tf.keras.Model(model.input, attn_tensors + [model.output])
            self._attn_extractor = extractor
        return extractor

    def get_vit_attention(self, model, image, method="chefer"):
        try:
            extractor = self._get_attn_extractor(model)
            image = tf.convert_to_tensor(image, dtype=tf.float32)

            with tf.GradientTape() as tape:
                outputs = extractor(image, training=False)
                attns, logits = outputs[:-1], outputs[-1]          # 12 x (1, H, N, N), (1, C)
                class_idx = tf.argmax(logits[0])
                score = logits[0][class_idx]

            N = attns[0].shape[-1]                                 # 197
            R = tf.eye(N)

            if method == "rollout":
                for a in attns:
                    a = tf.reduce_mean(a[0], axis=0)               # average heads -> (N, N)
                    a = 0.5 * a + 0.5 * tf.eye(N)                  # add residual connection
                    a = a / tf.reduce_sum(a, axis=-1, keepdims=True)
                    R = tf.matmul(a, R)
            else:
                grads = tape.gradient(score, attns)                # d(logit)/d(attention), per layer
                for a, g in zip(attns, grads):
                    cam = tf.reduce_mean(tf.nn.relu(a[0] * g[0]), axis=0)   # (N, N)
                    R = R + tf.matmul(cam, R)

            cls_to_patches = R[0, 1:].numpy()                      # CLS row, drop CLS column -> (196,)
            g = int(np.sqrt(cls_to_patches.shape[0]))              # 14
            heatmap = cls_to_patches.reshape(g, g)

            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            heatmap = np.maximum(heatmap, 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap.astype(np.float32)

        except Exception as e:
            print(f"Attention extraction failed: {e}")
            return None