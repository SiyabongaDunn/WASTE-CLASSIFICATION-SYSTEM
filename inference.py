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
        """Run prediction on an image."""
        processed_image = preprocess_image(image)

        model = self.models.get(model_name)
        if model is None:
            available = [k for k, v in self.models.items() if v is not None]
            raise ValueError(
                f"Model '{model_name}' not loaded. "
                f"Available models: {available}"
            )

        logits = model.predict(processed_image, verbose=0)[0]

        # head_out has no activation -- these are raw logits, not probabilities.
        # Softmax them before treating any value as a "confidence". This does
        # NOT change which class wins (argmax is monotonic under softmax), but
        # it fixes every displayed confidence/percentage number, which were
        # previously raw unbounded logits.
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

    def get_vit_attention(self, model, image):
        """Extract ViT attention map."""
        block_layers = [l for l in model.layers if l.name.startswith("block_")]
        if block_layers:
            target_layer = block_layers[-1]
        else:
            return None

        try:
            with tf.GradientTape() as tape:
                grad_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=[model.get_layer(target_layer.name).output, model.output]
                )
                conv_outputs, predictions = grad_model(image)
                class_idx = np.argmax(predictions[0])
                loss = predictions[0][class_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
            heatmap = heatmap.numpy()
            heatmap = cv2.resize(heatmap, (224, 224))
            return heatmap
        except Exception as e:
            print(f"Attention extraction failed: {e}")
            return None

    def get_grad_cam(self, model, image, class_idx):
        """Generate Grad-CAM heatmap."""
        return None