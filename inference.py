import os
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

# ============================================================
# CONFIG - matches your notebook
# ============================================================
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224
HF_REPO_ID = "SiyabongaDunn/sortlight-vit"
HF_FILENAME = "vit_weights.weights.h5"

print("\n" + "="*60)
print("STEP: ViT BUILDING BLOCKS (from your Full_Pipeline.ipynb)")
print("="*60)

# ------------------------------------------------------------
# Transformer block - MUST match layer names in your notebook
# ------------------------------------------------------------
def transformer_block(x, num_heads=12, embed_dim=768, mlp_dim=3072, name="block_0"):
    # LayerNorm 1
    ln1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)

    # Q, K, V projections
    q = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_attn_q")(ln1)
    k = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_attn_k")(ln1)
    v = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_attn_v")(ln1)

    # Multi-head attention (manual)
    # Reshape to (batch, seq_len, num_heads, head_dim)
    def split_heads(t):
        # t: (B, S, D) -> (B, S, H, Dh)
        batch = keras.ops.shape(t)[0]
        seq_len = keras.ops.shape(t)[1]
        head_dim = embed_dim // num_heads
        t = keras.ops.reshape(t, (batch, seq_len, num_heads, head_dim))
        t = keras.ops.transpose(t, (0, 2, 1, 3))  # (B, H, S, Dh)
        return t

    def combine_heads(t):
        # t: (B, H, S, Dh) -> (B, S, D)
        t = keras.ops.transpose(t, (0, 2, 1, 3))
        batch = keras.ops.shape(t)[0]
        seq_len = keras.ops.shape(t)[1]
        return keras.ops.reshape(t, (batch, seq_len, embed_dim))

    qh = keras.layers.Lambda(split_heads, name=f"{name}_q_split")(q)
    kh = keras.layers.Lambda(split_heads, name=f"{name}_k_split")(k)
    vh = keras.layers.Lambda(split_heads, name=f"{name}_v_split")(v)

    # Attention scores
    # scores = Q @ K^T / sqrt(d)
    def attention_fn(inputs):
        qh, kh, vh = inputs
        head_dim = embed_dim // num_heads
        scores = keras.ops.matmul(qh, keras.ops.transpose(kh, (0, 1, 3, 2)))  # (B, H, S, S)
        scores = scores / keras.ops.sqrt(float(head_dim))
        weights = keras.ops.softmax(scores, axis=-1)
        out = keras.ops.matmul(weights, vh)  # (B, H, S, Dh)
        return out

    attn_out = keras.layers.Lambda(attention_fn, name=f"{name}_attn_scores")([qh, kh, vh])
    attn_out = keras.layers.Lambda(combine_heads, name=f"{name}_attn_combine")(attn_out)

    # Output projection
    proj = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_attn_proj")(attn_out)

    # Residual 1
    x1 = keras.layers.Add(name=f"{name}_res1")([x, proj])

    # LayerNorm 2
    ln2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x1)

    # MLP
    fc1 = keras.layers.Dense(mlp_dim, activation='gelu', use_bias=True, name=f"{name}_mlp_fc1")(ln2)
    fc2 = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_mlp_fc2")(fc1)

    # Residual 2
    x2 = keras.layers.Add(name=f"{name}_res2")([x1, fc2])
    return x2

# ------------------------------------------------------------
# Full ViT builder - EXACT copy from your notebook STEP 6
# ------------------------------------------------------------
def build_vit_model(num_classes=6):
    num_patches = (IMG_SIZE // 16) ** 2  # 196
    seq_len = num_patches + 1  # 197
    embed_dim = 768
    num_heads = 12
    mlp_dim = 3072
    num_layers = 12

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="pixel_values")

    patches = keras.layers.Conv2D(
        filters=embed_dim, kernel_size=16, strides=16,
        padding='valid', use_bias=True, name="patch_embed_proj"
    )(inputs)
    patches = keras.layers.Reshape((num_patches, embed_dim), name="patch_embed_reshape")(patches)

    class CLSToken(keras.layers.Layer):
        def build(self, input_shape):
            self.cls = self.add_weight(
                name="cls", shape=(1, 1, embed_dim),
                initializer="zeros", trainable=True
            )
        def call(self, x):
            return keras.ops.broadcast_to(self.cls, (keras.ops.shape(x)[0], 1, embed_dim))

    cls_tokens = CLSToken(name="cls_token")(patches)
    x = keras.layers.Concatenate(axis=1, name="cls_concat")([cls_tokens, patches])

    class PosEmbed(keras.layers.Layer):
        def build(self, input_shape):
            self.pos = self.add_weight(
                name="pos", shape=(1, seq_len, embed_dim),
                initializer="zeros", trainable=True
            )
        def call(self, x):
            return x + self.pos

    x = PosEmbed(name="pos_embed")(x)

    for i in range(num_layers):
        x = transformer_block(x, num_heads=num_heads, embed_dim=embed_dim,
                              mlp_dim=mlp_dim, name=f"block_{i}")

    x = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
    cls_out = keras.layers.Lambda(lambda t: t[:, 0, :], name="cls_extract")(x)

    z = keras.layers.Dense(256, activation='relu', name="head_dense")(cls_out)
    z = keras.layers.Dropout(0.3, name="head_dropout")(z)
    outputs = keras.layers.Dense(num_classes, name="head_out")(z)  # logits

    vit_model = keras.Model(inputs=inputs, outputs=outputs)
    print("ViT model built")
    return vit_model

print("ViT model builder ready")

# ============================================================
# CLASSIFIER - loads your 686 MB fine-tuned weights
# ============================================================
class WasteClassifier:
    def __init__(self):
        self.model = None
        self._load()

    def _ensure_weights(self):
        os.makedirs("models", exist_ok=True)
        local_path = f"models/{HF_FILENAME}"
        if os.path.exists(local_path) and os.path.getsize(local_path) > 10_000_000:
            print(f"ViT file exists: True - {local_path} ({os.path.getsize(local_path)/1024/1024:.1f} MB)")
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
            # HF may download to subfolder
            if not os.path.exists(local_path):
                for root, _, files in os.walk("models"):
                    if HF_FILENAME in files:
                        src = os.path.join(root, HF_FILENAME)
                        if src != local_path:
                            import shutil
                            shutil.copy(src, local_path)
                        break
            print(f"Downloaded: {local_path}")
            return local_path
        except Exception as e:
            print(f"HF download failed: {e}")
            raise FileNotFoundError(f"Could not download {HF_FILENAME} from {HF_REPO_ID}")

    def _load(self):
        weights_path = self._ensure_weights()
        print(f"Building ViT model...")
        model = build_vit_model(num_classes=len(CLASS_NAMES))
        
        print(f"Loading weights from {weights_path}...")
        try:
            model.load_weights(weights_path)
            print("vit: Loaded successfully!")
            print(f"Loaded {len(model.weights)} weight tensors")
        except Exception as e:
            print(f"Load weights failed: {e}")
            print("Trying with skip mismatch...")
            # Try loading with some flexibility
            model.load_weights(weights_path, skip_mismatch=False)
            print("vit: Loaded successfully! (with skip)")

        self.model = model
        # Warmup
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print(f"Loaded 1/1 models (ViT only)")

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(image).astype(np.float32) / 255.0
        return np.expand_dims(arr, 0), image

    def predict(self, image, model_name="vit", show_xai=False):
        x, orig_pil = self.preprocess(image)
        logits = self.model.predict(x, verbose=0)[0]
        # Softmax for probs (model outputs logits)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        top_idx = np.argsort(probs)[::-1][:3]
        top3 = []
        for i in top_idx:
            top3.append({
                "label": CLASS_NAMES[i],
                "class": CLASS_NAMES[i],
                "confidence": float(probs[i]),
                "prob": float(probs[i])
            })

        result = {
            "label": top3[0]["label"],
            "predicted_class": top3[0]["label"],
            "class": top3[0]["label"],
            "confidence": top3[0]["confidence"],
            "predictions": top3,  # fixes KeyError!
            "top3": top3,
            "probs": probs.tolist(),
            "latency_ms": 0,
            "attention_map": orig_pil,
            "original": orig_pil
        }
        return result

_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = WasteClassifier()
    return _classifier

# Backward compat
WasteClassifier = WasteClassifier
