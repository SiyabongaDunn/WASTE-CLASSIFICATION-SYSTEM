import os
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

# ============================================================
# CONFIG - NOW USING FP16 VERSION (343 MB instead of 686 MB)
# ============================================================
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224
HF_REPO_ID = "SiyabongaDunn/sortlight-vit"
HF_FILENAME_FP16 = "vit_weights_fp16.weights.h5"  # NEW 343 MB FILE
HF_FILENAME_ORIG = "vit_weights.weights.h5"       # fallback

print("\n" + "="*60)
print("STEP 5: ViT BUILDING BLOCKS - EXACT COPY")
print("="*60)

def mlp_block(x, hidden_dim, out_dim, name):
    x = keras.layers.Dense(hidden_dim, activation='gelu', name=f"{name}_fc1")(x)
    x = keras.layers.Dense(out_dim, name=f"{name}_fc2")(x)
    return x

def attention_block(x, num_heads, head_dim, embed_dim, name):
    q = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_q")(x)
    k = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_k")(x)
    v = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_v")(x)
    q = keras.layers.Reshape((-1, num_heads, head_dim), name=f"{name}_q_reshape")(q)
    k = keras.layers.Reshape((-1, num_heads, head_dim), name=f"{name}_k_reshape")(k)
    v = keras.layers.Reshape((-1, num_heads, head_dim), name=f"{name}_v_reshape")(v)
    q = keras.layers.Permute((2, 1, 3), name=f"{name}_q_perm")(q)
    k = keras.layers.Permute((2, 1, 3), name=f"{name}_k_perm")(k)
    v = keras.layers.Permute((2, 1, 3), name=f"{name}_v_perm")(v)
    scale = head_dim ** -0.5
    attn = keras.layers.Lambda(lambda qk: keras.ops.softmax(keras.ops.matmul(qk[0], keras.ops.transpose(qk[1], (0,1,3,2))) * scale, axis=-1), name=f"{name}_attn_scores")([q, k])
    out = keras.layers.Lambda(lambda av: keras.ops.matmul(av[0], av[1]), name=f"{name}_attn_out")([attn, v])
    out = keras.layers.Permute((2, 1, 3), name=f"{name}_out_perm")(out)
    out = keras.layers.Reshape((-1, embed_dim), name=f"{name}_out_reshape")(out)
    out = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_proj")(out)
    return out

def transformer_block(x, num_heads=12, embed_dim=768, mlp_dim=3072, name="block"):
    y = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    y = attention_block(y, num_heads, head_dim=embed_dim//num_heads, embed_dim=embed_dim, name=f"{name}_attn")
    x = keras.layers.Add(name=f"{name}_add1")([x, y])
    y = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    y = mlp_block(y, hidden_dim=mlp_dim, out_dim=embed_dim, name=f"{name}_mlp")
    x = keras.layers.Add(name=f"{name}_add2")([x, y])
    return x

def build_vit_model(num_classes=6):
    num_patches = (IMG_SIZE // 16) ** 2
    seq_len = num_patches + 1
    embed_dim = 768
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="pixel_values")
    patches = keras.layers.Conv2D(filters=embed_dim, kernel_size=16, strides=16, padding='valid', use_bias=True, name="patch_embed_proj")(inputs)
    patches = keras.layers.Reshape((num_patches, embed_dim), name="patch_embed_reshape")(patches)
    class CLSToken(keras.layers.Layer):
        def build(self, input_shape): self.cls = self.add_weight(name="cls", shape=(1,1,embed_dim), initializer="zeros", trainable=True)
        def call(self, x): return keras.ops.broadcast_to(self.cls, (keras.ops.shape(x)[0], 1, embed_dim))
    class PosEmbed(keras.layers.Layer):
        def build(self, input_shape): self.pos = self.add_weight(name="pos", shape=(1,seq_len,embed_dim), initializer="zeros", trainable=True)
        def call(self, x): return x + self.pos
    cls_tokens = CLSToken(name="cls_token")(patches)
    x = keras.layers.Concatenate(axis=1, name="cls_concat")([cls_tokens, patches])
    x = PosEmbed(name="pos_embed")(x)
    for i in range(12): x = transformer_block(x, name=f"block_{i}")
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
    cls_out = keras.layers.Lambda(lambda t: t[:,0,:], name="cls_extract")(x)
    z = keras.layers.Dense(256, activation='relu', name="head_dense")(cls_out)
    z = keras.layers.Dropout(0.3, name="head_dropout")(z)
    outputs = keras.layers.Dense(num_classes, name="head_out")(z)
    return keras.Model(inputs=inputs, outputs=outputs)

print("ViT builder ready")

class WasteClassifier:
    def __init__(self):
        self.model = None
        self._load()

    def _ensure_weights(self):
        os.makedirs("models", exist_ok=True)
        # Try FP16 first (343 MB), then original
        for fname in [HF_FILENAME_FP16, HF_FILENAME_ORIG]:
            local_path = f"models/{fname}"
            if os.path.exists(local_path) and os.path.getsize(local_path) > 10_000_000:
                print(f"Found {fname}: {os.path.getsize(local_path)/1024/1024:.1f} MB")
                return local_path
        
        # Download - prefer FP16
        print(f"Trying to download {HF_FILENAME_FP16} (343 MB)...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME_FP16, local_dir="models", local_dir_use_symlinks=False)
            # Handle nested download
            final_path = f"models/{HF_FILENAME_FP16}"
            if not os.path.exists(final_path):
                for root,_,files in os.walk("models"):
                    if HF_FILENAME_FP16 in files:
                        import shutil
                        shutil.copy(os.path.join(root, HF_FILENAME_FP16), final_path)
                        break
            print(f"Downloaded FP16: {final_path}")
            return final_path
        except Exception as e:
            print(f"FP16 not found ({e}), downloading original 686 MB...")
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME_ORIG, local_dir="models", local_dir_use_symlinks=False)
            final_path = f"models/{HF_FILENAME_ORIG}"
            if not os.path.exists(final_path):
                for root,_,files in os.walk("models"):
                    if HF_FILENAME_ORIG in files:
                        import shutil
                        shutil.copy(os.path.join(root, HF_FILENAME_ORIG), final_path)
                        break
            return final_path

    def _load(self):
        weights_path = self._ensure_weights()
        print(f"Building ViT...")
        model = build_vit_model(num_classes=len(CLASS_NAMES))
        print(f"Loading weights from {weights_path} ({os.path.getsize(weights_path)/1024/1024:.1f} MB)...")
        model.load_weights(weights_path)
        print("vit: Loaded successfully!")
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        self.model = model
        print("Loaded 1/1 models")

    def preprocess(self, image):
        if isinstance(image, np.ndarray): image = Image.fromarray(image)
        image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(image).astype(np.float32) / 255.0
        return np.expand_dims(arr, 0), image

    def predict(self, image, model_name="vit", show_xai=False):
        x, orig = self.preprocess(image)
        import time; s=time.time()
        logits = self.model.predict(x, verbose=0)[0]
        latency = (time.time()-s)*1000
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [{"label": CLASS_NAMES[i], "class": CLASS_NAMES[i], "confidence": float(probs[i]), "prob": float(probs[i])} for i in top_idx]
        return {"label": top3[0]["label"], "predicted_class": top3[0]["label"], "class": top3[0]["label"], "confidence": top3[0]["confidence"], "predictions": top3, "top3": top3, "probs": probs.tolist(), "latency_ms": latency, "attention_map": orig, "original": orig}

_classifier=None
def get_classifier():
    global _classifier
    if _classifier is None: _classifier = WasteClassifier()
    return _classifier

WasteClassifier=WasteClassifier
