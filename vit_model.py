import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py

IMG_SIZE = 224

class CLSToken(keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.broadcast_to(self.cls, (batch_size, 1, self.embed_dim))
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

class PosEmbed(keras.layers.Layer):
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos",
            shape=(1, self.seq_len, self.embed_dim),
            initializer="zeros",
            trainable=True
        )
    
    def call(self, x):
        return x + self.pos
    
    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "embed_dim": self.embed_dim})
        return config

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

    scale = tf.sqrt(tf.cast(head_dim, tf.float32))
    scale = 1.0 / scale

    def attention_fn(inputs):
        q_val, k_val = inputs
        attn = tf.matmul(q_val, tf.transpose(k_val, (0, 1, 3, 2)))
        attn = attn * scale
        attn = tf.nn.softmax(attn, axis=-1)
        return attn

    attn = keras.layers.Lambda(attention_fn, name=f"{name}_attn_scores")([q, k])

    def attn_out_fn(inputs):
        attn_val, v_val = inputs
        return tf.matmul(attn_val, v_val)

    out = keras.layers.Lambda(attn_out_fn, name=f"{name}_attn_out")([attn, v])

    out = keras.layers.Permute((2, 1, 3), name=f"{name}_out_perm")(out)
    out = keras.layers.Reshape((-1, embed_dim), name=f"{name}_out_reshape")(out)
    out = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_proj")(out)
    return out

def transformer_block(x, num_heads=12, embed_dim=768, mlp_dim=3072, name="block"):
    y = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    y = attention_block(y, num_heads=num_heads, head_dim=embed_dim // num_heads,
                        embed_dim=embed_dim, name=f"{name}_attn")
    x = keras.layers.Add()([x, y])
    y = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    y = mlp_block(y, hidden_dim=mlp_dim, out_dim=embed_dim, name=f"{name}_mlp")
    x = keras.layers.Add()([x, y])
    return x

def build_vit_model(num_classes=6):
    num_patches = (IMG_SIZE // 16) ** 2
    seq_len = num_patches + 1
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

    cls_tokens = CLSToken(embed_dim, name="cls_token")(patches)
    x = keras.layers.Concatenate(axis=1, name="cls_concat")([cls_tokens, patches])

    x = PosEmbed(seq_len, embed_dim, name="pos_embed")(x)

    for i in range(num_layers):
        x = transformer_block(x, num_heads=num_heads, embed_dim=embed_dim,
                               mlp_dim=mlp_dim, name=f"block_{i}")

    x = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
    cls_out = keras.layers.Lambda(lambda t: t[:, 0, :], name="cls_extract")(x)

    z = keras.layers.Dense(256, activation='relu', name="head_dense")(cls_out)
    z = keras.layers.Dropout(0.3, name="head_dropout")(z)
    outputs = keras.layers.Dense(num_classes, name="head_out")(z)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def _default(base, n):
    return base if n == 0 else f"{base}_{n}"

def _vit_layer_map(num_layers=12):
    m = {
        "patch_embed_proj": "conv2d",
        "cls_token": "cls_token",
        "pos_embed": "pos_embed",
        "ln_final": _default("layer_normalization", 2 * num_layers),
        "head_dense": _default("dense", 6 * num_layers),
        "head_out": _default("dense", 6 * num_layers + 1),
    }
    for i in range(num_layers):
        d = 6 * i
        for j, n in enumerate(["attn_q", "attn_k", "attn_v", "attn_proj", "mlp_fc1", "mlp_fc2"]):
            m[f"block_{i}_{n}"] = _default("dense", d + j)
        m[f"block_{i}_ln1"] = _default("layer_normalization", 2 * i)
        m[f"block_{i}_ln2"] = _default("layer_normalization", 2 * i + 1)
    return m

def load_vit_weights(model, path):
    mapping = _vit_layer_map()
    with h5py.File(path, "r") as f:
        for layer_name, file_name in mapping.items():
            layer = model.get_layer(layer_name)
            grp = f["layers"][file_name]["vars"]
            arrays = [grp[str(k)][()] for k in range(len(grp))]
            for var, arr in zip(layer.weights, arrays):
                if tuple(var.shape) != arr.shape:
                    raise ValueError(f"{layer_name} <- {file_name}: {var.shape} vs {arr.shape}")
            layer.set_weights(arrays)
    print(f"Loaded {len(mapping)} layers from {path}")