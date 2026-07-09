import tensorflow as tf
import keras


def mlp_block(x, hidden_dim, out_dim, name):
    """MLP block for ViT."""
    x = keras.layers.Dense(hidden_dim, activation='gelu', name=f"{name}_fc1")(x)
    x = keras.layers.Dense(out_dim, name=f"{name}_fc2")(x)
    return x


def attention_block(x, num_heads, head_dim, embed_dim, name):
    # Query, Key, Value projections
    q = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_q")(x)
    k = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_k")(x)
    v = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_v")(x)
    
    # Reshape for multi-head attention
    q = keras.layers.Reshape((-1, num_heads, head_dim), name=f"{name}_q_reshape")(q)
    k = keras.layers.Reshape((-1, num_heads, head_dim), name=f"{name}_k_reshape")(k)
    v = keras.layers.Reshape((-1, num_heads, head_dim), name=f"{name}_v_reshape")(v)
    
    # Permute to (batch, heads, seq_len, head_dim)
    q = keras.layers.Permute((2, 1, 3), name=f"{name}_q_perm")(q)
    k = keras.layers.Permute((2, 1, 3), name=f"{name}_k_perm")(k)
    v = keras.layers.Permute((2, 1, 3), name=f"{name}_v_perm")(v)
    
    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn = keras.layers.Lambda(
        lambda qk: tf.nn.softmax(
            tf.matmul(qk[0], tf.transpose(qk[1], (0, 1, 3, 2))) * scale,
            axis=-1
        ),
        name=f"{name}_attn_scores"
    )([q, k])
    
    # Apply attention to values
    out = keras.layers.Lambda(
        lambda av: tf.matmul(av[0], av[1]),
        name=f"{name}_attn_out"
    )([attn, v])
    
    # Restore original shape
    out = keras.layers.Permute((2, 1, 3), name=f"{name}_out_perm")(out)
    out = keras.layers.Reshape((-1, embed_dim), name=f"{name}_out_reshape")(out)
    out = keras.layers.Dense(embed_dim, use_bias=True, name=f"{name}_proj")(out)
    
    return out


def transformer_block(x, num_heads=12, embed_dim=768, mlp_dim=3072, name="block"):
    # Self-attention with residual
    y = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    y = attention_block(
        y,
        num_heads=num_heads,
        head_dim=embed_dim // num_heads,
        embed_dim=embed_dim,
        name=f"{name}_attn"
    )
    x = keras.layers.Add()([x, y])
    
    # MLP with residual
    y = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    y = mlp_block(y, hidden_dim=mlp_dim, out_dim=embed_dim, name=f"{name}_mlp")
    x = keras.layers.Add()([x, y])
    
    return x


class CLSToken(keras.layers.Layer):
    """Learnable CLS token layer."""
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
        # FIXED: Use tf.shape instead of keras.ops
        batch_size = tf.shape(x)[0]
        return tf.broadcast_to(self.cls, (batch_size, 1, self.embed_dim))


class PosEmbed(keras.layers.Layer):
    """Learnable positional embedding layer."""
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