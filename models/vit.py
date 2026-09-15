import keras
import numpy as np
import tensorflow as tf

from .vit_layers import transformer_block, CLSToken, PosEmbed
from .pretrained_weights import get_weight

def build_vit_model(
    num_classes=6,
    image_size=224,
    patch_size=16,
    embed_dim=768,
    num_heads=12,
    mlp_dim=3072,
    num_layers=12,
    learning_rate=2e-5,
    weight_decay=0.01
):
    
    num_patches = (image_size // patch_size) ** 2
    seq_len = num_patches + 1
    
    inputs = keras.Input(shape=(image_size, image_size, 3), name="pixel_values")
    
    patches = keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        use_bias=True,
        name="patch_embed_proj"
    )(inputs)
    patches = keras.layers.Reshape((num_patches, embed_dim), name="patch_embed_reshape")(patches)
    
    cls_tokens = CLSToken(embed_dim, name="cls_token")(patches)
    x = keras.layers.Concatenate(axis=1, name="cls_concat")([cls_tokens, patches])
    x = PosEmbed(seq_len, embed_dim, name="pos_embed")(x)
    
    for i in range(num_layers):
        x = transformer_block(
            x,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            name=f"block_{i}"
        )
    
    # Layer norm and CLS extraction
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
    cls_out = keras.layers.Lambda(lambda t: t[:, 0, :], name="cls_extract")(x)
    
    # Classification head
    z = keras.layers.Dense(256, activation='relu', name="head_dense")(cls_out)
    z = keras.layers.Dropout(0.3, name="head_dropout")(z)
    outputs = keras.layers.Dense(num_classes, name="head_out")(z)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Transfer pretrained weights
    _transfer_weights(model)
    
    # Freeze backbone
    for layer in model.layers:
        if not layer.name.startswith("head_"):
            layer.trainable = False
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Print parameter counts
    trainable = sum([tf.size(v).numpy() for v in model.trainable_variables])
    non_trainable = sum([tf.size(v).numpy() for v in model.non_trainable_variables])
    print(f"Trainable:     {trainable:,}")
    print(f"Non-trainable: {non_trainable:,}")
    print(f"Total:         {trainable + non_trainable:,}")
    
    return model


def _transfer_weights(model):
    print("Transferring pretrained weights...")
    
    def get_pt(key):
        return get_weight(key)
    
    # Patch embedding
    model.get_layer("patch_embed_proj").set_weights([
        np.transpose(get_pt("embeddings.patch_embeddings.projection.weight"), (2, 3, 1, 0)),
        get_pt("embeddings.patch_embeddings.projection.bias")
    ])
    
    # CLS token
    model.get_layer("cls_token").set_weights([
        get_pt("embeddings.cls_token").reshape(1, 1, 768)
    ])
    
    # Positional embedding
    model.get_layer("pos_embed").set_weights([
        get_pt("embeddings.position_embeddings").reshape(1, 197, 768)
    ])
    
    # Transformer blocks
    for i in range(12):
        p = f"layers.{i}"
        
        # Layer norm 1
        model.get_layer(f"block_{i}_ln1").set_weights([
            get_pt(f"{p}.layernorm_before.weight"),
            get_pt(f"{p}.layernorm_before.bias")
        ])
        
        # Q, K, V projections
        model.get_layer(f"block_{i}_attn_q").set_weights([
            get_pt(f"{p}.attention.q_proj.weight").T,
            get_pt(f"{p}.attention.q_proj.bias")
        ])
        model.get_layer(f"block_{i}_attn_k").set_weights([
            get_pt(f"{p}.attention.k_proj.weight").T,
            get_pt(f"{p}.attention.k_proj.bias")
        ])
        model.get_layer(f"block_{i}_attn_v").set_weights([
            get_pt(f"{p}.attention.v_proj.weight").T,
            get_pt(f"{p}.attention.v_proj.bias")
        ])
        model.get_layer(f"block_{i}_attn_proj").set_weights([
            get_pt(f"{p}.attention.o_proj.weight").T,
            get_pt(f"{p}.attention.o_proj.bias")
        ])
        
        # Layer norm 2
        model.get_layer(f"block_{i}_ln2").set_weights([
            get_pt(f"{p}.layernorm_after.weight"),
            get_pt(f"{p}.layernorm_after.bias")
        ])
        
        # MLP
        model.get_layer(f"block_{i}_mlp_fc1").set_weights([
            get_pt(f"{p}.mlp.fc1.weight").T,
            get_pt(f"{p}.mlp.fc1.bias")
        ])
        model.get_layer(f"block_{i}_mlp_fc2").set_weights([
            get_pt(f"{p}.mlp.fc2.weight").T,
            get_pt(f"{p}.mlp.fc2.bias")
        ])
    
    # Final layer norm
    model.get_layer("ln_final").set_weights([
        get_pt("layernorm.weight"),
        get_pt("layernorm.bias")
    ])
    
    print(" Pretrained weights transferred")