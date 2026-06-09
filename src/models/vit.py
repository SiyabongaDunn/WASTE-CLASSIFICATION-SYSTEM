# Vision Transformer (ViT-base-patch16-224) for waste classification
# Uses HuggingFace transformers with TensorFlow backend

import numpy as np
import tensorflow as tf
from transformers import TFViTModel, ViTFeatureExtractor
from data_loader import IMG_SIZE, CLASS_NAMES

NUM_CLASSES    = len(CLASS_NAMES)
VIT_CHECKPOINT = "google/vit-base-patch16-224"


def build_vit(trainable: bool = True) -> tf.keras.Model:
    """
    Build a ViT-base-patch16-224 model with a custom classification head.

    The [CLS] token output from the final transformer encoder block is
    passed to a Dense(6) softmax head for 6-class waste classification.

    Args:
        trainable: Whether to fine-tune the ViT backbone weights.

    Returns:
        Compiled Keras model.
    """
    # Load pretrained ViT backbone
    vit_backbone = TFViTModel.from_pretrained(VIT_CHECKPOINT)
    vit_backbone.trainable = trainable

    # Build functional model
    # Input: normalised image tensor [batch, 224, 224, 3]
    pixel_values = tf.keras.Input(
        shape=(*IMG_SIZE, 3),
        name='pixel_values'
    )

    # HuggingFace ViT expects channels-first: [batch, 3, 224, 224]
    # Transpose from channels-last (TF default)
    x = tf.transpose(pixel_values, perm=[0, 3, 1, 2])

    # Extract [CLS] token from ViT output (index 0 of sequence)
    vit_output = vit_backbone(x, training=trainable)
    cls_token  = vit_output.last_hidden_state[:, 0, :]   # shape [batch, 768]

    # Classification head
    x       = tf.keras.layers.Dropout(0.1)(cls_token)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs=pixel_values, outputs=outputs, name='ViT_Waste')

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=2e-5,
            weight_decay=0.01
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_feature_extractor():
    """Return the ViT feature extractor for preprocessing."""
    return ViTFeatureExtractor.from_pretrained(VIT_CHECKPOINT)


if __name__ == "__main__":
    model = build_vit()
    model.summary()