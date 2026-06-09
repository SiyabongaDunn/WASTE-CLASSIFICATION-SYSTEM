# MobileNetV2 model for waste classification
# Uses transfer learning from ImageNet weights

import tensorflow as tf
from data_loader import IMG_SIZE, CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)


def build_mobilenet(trainable_base: bool = False) -> tf.keras.Model:
    """
    Build a MobileNetV2 model with a custom classification head.

    Args:
        trainable_base: If True, all base layers are unfrozen for fine-tuning.
                        If False, only the classification head is trained.

    Returns:
        Compiled Keras model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='MobileNetV2_Waste')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_mobilenet()
    model.summary()