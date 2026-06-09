# This file is responsible for loading the dataset and returning the training
# and validation datasets that can be used by train.py and evaluate.py

import tensorflow as tf

DATASET_PATH = "Dataset/Garbage classification/Garbage classification"
IMG_SIZE     = (224, 224) # All images will be resized to this size for MobileNetV2 input
BATCH_SIZE   = 32 # Number of samples per batch during training/validation
SEED         = 42 # For reproducibility — ensures the same train/val split and shuffling order across runs

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Data augmentation pipeline — applied to training set only
# Addresses the small sample size of specific classes (e.g. 'trash' with 127 images)
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),          # ±15% rotation
    tf.keras.layers.RandomZoom(0.10),              # ±10% zoom
    tf.keras.layers.RandomContrast(0.20),          # colour jitter — contrast
    tf.keras.layers.RandomBrightness(0.20),        # colour jitter — brightness
    tf.keras.layers.RandomTranslation(0.10, 0.10), # slight shift/scaling
], name="augmentation")


# define this function so that train.py and evaluate.py can import it
# and get the datasets that it returns
def get_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # Normalise pixels to [0, 1] for both splits
    normalise = tf.keras.layers.Rescaling(1./255)
    train_ds  = train_ds.map(lambda x, y: (normalise(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds    = val_ds.map(lambda x, y: (normalise(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation to training set ONLY
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    # Optimise pipeline performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# Define this function to return the class names for use in train.py and evaluate.py
def get_class_names():
    return CLASS_NAMES