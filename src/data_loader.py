import tensorflow as tf

DATASET_PATH = "Dataset/Garbage classification/Garbage classification"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

#define this function so that train.py and evaluate.py can import it and get the datasets that it returns
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
    normalise = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalise(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalise(x), y))
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds