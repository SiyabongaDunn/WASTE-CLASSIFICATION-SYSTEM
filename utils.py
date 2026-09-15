import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def preprocess_image(image):
    """Mirrors the training-time pipeline exactly:
    decode -> tf.image.resize (bilinear) -> cast float32 -> /255.0.
    Previously this used cv2.resize, whose bilinear kernel doesn't produce
    byte-identical output to tf.image.resize -- using tf.image here removes
    that mismatch. Also guards against RGBA/grayscale uploads, which would
    otherwise break the model's fixed (224, 224, 3) input shape.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    else:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def get_top_classes(predictions, top_k=3):
    preds = predictions[0]
    top_indices = np.argsort(preds)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'class_idx': int(idx),
            'confidence': float(preds[idx])
        })
    return results