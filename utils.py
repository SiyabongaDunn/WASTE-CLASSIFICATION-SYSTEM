import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 224
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image):
    # 1. Convert to RGB numpy
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = np.array(image)
    else:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

    # 2. Ensure uint8
    if image.dtype!= np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # 3. Resize with cv2 (faster, no TF dependency in preprocess)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0) # (1,224,224,3)
    return image

def get_top_classes(predictions, top_k=3):
    if isinstance(predictions, dict):
        preds = predictions['predictions']
    else:
        preds = predictions[0] if len(predictions.shape) > 1 else predictions
    top_indices = np.argsort(preds)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'class_idx': int(idx),
            'confidence': float(preds[idx])
        })
    return results
