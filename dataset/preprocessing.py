import tensorflow as tf

def preprocess_image(image_path, label, data_dir, image_size=224):
    full_path = tf.strings.join([data_dir, "/", image_path])

    image = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label