import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image  # ← fehlt in deinem Code

# Für Dataset-Batches (Training)
def preproc_img(image_tensor, label, target_size=(128, 128)):
    image_tensor = tf.image.resize(image_tensor, target_size)
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
    return image_tensor, label

def prep_dataset(dataset, target_size=(128, 128), shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(lambda x, y: preproc_img(x, y, target_size))
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# Für Einzelbild (Prediction)
def load_and_preproc_image(img_path, target_size=(128, 128)):
    """
    Lädt ein einzelnes Bild von Pfad, resized es und normalisiert es.
    Gibt ein numpy-Array mit Batch-Dimension zurück.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
