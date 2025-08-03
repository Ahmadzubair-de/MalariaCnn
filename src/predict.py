import tensorflow as tf
import os
from src.utils import load_and_preproc_image


def predict(img_path, model_path="../models/model1/model.keras"):
    # ⛔ Error, wenn das Modell nicht existiert
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modell nicht gefunden unter: {model_path}")

    # 📥 Modell laden
    model = tf.keras.models.load_model(model_path)

    # 🖼️ Bild vorbereiten
    img = load_and_preproc_image(img_path)

    # 🤖 Prediction
    pred = model.predict(img)[0][0]
    label = "Uninfected 💉" if pred > 0.5 else "Parasitized 🦠"
    confidence = pred if pred > 0.5 else 1 - pred

    print(f"\n🔍 Prediction: {label}")
    print(f"📊 Confidence: {confidence * 100:.2f}%\n")

    return label, confidence
