import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.model import build_malaria_cnn
from src.utils import prep_dataset


def train():  # Direkt "train" als Funktionsname
    # 📂 Pfade
    DATA_PATH = "./data/train"
    MODEL_PATH = "./models/model1/model.keras"
    PLOT_DIR = "./models/model1/plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 📸 Dataset laden
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        image_size=(128, 128),
        label_mode="binary",
        shuffle=True,
        batch_size=32
    )

    # Aufteilung in Training und Validation
    dataset_size = dataset.cardinality().numpy()
    val_size = int(0.2 * dataset_size)
    val_ds = dataset.take(val_size)
    train_ds = dataset.skip(val_size)

    # Vorverarbeitung
    train_ds = prep_dataset(train_ds, shuffle=True)
    val_ds = prep_dataset(val_ds, shuffle=False)

    # Modell erstellen und trainieren
    model = build_malaria_cnn()
    history = model.fit(train_ds, validation_data=val_ds, epochs=5)

    # Modell speichern
    model.save(MODEL_PATH)

    # Visualisierung
    create_plots(history, PLOT_DIR)


def create_plots(history, plot_dir):
    # Loss-Verlauf
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "loss.png"))
    plt.close()

    # Accuracy-Verlauf
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "accuracy.png"))
    plt.close()


# Direktausführung ermöglichen
if __name__ == "__main__":
    train()