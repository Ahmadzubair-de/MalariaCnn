import torch.nn as nn
import matplotlib.pyplot as plt 

# 🦠 Definiert das Modell für die Erkennung von Malariazellen
class MalariaModel(nn.Module):  # 👷 Erbt von nn.Module, weil alle PyTorch-Modelle davon ableiten müssen
    def __init__(self):  # 🔁 Konstruktor – wird beim Erstellen eines Objekts dieser Klasse aufgerufen
        super().__init__()  # 🧬 Ruft den Konstruktor der Elternklasse auf (nn.Module)

        # 🧠 Feature Extraction – Convolutional Neural Network (CNN)
        self.cnn = nn.Sequential(  # 📦 'Sequential' stapelt alle Layer in der Reihenfolge
            nn.Conv2d(3, 32, 3, padding=1),  # 🧩 3 RGB-Kanäle → 32 Filter mit 3x3 Größe (kein Größenverlust durch padding=1)
            nn.ReLU(),                       # ⚡ Aktivierungsfunktion für Nichtlinearität
            nn.BatchNorm2d(32),             # 🧼 Normalisiert die Outputs (schnelleres & stabileres Training)
            nn.MaxPool2d(2),                # 🔻 Halbiert die Breite & Höhe (z. B. von 224x224 → 112x112)

            nn.Conv2d(32, 64, 3, padding=1), # ⏭️ Weitere Conv-Schicht: jetzt mit 64 Filter
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                # 🔻 wieder halbieren → 56x56

            nn.Conv2d(64, 128, 3, padding=1), # ➕ Noch mehr Filter = tiefere Merkmale erkennen
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                # 🔻 wieder halbieren → 28x28

            nn.Conv2d(128, 256, 3, padding=1), # 🔬 Jetzt 256 Filter → sehr detaillierte Erkennung
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)                 # 🔻 Endgröße 14x14 bei Input-Größe 224x224
        )

        # 🧠 Fully Connected Classifier – entscheidet, ob Malaria oder nicht
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 📜 Wandelt 4D-Output der CNNs in 1D um → nötig für Linear Layer
            nn.Linear(256 * 14 * 14, 512),  # 🔢 FC-Layer: Verbindet jedes der 50.176 Features mit 512 Neuronen
            nn.ReLU(),                      # ⚡ Aktivierungsfunktion
            nn.Dropout(0.3),                # 💧 30% der Neuronen werden zufällig deaktiviert (Overfitting vermeiden)
            nn.Linear(512, 1),              # 🎯 Letzter Layer: Gibt 1 Wert aus (→ Wahrscheinlichkeit für Malaria)
            nn.Sigmoid()                    # 🧪 Skaliert Wert auf [0, 1] → perfekt für binäre Klassifikation
        )

    def forward(self, x):  # 🔄 Definiert, wie Daten durch das Modell fließen
        x = self.cnn(x)     # 📷 Erst durch die CNNs
        return self.classifier(x)  # 🎯 Dann durch den Klassifikator → Output = Wahrscheinlichkeit
