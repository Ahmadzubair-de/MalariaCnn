import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from .model import MalariaModel

"""
🔧 Imports erklären:
- torch: PyTorch Framework fürs Deep Learning (GPU-Boosted, voll effizient)
- torch.nn as nn: Module für neuronale Netze (Layer, Loss-Funktionen)
- torchvision: Dataset & Bild-Transformationen (Mega praktisch für Bilder)
- transforms: Bilder anpassen (größer, drehen, spiegeln, normalisieren)
- DataLoader & random_split: Laden und Splitten von Daten in kleine Portionen
- matplotlib.pyplot & seaborn: Für coole Visualisierungen, z.B. Plots, Heatmaps
- numpy: High-Performance Array-Handling, wichtig fürs Bild-Math
- tqdm: Fortschrittsbalken für Trainingsloops (nice to have, easy zu checken)
- sklearn.metrics: Evaluation, um zu checken wie gut dein Modell wirklich ist
- .model import MalariaModel: Dein eigenes, custom neuronales Netz
"""

# 📦 Daten vorbereiten – Die Grundlage für jedes Modell
def prepare_data(batch_size=32):
    """
    Ziel: Daten laden und fit für's Training machen
    
    Warum so wichtig?
    - Deep Learning will große, saubere Daten in gleicher Form (224x224 px hier)
    - Data Augmentation sorgt dafür, dass dein Modell nicht nur 'ein Bild' lernt,
      sondern robust wird gegen kleine Änderungen (Drehen, Spiegeln, Helligkeit)
    - Normalisierung bringt Pixelwerte in standardisierten Bereich, Training
      läuft dadurch stabiler und schneller.
    
    Wie?
    - Compose: Kette von Transformationen, die nacheinander angewendet werden
    - ImageFolder: Automatisch Labeln basierend auf Ordnernamen, easy!
    - random_split: Teilt Datensatz in Training & Validierung (80/20)
    - DataLoader: Bricht Daten in Batches, mischt Training (Shuffle!), 
      damit das Modell nicht sequenziell lernt
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Alle Bilder auf gleiche Größe, wichtig für CNNs
        transforms.RandomHorizontalFlip(),   # Spiegeln horizontal - mehr Variation, bessere Generalisierung
        transforms.RandomRotation(10),       # Kleine Drehungen, damit Modell rotationsrobust wird
        transforms.RandomAffine(0, shear=10),# Verzerren, damit Modell nicht überoptimiert auf perfekte Bilder
        transforms.ColorJitter(0.2, 0.2),    # Helligkeit & Kontrast verändern, robust gegen Lichtverhältnisse
        transforms.ToTensor(),                # Konvertiert Bild in Tensor (numerische Matrix)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standardwerte aus ImageNet, normalisiert Farben
    ])

    dataset = torchvision.datasets.ImageFolder('./data/cell_images/cell_images', transform=transform)
    # Erwartet Ordnerstruktur mit Klassen als Ordnernamen (Parasitized, Uninfected)
    # Labels werden automatisch zugeordnet - super easy

    train_len = int(0.8 * len(dataset))  # 80% fürs Training, viel mehr Daten = besseres Modell
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    # Validierungsdaten wichtig, um Modell zwischendurch zu checken (nicht überfitten!)

    return (DataLoader(train_set, batch_size, shuffle=True),    # Shuffle für Zufälligkeit beim Training (keine Reihenfolge merken)
            DataLoader(val_set, batch_size, shuffle=False))     # Kein Shuffle bei Validierung, damit stabiler Vergleich

# 🔍 Bilder anzeigen - Check deine Daten bevor du startest
def visualize_samples(loader, path):
    """
    Zeigt ein paar Beispielbilder, um sicherzugehen, dass alles passt.
    
    Warum?
    - Visuelle Kontrolle hilft mega, Fehler in der Datenvorbereitung zu finden
    - Du kannst dir die Labels und Bildqualität anschauen, bevor du trainierst
    
    Wie?
    - next(iter(loader)): nimmt den ersten Batch aus dem DataLoader
    - permute(1,2,0): Tensor vom Format (C,H,W) in (H,W,C) umwandeln für Matplotlib
    - Normalisierung wird rückgängig gemacht, damit Farben realistisch aussehen
    - np.clip verhindert Farbüberläufe und unschöne Darstellungen
    - plt.subplot: Bilder ordentlich im Grid darstellen
    - plt.savefig: Speichert die Grafik, damit du sie später anschauen kannst
    """
    
    images, labels = next(iter(loader))
    classes = ['Parasitized', 'Uninfected']  # Labels, die zum Dataset passen
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    plt.figure(figsize=(12, 8))
    for i in range(8):
        img = images[i].permute(1, 2, 0).numpy() * std + mean  # Normalisierung rückgängig
        plt.subplot(2, 4, i+1)
        plt.imshow(np.clip(img, 0, 1))  # Werte in gültigen Bereich clippen
        plt.title(classes[labels[i]])    # Titel mit Label
        plt.axis('off')                 # Achsen aus, sieht cleaner aus
    plt.tight_layout()
    plt.savefig(path)  # Bild abspeichern, kannst du später checken

# 🚀 Das eigentliche Training starten
def train():
    """
    Startet den gesamten Trainingsprozess.
    
    Was passiert?
    - Daten vorbereiten
    - Modell initialisieren
    - Loss-Function & Optimierer festlegen
    - Trainings- und Validierungsloop über mehrere Epochen
    - Scheduler passt Lernrate an, falls Verlust stagniert
    - Ergebnisse visualisieren (Loss, Accuracy, Confusion Matrix, ROC-Kurve)
    - Modell speichern
    
    Warum so?
    - Strukturierter Ablauf sorgt für sauberen Code, den du easy anpassen kannst
    - Abwechselnd trainieren und validieren verhindert, dass Modell zu sehr auf Trainingsdaten angepasst wird (Overfitting)
    - Lernratenanpassung (Scheduler) hilft Training zu verbessern und schneller zu konvergieren
    """

    train_loader, val_loader = prepare_data()  # Daten ready machen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Nutzt GPU wenn verfügbar, sonst CPU (GPU ist mega schnell für ML)

    visualize_samples(train_loader, "./models/sample_images.png")  # Check deine Samples

    model = MalariaModel().to(device)  # Modell auf GPU/CPU laden
    criterion = nn.BCELoss()  # Binary Cross Entropy für 2-Klassen-Klassifikation (Sigmoid-Ausgabe)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam ist Standard, lernt Gewichte gut
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    # Scheduler: Wenn Validierungsverlust 3 Epochen nicht besser wird, Lernrate halbieren

    stats = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}  # Zum Tracken

    for epoch in range(4):  # 4 Epochen, meist guter Startpunkt (mehr bei Bedarf)
        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader

            # Unterschiedliche Modi fürs Modell:
            if phase == 'train':
                model.train()  # Aktiviert Dropout, BatchNorm (nötig fürs Training)
            else:
                model.eval()   # Deaktiviert Dropout etc. (für Validierung/Inferenz)

            running_loss, correct, total = 0, 0, 0  # Tracking für diese Epoche
            preds_list, labels_list, probs_list = [], [], []  # Für Auswertung in Val

            for images, labels in tqdm(loader, desc=f"{'🧠' if phase == 'train' else '🧪'} Epoch {epoch}/3"):
                images = images.to(device)
                labels = labels.float().view(-1, 1).to(device)  # Labels in FloatTensor & richtige Form (Batch,1)

                with torch.set_grad_enabled(phase == 'train'):  # Gradienten nur beim Training berechnen
                    outputs = model(images)                       # Vorhersagen, Werte zwischen 0 und 1 (Sigmoid)
                    loss = criterion(outputs, labels)             # Loss berechnen (Fehlermaß)

                    if phase == 'train':
                        optimizer.zero_grad()  # Alte Gradienten löschen (mega wichtig sonst accumuliert sich der Gradient)
                        loss.backward()        # Backpropagation (Fehler zurück ins Netz)
                        optimizer.step()       # Update der Gewichte, Schritt in Richtung besserer Ergebnisse

                running_loss += loss.item()  # Loss in float, für Durchschnittsberechnung
                preds = (outputs >= 0.5).float()  # 0.5 Threshold, ab da Klasse 1 (positiv)
                correct += (preds == labels).sum().item()  # Richtig klassifizierte Bilder zählen
                total += labels.size(0)  # Anzahl Bilder im Batch zählen

                if phase == 'val':  # Ergebnisse für Auswertung sammeln
                    preds_list.extend(preds.cpu().numpy())
                    labels_list.extend(labels.cpu().numpy())
                    probs_list.extend(outputs.cpu().numpy())

            avg_loss = running_loss / len(loader)  # Durchschnittlicher Loss
            acc = correct / total                   # Accuracy (Genauigkeit)
            stats[f'{phase}_loss'].append(avg_loss)
            stats[f'{phase}_acc'].append(acc)

            print(f"✅ {phase.capitalize()} | Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

            if phase == 'val':
                scheduler.step(avg_loss)  # Lernrate anpassen, falls Val Loss stagniert

    # 📈 Trainingsergebnisse visualisieren (Loss & Accuracy)
    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(['loss', 'acc']):
        plt.subplot(1, 2, i+1)
        plt.plot(stats[f'train_{metric}'], label='Train')
        plt.plot(stats[f'val_{metric}'], label='Val')
        plt.title(f'{metric.title()} über Epochen')
        plt.legend()
    plt.savefig('./models/training_history.png')

    # 📊 Confusion Matrix visualisieren (klare Übersicht über Fehlerarten)
    cm = confusion_matrix(labels_list, preds_list)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Parasitized', 'Uninfected'],
                yticklabels=['Parasitized', 'Uninfected'])
    plt.title('Konfusionsmatrix')
    plt.savefig('./models/confusion_matrix.png')

    # 📉 ROC-Kurve plotten (zeigt wie gut dein Modell zwischen Klassen unterscheidet)
    fpr, tpr, _ = roc_curve(labels_list, probs_list)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')  # AUC = Area Under Curve, max = 1.0
    plt.plot([0, 1], [0, 1], 'k--')  # Zufallslinie (Baseline)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Kurve')
    plt.legend()
    plt.savefig('./models/roc_curve.png')

    # 🏁 Modell abspeichern für späteren Gebrauch oder Deployment
    torch.save(model.state_dict(), './models/malaria_model_final.pth')
    print("🎉 Modell gespeichert!")

# ▶️ Entry Point des Skripts, wenn direkt ausgeführt
if __name__ == "__main__":
    train()  # Startet Training
