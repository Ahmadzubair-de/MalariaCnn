import torch
import torchversion as tv
"""
Importiert PyTorch, die Framework-Bibliothek für Deep Learning. 
Wichtig, weil wir das Modell laden, Eingabedaten verarbeiten und Berechnungen durchführen.
Ohne torch läuft hier nix mit KI.
"""

import torchvision.transforms as transforms
"""
Importiert die 'transforms' aus torchvision, damit wir Bilder vorverarbeiten können.
Bilder müssen z.B. skaliert, normalisiert und in Tensoren verwandelt werden, bevor sie ins Modell gehen.
Ohne saubere Vorverarbeitung hat das Modell keinen Plan, was es sieht.
"""

from PIL import Image
"""
PIL (Python Imaging Library) ist der Klassiker für Bild-Handling.
Hier laden wir Bilder, konvertieren sie in RGB usw.
Wichtig, weil wir das Bild zum Modell schicken müssen.
"""

import matplotlib.pyplot as plt
import seaborn as sb
"""
Matplotlib ist das Standard-Tool für Visualisierungen in Python.
Wir zeigen später Bilder, Ergebnisse und Vorhersagen damit an.
Sehr praktisch, um die Resultate verständlich zu machen.
"""

import numpy as np
"""
Numpy ist die Basis für alle numerischen Operationen in Python.
Hier wandeln wir Bilder in Arrays um, um sie mit OpenCV weiterzuverarbeiten.
"""

import cv2
"""
OpenCV ist eine Bibliothek für Bildverarbeitung.
Hier erkennen wir die Konturen (Zellen) im Bild, markieren sie und machen das Ergebnis sichtbar.
Unverzichtbar für Computer Vision Stuff neben Deep Learning.
"""

from .model import MalariaModel
"""
Importiert dein selbstgebautes Malaria-Modell.
Das ist das Herzstück, mit dem wir die Vorhersagen machen.
Das Modell wurde vorher trainiert, jetzt wird es geladen und benutzt.
"""

def predict(image_path):
    """
    Funktion, die ein Bild als Pfad nimmt und dann den Malaria-Status vorhersagt und visualisiert.
    Hier passiert der ganze Ablauf von Laden bis Ausgabe.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    Checkt, ob du eine GPU hast (cuda), sonst nutzt es die CPU.
    GPU macht Berechnungen viel schneller, besonders bei Deep Learning.
    Falls keine GPU da ist, geht es trotzdem, aber langsamer.
    """

    model = MalariaModel().to(device)
    """
    Erstellt eine Instanz von deinem Modell und schickt sie auf das passende Gerät (GPU/CPU).
    Wichtig, sonst läuft das Modell nicht oder es gibt Fehler bei der Berechnung.
    """

    model.load_state_dict(torch.load('models/malaria_model_best.pth', map_location=device))
    """
    Lädt die trainierten Gewichte (gelerntes Wissen) in das Modell.
    'map_location=device' stellt sicher, dass die Gewichte auf dem richtigen Gerät geladen werden.
    Ohne das ist dein Modell leer und kann nichts vorhersagen.
    """

    model.eval()
    """
    Setzt das Modell in den Evaluierungsmodus.
    Wichtig, damit z.B. BatchNorm und Dropout richtig funktionieren und keine Updates mehr passieren.
    Modell verhält sich dann so, als würde es "nur" vorhersagen, nicht trainieren.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    """
    Definition der Bildvorverarbeitung:
    - Resize: Bild auf 224x224 Pixel skalieren, weil das Modell diese Größe erwartet.
    - ToTensor: Bild in Tensor (Zahlenarray) umwandeln, da PyTorch damit arbeitet.
    - Normalize: Pixelwerte standardisieren (mit Mittelwert und Standardabweichung),
      damit das Modell stabile und vergleichbare Eingaben bekommt.
    Beispiel: Ohne Normalisierung könnten Bilder zu hell oder dunkel sein, Modell macht schlechte Vorhersagen.
    """

    img = Image.open(image_path).convert('RGB')
    """
    Lädt das Bild vom Pfad und stellt sicher, dass es RGB ist (3 Kanäle).
    Manche Bilder haben z.B. Alpha-Kanal oder sind Graustufen, das kann Probleme machen.
    """

    input_tensor = transform(img).unsqueeze(0).to(device)
    """
    Wendet die Vorverarbeitung auf das Bild an.
    .unsqueeze(0) fügt eine Batch-Dimension hinzu, weil Modelle mehrere Bilder gleichzeitig erwarten.
    Danach schickt es den Tensor auf das richtige Gerät (GPU/CPU).
    Beispiel: Ein Modell ohne Batch-Dimension würde Fehler werfen.
    """

    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
    """
    Mit 'torch.no_grad()' werden keine Gradienten berechnet, was Speicher spart und schneller macht.
    Modell macht eine Vorhersage auf das Bild.
    output ist ein Tensor mit Wahrscheinlichkeit, dass die Zelle parasitiert ist.
    .item() wandelt den Tensor in eine normale Zahl um.
    """

    img_np = np.array(img)
    """
    Konvertiert das PIL-Bild in ein NumPy Array, damit OpenCV damit arbeiten kann.
    OpenCV arbeitet nicht direkt mit PIL, sondern braucht numpy-Arrays.
    """

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    """
    Wandelt das Farbbild in ein Graustufenbild um.
    Für Konturenerkennung braucht man oft nur Helligkeit, keine Farbe.
    Das vereinfacht und beschleunigt die Bildverarbeitung.
    """

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    """
    Wendet einen Gaußschen Weichzeichner an, um Bildrauschen zu reduzieren.
    Das hilft, falsche Konturen zu vermeiden.
    Beispiel: Ohne Blur könnten viele kleine Punkte als Konturen erkannt werden.
    """

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    """
    Schwellenwertverfahren: Trennt Bild in Vordergrund und Hintergrund (schwarz/weiß).
    THRESH_BINARY_INV kehrt Farben um (weiß wird schwarz und andersrum).
    OTSU sucht automatisch den besten Schwellenwert.
    Wichtig für saubere Konturenerkennung.
    """

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """
    Findet alle äußeren Konturen (Umrisse) im binarisierten Bild.
    Konturen sind einfach Linien um Formen, z.B. Zellen.
    RETR_EXTERNAL holt nur äußere Konturen, CHAIN_APPROX_SIMPLE spart Speicher.
    """

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """
    Wenn Konturen gefunden wurden, wird die größte (vermutlich die wichtigste Zelle) ausgewählt.
    Dann wird ein grünes Rechteck darum gezeichnet.
    So sieht man visuell, wo die Zelle ist.
    """

    plt.figure(figsize=(12, 6))
    """
    Öffnet ein neues Plot-Fenster mit Größe 12x6 Zoll für die Visualisierung.
    """

    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Detected Cell" if contours else "No Cell Detected", fontsize=12, color='green' if contours else 'red')
    plt.axis('off')
    """
    Zeigt das Bild mit der markierten Zelle (falls gefunden).
    Titel ändert sich je nachdem, ob Zellen erkannt wurden.
    Achsen werden ausgeblendet für bessere Optik.
    """

    plt.subplot(1, 2, 2)
    status = "Parasitized" if probability >= 0.5 else "Uninfected"
    color = "red" if status == "Parasitized" else "green"
    conf = probability*100 if status == "Parasitized" else (1 - probability)*100

    plt.text(0.5, 0.6, status, fontsize=20, ha='center', color=color, weight='bold')
    plt.text(0.5, 0.4, f"Confidence: {conf:.1f}%", fontsize=16, ha='center')
    plt.text(0.5, 0.2, f"Probability: {probability:.4f}", fontsize=14, ha='center')
    plt.axis('off')
    """
    Zeigt den Vorhersage-Status, die Sicherheit (Confidence) und die genaue Wahrscheinlichkeit.
    Farbe signalisiert Gefahr (rot für infiziert, grün für gesund).
    Text wird zentriert dargestellt, Achsen ausgeblendet.
    """

    plt.tight_layout()
    plt.savefig('models/prediction_result.png', bbox_inches='tight')
    """
    Optimiert die Abstände im Plot für bessere Optik.
    Speichert die Visualisierung als PNG.
    Praktisch, um Ergebnisse später nochmal anzuschauen oder zu teilen.
    """

    print(f"\nMalaria Detection für {image_path}:")
    print(f"- Status: {status}")
    print(f"- Confidence: {conf:.1f}%")
    print(f"- Probability: {probability:.4f}")
    print("Visualisierung unter 'models/prediction_result.png' gespeichert")
    """
    Gibt die wichtigsten Infos auch in der Konsole aus.
    Nützlich für schnelle Checks, wenn du keine Bilder anschauen willst.
    """

if __name__ == "__main__":
    import sys
    predict(sys.argv[1])
    """
    Wenn das Skript direkt ausgeführt wird (z.B. python predict.py bild.jpg),
    wird die predict-Funktion mit dem ersten Kommandozeilenargument (Bildpfad) aufgerufen.
    So kannst du das Script einfach in der Shell benutzen.
    """
