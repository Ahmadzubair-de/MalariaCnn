import argparse
from src import train, predict

def main():
    parser = argparse.ArgumentParser(description="Malaria CNN Tool")
    parser.add_argument('--train', action='store_true', help="Starte Training")
    parser.add_argument('--predict', type=str, help="Pfad zum Testbild")
    parser.add_argument('--model', type=str, default="./models/model1/model.keras", help="Pfad zum Modell (optional)")

    args = parser.parse_args()

    if args.train:
        print("Starte Training...")
        train.train()  # Einfacher Aufruf der train-Funktion
    elif args.predict:
        print(f"Starte Prediction für Bild: {args.predict}")
        predict.predict(args.predict, model_path=args.model)
    else:
        print("Kein gültiger Befehl! Benutze --train oder --predict <Pfad zum Bild>")

if __name__ == "__main__":
    main()