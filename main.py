import argparse
from src import train, predict

parser = argparse.ArgumentParser(description="Malaria Detection")
parser.add_argument("--train", action="store_true", help="Modell trainieren")
parser.add_argument("--predict", type=str, help="Pfad zum Testbild")

args = parser.parse_args()

if args.train:
    train.train()
elif args.predict:
    predict.predict(args.predict)
else:
    print("Bitte Option angeben: --train oder --predict <bildpfad>")