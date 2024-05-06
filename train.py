import argparse

from utils import train

parser = argparse.ArgumentParser(description="Train model parse arguments.")

parser.add_argument("--model", type=str, dest="model",
                    help="yaml model file", default="yolov8s")
parser.add_argument("--dataset", type=str, dest="dataset",
                    help="dataset config file path", default="VisDrone")
parser.add_argument("--epochs", type=int, default=100,
                    help="train epochs", dest="epochs")
parser.add_argument("--workers", type=int, default=4,
                    help="train workers", dest="workers")
parser.add_argument("--batch", type=int, default=8,
                    help="train batch", dest="batch")
parser.add_argument("--project", type=str, default="",
                    help="results save dir", dest="project")

args = parser.parse_args()

model_name = args.model
project = args.project if not args.project == "" else model_name

model = train(model_name=model_name, dataset=args.dataset,
              result_dir=f"{project}/{args.dataset}", mode="train",
              epochs=args.epochs, exist_ok=True, 
              batch=args.batch, workers=args.workers,
              patience=50)
