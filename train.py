from utils import load,train,config,value
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Train model parse arguments.")

parser.add_argument("--model", type=str, dest="model",
                    help="yaml model file", default="yolov8s")
parser.add_argument("--dataset", type=str, dest="dataset",
                    help="dataset config file path", default="VisDrone.yaml")
parser.add_argument("--epochs", type=int, default=100,
                    help="train epochs", dest="epochs")
parser.add_argument("--project", type=str, default="",
                    help="results save dir", dest="project")

args = parser.parse_args()

model_name = args.model
project = args.project if not args.project == "" else model_name

model = train(model=load(model_name), dataset=args.dataset, 
        project=project, name = "train",
        epochs=args.epochs, exist_ok=True, batch=16, workers=2)

value(
    model=YOLO(f"./{project}/train/weights/best.pt"), dataset=args.dataset,
    project=project, name="val")
    