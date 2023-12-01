from ultralytics import YOLO
from utils import predict, config
import argparse, os

parser = argparse.ArgumentParser(description="Predict model parse arguments.")

parser.add_argument("--model", type=str, dest="model",
                    help="trained pt model file in models/ dir", default="yolov8s.pt")
parser.add_argument("--source", type=str, dest="source",
                    help="predict source path", default="../datasets/VisDrone/VisDrone2019-DET-test-dev/images/")

args = parser.parse_args()

model_name = args.model
name = model_name.removesuffix(".pt") if model_name.endswith(".pt") else model_name.removesuffix(".yaml")

predict(
    model=YOLO(f"./{name}/train/weights/best.pt"), source=args.source,
    project=name, name="predict")
    
