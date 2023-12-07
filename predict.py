from ultralytics import YOLO
from utils import predict
import argparse

parser = argparse.ArgumentParser(description="Predict model parse arguments.")

parser.add_argument("--model", type=str, dest="model",
                    help="trained pt model file in models/ dir", default="yolov8s.pt")
parser.add_argument("--source", type=str, dest="source",
                    help="predict source path", default="../datasets/VisDrone/VisDrone2019-DET-test-dev/images/")

args = parser.parse_args()

predict(
    model=YOLO(f"./runs/{args.model}/train/weights/best.pt"), 
    source=args.source,
    project=args.model, name="predict")
    
