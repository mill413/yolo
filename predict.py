from ultralytics import YOLO
from utils import predict
import argparse

parser = argparse.ArgumentParser(description="Predict model parse arguments.")

parser.add_argument("--model", type=str, dest="model",
                    help="trained pt model file in models/ dir", default="yolov8s.pt")
parser.add_argument("--source", type=str, dest="source",
                    help="predict source path", default="../datasets/VisDrone/VisDrone2019-DET-test-dev/images/")
parser.add_argument("--dataset", type=str, dest="dataset",
                    help="predict dataset", default="visdrone")
args = parser.parse_args()

predict(
    model_name=args.model, 
    source=args.source,
    dataset=args.dataset,
    result_dir=f"{args.model}/{args.dataset}", mode="predict")
    
