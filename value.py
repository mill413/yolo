from utils import load,train,config,value
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Value model parse arguments.")

parser.add_argument("--model", type=str, dest="model",
                    help="model name", default="yolov8s")
parser.add_argument("--dataset", type=str, dest="dataset",
                    help="dataset config file path", default="VisDrone.yaml")
parser.add_argument("--project", type=str, default="",
                    help="results save dir", dest="project")

args = parser.parse_args()

model_name = args.model
project = args.project if not args.project == "" else model_name

value(
    model=YOLO(f"./runs/{project}/train/weights/best.pt"), dataset=args.dataset,
    project=project, name="val", exist_ok=True)
    
