from ultralytics import YOLO
from utils import predict, config

for model_name in config["models"]:
    name = model_name.removesuffix(".pt") if model_name.endswith(".pt") else model_name.removesuffix(".yaml")
    predict(
        model = YOLO(f"./{config['project']}/{name}-train/weights/best.pt"),
        source = "../datasets/VisDrone/VisDrone2019-DET-test-dev/images/",
        project = config["project"], name = name + "-predict")
