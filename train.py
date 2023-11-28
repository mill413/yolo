from ultralytics import YOLO
from utils import load,train,config,value

for model_name in config["models"]:
    name = model_name.removesuffix(".pt") if model_name.endswith(".pt") else model_name.removesuffix(".yaml")

    model = train(model=load(model_name), dataset=config["dataset"], 
          project=config["project"], name = name+"-train",
          epochs=100, exist_ok=True, batch=16, workers=2)
    value(
        model=model,dataset=config["dataset"],
        project=config["project"], name=name+"-val")
