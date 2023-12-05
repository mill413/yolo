from pathlib import Path
import time
from ultralytics import YOLO

split_line = "-------------------------------------------"

config = {
    "models": ["yolov8s.pt", "yolov8-sod.yaml"],
    "dataset": "VisDrone.yaml",
    "project": "test"
}

def log(msg:str):
    date = time.strftime("%Y-%m-%d", time.localtime())
    log_file = Path(f"./logs/{date}.log")
    log_file.touch()

    with open(log_file, mode = "a+") as log:
        print(msg)
        log.write(msg+"\n" if not msg.endswith("\n") else msg)

def load(model_name:str):
    model = YOLO(f"./models/{model_name}.yaml")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    log(f"{split_line} Load model {model_name} at {current_time} {split_line}")
    return model
    
def train(
        model:YOLO,dataset:str,
        project:str,name:str="train",
        epochs=100,exist_ok=False,
        batch=16,device=0,workers=2,
):
    log(f"Start train model on {dataset}.\nEpochs:{epochs} Batch:{batch} Workers:{workers} Exist_ok:{exist_ok} Device:{device}")
    model.train(data=f"./datasets/{dataset}", epochs=epochs, project=f"runs/{project}", name=name, exist_ok=exist_ok, batch=batch, device=device,workers=workers)
    log(f"End train.")

    return model

def value(
        model:YOLO,dataset:str,
        project:str,name:str="val",
        batch=16,device=0,workers=2, exist_ok=False
):
    log(f"Start value model on {dataset}.")
    metrics = model.val(device=device, batch=batch, workers=workers, project=f"runs/{project}", name=name, exist_ok=exist_ok)
    log(f"map50-95:{metrics.box.map}")
    log(f"map50:{metrics.box.map50}")
    log(f"End value.")

def predict(model:YOLO,source:str,project:str,name:str="predict",
         save=True, show_conf=False, show=False):
    log(f"Start predict on {source} via {project}.")
    results = model(source, save=save, show_conf=show_conf, show=show, project=f"runs/{project}", name=name, exist_ok=True, line_width=1)
    log(f"End predict.")
