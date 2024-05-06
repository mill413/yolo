import time
import traceback
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

split_line = "------------------------------------------"


class YoloLogger:
    def __init__(self, model_name: str) -> None:
        self._model = model_name
        self._logs_dir_path = Path(f"./logs/{self._model}")
        self._logs_dir_path.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        date = time.strftime("%Y-%m-%d", time.localtime())
        log_file = self._logs_dir_path / f"{date}.log"
        log_file.touch()

        with open(log_file, mode="a+") as log:
            print(msg)
            log.write(msg+"\n" if not msg.endswith("\n") else msg)


def load(model_name: str, dataset: str = "", mode: str = "train"):
    logger = YoloLogger(model_name=f"{model_name}")
    log = logger.log

    match mode:
        case "train":
            model_path = f"./models/{model_name}.yaml"
        case "val" | "test":
            model_path = f"./runs/{model_name}/{dataset}/train/weights/best.pt"
        case _:
            log(f"ERROR occur when loading model with wrong mode {mode}")
            raise Exception(f"Wrong mode {mode}")

    current_time = time.strftime("%H:%M:%S", time.localtime())
    log(f"{split_line} Load model {model_name} at {current_time} {split_line}")

    if not Path(model_path).exists:
        log(f"ERROR occur when loading NOT FOUND MODEL {model_path}")
        raise FileNotFoundError(f"Model {model_path} Not Found!")
    
    model = YOLO(model_path)
    return model


def train(
        model_name: str, dataset: str,
        result_dir: str, mode: str = "train",
        epochs=100, exist_ok=False,
        batch=16, device=0, workers=2,
        patience=50
):
    model = load(model_name=model_name)

    logger = YoloLogger(model_name=f"{model_name}")
    log = logger.log
    log(f"Start train model on {dataset}.\n" +
        f"Epochs: {epochs}, Batch: {batch}, Workers: {workers}, Exist_ok: {exist_ok}, Device: {device}, " +
        f"Paras: {get_num_params(model)/1000000:.3f}M\n" +
        f"Save to runs/{result_dir}/{mode}")

    try:
        model.train(
            data=f"./datasets/{dataset}.yaml",
            epochs=epochs,
            project=f"runs/{result_dir}", name=mode,
            exist_ok=exist_ok,
            batch=batch,
            device=device,
            workers=workers,
            patience=patience)
    except Exception as e:
        log("=======")
        log(f"{traceback.format_exc()}")
    finally:
        log(f"End train.")

    return model


def value(
    model_name: str, dataset: str,
    result_dir: str, mode: str = "val",
    batch=16, device=0, workers=2, exist_ok=False
):
    model = load(model_name=model_name, dataset=dataset, mode="val")
    logger = YoloLogger(model_name=f"{model_name}")
    log = logger.log
    log(f"Start value model on {dataset}.")

    metrics = model.val(
        data=f"./datasets/{dataset}.yaml",
        device=device,
        batch=batch,
        workers=workers,
        project=f"runs/{result_dir}", name=mode,
        exist_ok=exist_ok)

    # print value results
    class_names = model.names
    log(f"|{'Class|':>22s}\t{'Precision|':>11s}\t{'Recall|':>11s}\t{'mAP50|':>11s}\t{'mAP50-95|':>11s}")
    log(f"|{'---:|':>22s}\t{'---:|':>11s}\t{'---:|':>11s}\t{'---:|':>11s}\t{'---:|':>11s}")
    mean_result = metrics.mean_results()
    log(f"|{'all|':>22s}\t{mean_result[0]:10.3g}|\t{mean_result[1]:10.3g}|\t{mean_result[2]:10.3g}|\t{mean_result[3]:10.3g}|")
    for i, c in enumerate(metrics.ap_class_index):
        mode = class_names[c]
        result = metrics.class_result(i)
        precision = result[0]
        recall = result[1]
        map50 = result[2]
        map = result[3]
        log(f"|{mode+'|':>22s}\t{precision:10.3g}|\t{recall:10.3g}|\t{map50:10.3g}|\t{map:10.3g}|")

    log(f"End value.")


def predict(
        model_name: str, source: str, dataset: str,
        result_dir: str, mode: str = "predict",
        save=True,
        show_conf=False, show=False, show_labels=False):
    model = load(model_name,dataset, "test")
    logger = YoloLogger(model_name=f"{model_name}")
    log = logger.log
    log(f"Start predict on {source} via {result_dir}.")

    results = model(
        source,
        save=save,
        show_conf=show_conf,
        show=show,
        show_labels=show_labels,
        project=f"runs/{result_dir}", name=mode,
        exist_ok=True,
        line_width=1)
    log(f"End predict.")


def heatmap(
        model_name: str, 
        source: str, 
        dataset: str):
    model = load(model_name, dataset, "test")
    logger = YoloLogger(model_name=f"{model_name}")
    log = logger.log
    log(f"Start heatmap of {source} using {model_name}.")

    
    log(f"End heatmap.")