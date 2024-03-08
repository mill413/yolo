from pathlib import Path
import time
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params
import traceback

split_line = "------------------------------------------"


def log(msg: str):
    date = time.strftime("%Y-%m-%d", time.localtime())
    log_file = Path(f"./logs/{date}.log")
    log_file.touch()

    with open(log_file, mode="a+") as log:
        print(msg)
        log.write(msg+"\n" if not msg.endswith("\n") else msg)


def load(model_name: str):
    model = YOLO(f"./models/{model_name}.yaml")
    current_time = time.strftime("%H:%M:%S", time.localtime())
    log(f"{split_line} Load model {model_name} at {current_time} {split_line}")
    return model


def train(
        model: YOLO, dataset: str,
        project: str, name: str = "train",
        epochs=100, exist_ok=False,
        batch=16, device=0, workers=2,
        patience=50
):
    log(f"Start train model on {dataset}.\n" +
        f"Epochs: {epochs} Batch: {batch} Workers: {workers} Exist_ok: {exist_ok} Device: {device} " +
        f"Paras: {get_num_params(model)/1000000:.3f}M\n" +
        f"Save to runs/{project}/{name}")
    try:
        model.train(
            data=f"./datasets/{dataset}.yaml",
            epochs=epochs,
            project=f"runs/{project}", name=name,
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
        model: YOLO, dataset: str,
        project: str, name: str = "val",
        batch=16, device=0, workers=2, exist_ok=False
):
    log(f"Start value model on {dataset}.")
    metrics = model.val(
        data=f"./datasets/{dataset}.yaml",
        device=device,
        batch=batch,
        workers=workers,
        project=f"runs/{project}", name=name,
        exist_ok=exist_ok)
    
    # print value results
    class_names = model.names
    log(f"{'Class':>22s}\t{'Precision':>11s}\t{'Recall':>11s}\t{'mAP50':>11s}\t{'mAP50-95':>11s}")
    mean_result = metrics.mean_results()
    log(f"{'all':>22s}\t{mean_result[0]:11.3g}\t{mean_result[1]:11.3g}\t{mean_result[2]:11.3g}\t{mean_result[3]:11.3g}")
    for i, c in enumerate(metrics.ap_class_index):
        name = class_names[c]
        result = metrics.class_result(i)
        precision = result[0]
        recall = result[1]
        map50 = result[2]
        map = result[3]
        log(f"{name:>22s}\t{precision:11.3g}\t{recall:11.3g}\t{map50:11.3g}\t{map:11.3g}")

    log(f"End value.")


def predict(model: YOLO, source: str, project: str, name: str = "predict",
            save=True, show_conf=False, show=False):
    log(f"Start predict on {source} via {project}.")
    results = model(
        source,
        save=save,
        show_conf=show_conf,
        show=show,
        project=f"runs/{project}", name=name,
        exist_ok=True,
        line_width=1)
    log(f"End predict.")
