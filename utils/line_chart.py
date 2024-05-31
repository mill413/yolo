import csv
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def remove_spaces(file: str):
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file+".tmp", 'w') as f:
        for line in lines:
            f.write(line.replace(' ', ''))


def read_from_csv(csvFile: str):
    remove_spaces(csvFile)
    data = {
        "box_loss": [],
        "cls_loss": [],
        "dfl_loss": [],
        "precision": [],
        "recall": [],
        "mAP50": [],
        "mAP50-95": []
    }

    with open(csvFile+".tmp", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["box_loss"].append(float(row["train/box_loss"]))
            data["cls_loss"].append(float(row["train/cls_loss"]))
            data["dfl_loss"].append(float(row["train/dfl_loss"]))
            data["precision"].append(float(row["metrics/precision(B)"]))
            data["recall"].append(float(row["metrics/recall(B)"]))
            data["mAP50"].append(float(row["metrics/mAP50(B)"]))
            data["mAP50-95"].append(float(row["metrics/mAP50-95(B)"]))
    os.remove(csvFile+".tmp")
    return data


def draw_line(dataFile: str, dataName: str, lineColor: str, key: str):
    data = read_from_csv(dataFile)
    data_len = len(data[key])
    plt.plot(
        range(1, data_len+1), data[key],
        color=lineColor, alpha=1,
        linestyle='-', linewidth=1,
        marker='.', markevery=range(1, data_len+1, 10),
        label=dataName
    )


@dataclass
class DataLine:
    file: str
    name: str
    color: str


def generate_pic(
        x_label: str, y_label: str,
        key: str,
        lines: list[DataLine],
        save_file: str = "res.png",
        show=False
):

    plt.figure(figsize=(10, 6))

    for line in lines:
        draw_line(
            line.file,
            line.name,
            line.color,
            key
        )

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))

    plt.legend(fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.xticks(fontsize=10)
    plt.ylabel(y_label, fontsize=20)
    plt.yticks(fontsize=10)

    if show:
        plt.show()

    plt.savefig("runs/"+save_file)


colors = ["red", "lightgreen", "blue", "deepskyblue", "violet"]
random.shuffle(colors)
lines = []
for ind, model in enumerate([]):
    lines.append(DataLine(
        f"runs/{model}/yolov8s/visdrone/train/results.csv",
        model.upper(),
        colors[ind]))

generate_pic("Epoch", "Precision", "precision",
             lines,
             "precision.png")
generate_pic("Epoch", "Recall", "recall",
             lines,
             "recall.png")
generate_pic("Epoch", "mAP@50", "mAP50",
             lines,
             "mAP50.png")
generate_pic("Epoch", "mAP@50:95", "mAP50-95",
             lines,
             "mAP50-95.png")
