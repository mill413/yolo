#!/bin/bash
prefix="baseline/"
models=(
    "yolov5n"
    "yolov5s"
    "yolov5m"
    "yolov8n"
    "yolov8s"
)

dataset="flir"
# shellcheck disable=SC2034
# test_source="$DATASETS/SODA-D/images/test/00065.jpg"
log_file="./logs/$(date +%Y-%m-%d).log"

if [ ! -f "${log_file}" ]; then
    touch "${log_file}"
fi

printf "\nProgram Start!\n" >>"${log_file}"

for model in "${models[@]}"; do
    model_path="$prefix$model"
    if [ $# -eq 1 ] && [ "$1" == "--test" ]; then
        python train.py --model "$model_path" --dataset $dataset --epochs 1 --workers 1 --batch 1
        python value.py --model "$model_path" --dataset $dataset
        # python predict.py --model "$model_path" --source "$test_source"
    elif [ $# -eq 0 ]; then
        python train.py --model "$model_path" --dataset $dataset --epochs 200 --workers 8 --batch 8
        python value.py --model "$model_path" --dataset $dataset
    fi
done
