#!/bin/bash
models=(
    # "yolov5n-p2"
    "yolov5s-p2"
    "yolov5m-p2"
    "yolov8n-p2"
    "yolov8s-p2"
)

dataset="flir"
# shellcheck disable=SC2034
test_source="$DATASETS/SODA-D/images/test/00065.jpg"
log_file="./logs/$(date +%Y-%m-%d).log"

if [ ! -f "${log_file}" ]; then
    touch "${log_file}"
fi

printf "\nProgram Start!\n" >>"${log_file}"

for model in "${models[@]}"; do
    if [ $# -eq 1 ] && [ "$1" == "--test" ]; then
        python train.py --model "$model" --dataset $dataset --epochs 1 --workers 8 --batch 16
        python value.py --model "$model" --dataset $dataset
        python predict.py --model "$model" --source "$test_source"
    elif [ $# -eq 0 ]; then
        python train.py --model "$model" --dataset $dataset --epochs 200 --workers 8 --batch 8
        python value.py --model "$model" --dataset $dataset
    fi
done
