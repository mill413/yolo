#!/bin/bash
models=(
# "yolov8m-cbam"
# "yolov8m"
# "yolov8m-p2"
# "yolov5m"
# "yolov5m-resnet"
"yolov5m-p2"
)

dataset="VisDrone"
test_source="/home/wsy_2022301480/jupyterlab/datasets/FLIR/test/data/video-4FRnNpmSmwktFJKjg-frame-000745-L6K5SC6fYjHNC8uff.jpg"
log_file="./logs/$(date +%Y-%m-%d).log"

if [ ! -f "${log_file}" ];then
    touch "${log_file}"
fi

printf "\nProgram Start!" >> "${log_file}"

for model in "${models[@]}";do
    if [ $# -eq 1 ] && [ "$1" == "--test" ];then
        python train.py --model "$model" --dataset $dataset --epochs 1 
        python value.py --model "$model" --dataset $dataset
       	python predict.py --model "$model" --source $test_source
    elif [ $# -eq 0 ];then
        python train.py --model "$model" --dataset $dataset --epochs 200 --workers 8 --batch 16
        python value.py --model "$model" --dataset $dataset
    fi 
done
