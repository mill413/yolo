#!/bin/bash
models=(
# "yolov5m"
# "yolov5s-p234"
# "yolov5m-cbam"
# "yolov5s-cbam-p234"
"yolov8s-acmix"
)

dataset="VisDrone"
test_source="/home/haruto/Document/datasets/VisDrone/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg"
log_file="./logs/$(date +%Y-%m-%d).log"

if [ ! -f "${log_file}" ];then
    touch "${log_file}"
fi

printf "\nProgram Start!" >> "${log_file}"

for model in "${models[@]}";do
    if [ $# -eq 1 ] && [ "$1" == "--test" ];then
        python train.py --model "$model" --dataset $dataset --epochs 1 --workers 4 --batch 2
        python value.py --model "$model" --dataset $dataset
       	# python predict.py --model "$model" --source $test_source
    elif [ $# -eq 0 ];then
        python train.py --model "$model" --dataset $dataset --epochs 200 --workers 8 --batch 8
        python value.py --model "$model" --dataset $dataset
    fi 
done
