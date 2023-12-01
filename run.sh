#!/bin/zsh
models=("yolov8s" "yolov8-sod" "yolov5s")

for model in ${models[*]}
do
    python train.py --model $model --epochs 200
    python predict.py --model $model 
done
