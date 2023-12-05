#!/bin/zsh
models=(
# "yolov8m-cbam"
# "yolov8m"
# "yolov8m-sod"
"yolov5m"
"yolov5m-resnet"
)

log_file="./logs/$(date +%Y-%m-%d).log"

if [ ! -f ${log_file} ];then
    touch ${log_file}
fi

echo "\nProgram Start!" >> ${log_file}

for model in ${models[*]};do
    if [ $# -eq 1 ] && [ $1=="--test" ];then
        python train.py --model $model --epochs 1
        python value.py --model $model
       	python predict.py --model $model --source "/home/wsy_2022301480/jupyterlab/datasets/VisDrone/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg"
    elif [ $# -eq 0 ];then
        python train.py --model $model --epochs 200
        python value.py --model $model
        python predict.py --model $model
    fi 
done
