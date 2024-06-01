#!/bin/bash

help(){
    info="Usage: ./run.sh -p path/to/models -d path/to/dataset [OPTION] \n\
    Options:\n \
        [ -t | --test] \t\t Test models with 1 epoch, 1 worker, 1 batch \n \
        [ -p | --prefix] \t\t Models' relative path \n \
        [ -d | --dataset ] \t\t Dataset's path \n \
        [ --v5 ] \t\t The scales of yolov5. \n \
        [ --v8 ] \t\t The scales of yolov8. \n \
        [ -h | --help] \t\t Get options info \n \
        "
    echo -e "$info"
    exit 0
}

prefix=""
test=0
dataset=""
models=()
batch=8
predict=0
epochs=200
declare -A source
source=(
    [visdrone]="../datasets/VisDrone/VisDrone2019-DET-test-dev/images/"
    [flir]="../datasets/FLIR/test/data/"
    [auav]="../datasets/anti-uav/images/test/20190925_124000_1_6"
)

# parse arguments
args=$(getopt -o thp:d:e --long dataset:,prefix:,test,help,v5:,v8:,batch:,predict,epochs -n "$0" -- "$@")
eval set -- "${args}"
while true; do
    case "$1" in 
        -h|--help)
            help
            ;;
        -p|--prefix)
            prefix="$2"
            if [[ "$prefix" != *"/" ]]; then
                prefix="$prefix/"
            fi
            # whether prefix is valid
            if [ "$prefix" == "" ]; then
                echo "Lack of prefix option!"
                exit 0
            elif [ ! -d "./models/$prefix" ]; then
                echo "Directory ./models/$prefix doesn't exist!"
                exit 0
            fi
            shift 2
            ;;
        -t|--test)
            test=1
            shift
            ;;
        -d|--dataset)
            dataset="$2"
            # whether dataset is valid
            if [ "$dataset" == "" ]; then
                echo "Lack of dataset option!"
                exit 0
            elif [ ! -f "./datasets/$dataset.yaml" ]; then
                echo "File ./datasets/$dataset.yaml doesn't exist!"
                exit 0
            fi
            shift 2
            ;;
        -e|--epochs)
            epochs="$2"
            shift 2
            ;;
        --v5)
            v5_scales="$2"
            if [ "$v5_scales" == "" ]; then
                v5_scales="n"
            fi
            for i in $(seq ${#v5_scales}); do
                scale=${v5_scales:$i-1:1}
                models+=("yolov5$scale")
            done
            shift 2
            ;;
        --v8)
            v8_scales="$2"
            if [ "$v8_scales" == "" ]; then
                v8_scales="n"
            fi
            for i in $(seq ${#v8_scales}); do
                scale=${v8_scales:$i-1:1}
                models+=("yolov8$scale")
            done
            shift 2
            ;;
        --batch)
            batch="$2"
            if [ "$batch" == "" ]; then
                echo "batch number is need!"
                exit 0
            fi
            shift 2
            ;;
        --predict)
            if [[ -n ${source[$dataset]} ]];then
                predict=1
            else
                predict=0
            fi
            shift
            ;;
        --) 
            shift
            break
            ;;
    esac
done

for model in "${models[@]}"; do
    model_name="$prefix$model"
    if [ $test == 1 ]; then
        python train.py --model "$model_name" --dataset "$dataset" --epochs 1 --workers 8 --batch 8
        python value.py --model "$model_name" --dataset "$dataset"
    elif [ $# -eq 0 ]; then
        # if model's scale is m, batch should be 8
        if [[ "$model" == *"m" ]];then
            batch=8
        fi

        python train.py --model "$model_name" --dataset "$dataset" --epochs "$epochs" --workers 8 --batch "$batch" &&
        python value.py --model "$model_name" --dataset "$dataset"
    fi

    if [ $predict == 1 ];then
        python predict.py --model "${model_name}" --dataset "${dataset}" --source "${source[$dataset]}"
    fi
done
