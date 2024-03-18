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

# parse arguments
args=$(getopt -o thp:d: --long dataset:,prefix:,test,help,v5:,v8: -n "$0" -- "$@")
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
            # wheth prefix is valid
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
            # wheth dataset is valid
            if [ "$dataset" == "" ]; then
                echo "Lack of dataset option!"
                exit 0
            elif [ ! -f "./datasets/$dataset.yaml" ]; then
                echo "File ./datasets/$dataset.yaml doesn't exist!"
                exit 0
            fi
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
        --) 
            shift
            break
            ;;
    esac
done

# echo "prefix=$prefix test=$test dataset=$dataset models=${models[*]}"

log_file="./logs/$(date +%Y-%m-%d).log"
if [ ! -f "${log_file}" ]; then
    touch "${log_file}"
fi

printf "\n>>> Models (%s) in %s on %s Start! <<<\n" "${models[*]}" "$prefix" "${dataset^^}" >>"${log_file}"

for model in "${models[@]}"; do
    model_path="$prefix$model"
    if [ $# -eq 1 ] && [ $test == 1 ]; then
        python train.py --model "$model_path" --dataset "$dataset" --epochs 1 --workers 1 --batch 1
        python value.py --model "$model_path" --dataset "$dataset"
    elif [ $# -eq 0 ]; then
        python train.py --model "$model_path" --dataset "$dataset" --epochs 200 --workers 8 --batch 8
        python value.py --model "$model_path" --dataset "$dataset"
    fi
done
