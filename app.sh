#!/bin/bash

./run.sh -p baseline --v5 sm --v8 sm -d visdrone --predict | tee last.log &&
./run.sh -p baseline --v5 sm --v8 sm -d flir --predict | tee last.log &&

./run.sh -p p2 --v5 sm --v8 sm -d visdrone --predict | tee last.log &&
./run.sh -p p2 --v5 sm --v8 sm -d flir --predict | tee last.log &&

./run.sh -p p234 --v5 sm --v8 sm -d visdrone --predict | tee last.log &&
./run.sh -p p234 --v5 sm --v8 sm -d flir --predict | tee last.log &&

./run.sh -p nwd --v5 sm --v8 sm -d visdrone --predict --nwd | tee last.log &&
./run.sh -p nwd --v5 sm --v8 sm -d flir --predict --nwd | tee last.log &&

./run.sh -p attention/cbam --v5 sm --v8 sm -d visdrone --predict | tee last.log &&
./run.sh -p attention/cbam --v5 sm --v8 sm -d flir --predict | tee last.log &&

./run.sh -p attention/gam --v5 sm --v8 sm -d visdrone | tee last.log &&
./run.sh -p attention/gam --v5 sm --v8 sm -d flir | tee last.log &&

./run.sh -p attention/ca --v5 sm --v8 sm -d visdrone | tee last.log &&
./run.sh -p attention/ca --v5 sm --v8 sm -d flir | tee last.log &&

./run.sh -p attention/eca --v5 sm --v8 sm -d visdrone | tee last.log &&
./run.sh -p attention/eca --v5 sm --v8 sm -d flir | tee last.log &&

./run.sh -p attention/se --v5 sm --v8 sm -d visdrone | tee last.log &&
./run.sh -p attention/se --v5 sm --v8 sm -d flir | tee last.log &&

./run.sh -p p234-cbam --v5 sm --v8 sm -d visdrone --predict | tee last.log &&
./run.sh -p p234-cbam --v5 sm --v8 sm -d flir --predict | tee last.log &&

./run.sh -p p234-nwd --v5 sm --v8 sm -d visdrone --predict --nwd | tee last.log &&
./run.sh -p p234-nwd --v5 sm --v8 sm -d flir --predict --nwd | tee last.log &&


./run.sh -p cbam-nwd --v5 sm --v8 sm -d visdrone --predict --nwd | tee last.log &&
./run.sh -p cbam-nwd --v5 sm --v8 sm -d flir --predict --nwd | tee last.log &&


./run.sh -p p234-cbam-nwd --v5 sm --v8 sm -d visdrone --predict --nwd | tee last.log &&
./run.sh -p p234-cbam-nwd --v5 sm --v8 sm -d flir --predict --nwd | tee last.log