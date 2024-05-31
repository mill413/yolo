#!/bin/bash

# ./scripts/run.sh -p baseline --v5 s --v8 s -d visdrone -e 200 --predict | tee last.log

./scripts/run.sh -p siou --v5 s --v8 s -d auav --predict | tee last.log &&
./scripts/run.sh -p p2 --v5 s --v8 s -d auav --predict | tee last.log