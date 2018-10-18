#! /bin/bash

caffepath="/home/pangzhanzhong/Software/Caffe_ts/build/"


### train a cnn model
${caffepath}tools/caffe train \
    --solver=config/solver_KD.prototxt \
    --gpu=0 \
    --weights=model/org/u__iter_344688.caffemodel\
    2>&1 | tee info_kd.txt

