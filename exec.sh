#!/bin/bash
set -x
shopt -s expand_aliases
export TF_CPP_MIN_LOG_LEVEL=3
tensorflow=~/git/DockerFiles/tensorflow/run.sh

for proj in 64 128 256 512; do
    for i in {0..99}; do
        $tensorflow $PWD python3 dd_net.py 360 $proj $i
    done
done
