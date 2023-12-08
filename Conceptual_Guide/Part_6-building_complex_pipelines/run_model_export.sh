#!/bin/bash

docker run --pull missing --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:23.11-py3
