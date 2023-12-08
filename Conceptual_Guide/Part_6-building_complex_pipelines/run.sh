#!/bin/bash

docker run --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace -v${PWD}/model_repository:/workspace/models -w /workspace rayserve-triton
