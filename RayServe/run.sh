#!/bin/bash

docker run --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -eHF_TOKEN -v ${PWD}:/workspace -v${PWD}/models_test:/workspace/models -w /workspace --name rayserve-triton rayserve-triton
