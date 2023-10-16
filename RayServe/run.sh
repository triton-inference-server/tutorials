docker run --gpus all -it --rm --network host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace -w /workspace rayserve-triton
