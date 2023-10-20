#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath "$0"))

BUILD_DIR=.tensorrt_llm-build

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git

# Update the submodules
cd tensorrtllm_backend
git submodule set-url tensorrt_llm https://github.com/NVIDIA/TensorRT-LLM.git
git submodule update --init --recursive
git lfs install
git lfs pull

# Use the Dockerfile to build the backend in a container
# For x86_64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm --build-arg BASE_TAG=23.09-py3 -f dockerfile/Dockerfile.trt_llm_backend .

echo $SCRIPT_DIR
cd $SCRIPT_DIR

docker build --build-arg BASE_IMAGE=triton_trt_llm --build-arg BASE_IMAGE_TAG=latest -t rayserve-triton-trt-llm -f Dockerfile.trt_llm .

