# RayServe + Triton Inference Server Prototype

The goal of this project is to demonstrate integration with RayServe
based on Triton's low level python bindings.

A partial "pythonic wrapper" is provided as an early preview of the
type of high level api that will be provided in a coming Triton
release.

>**Note** This code is not meant for production and is subject to
>change.

## Included Files

|File|Description|
|----|-----------|
|build.sh| builds a docker image based on Triton 23.09 + RayServe |
|build_tensorrt_llm.sh| builds a docker image based on Triton 23.09 + TensorRT-LLM + Ray Server |
|Dockerfile | Dockerfile for Triton 23.09 + RayServe |
|Dockerfile.trt_llm| Dockerfile for Triton 23.09 + TensorRT-LLM + RayServe|
|models_test | Simple model repository with simple custom python model |
|models_trt_llm | config files for TensorRT-LLM gpt2 model |
|build_gpt_trt_llm.sh | builds the engine file for gpt 2 |
|run.sh | runs basic image |
|run_trt_llm.sh| runs TensorRT-LLM enabled image |
| triton_deployment.py | RayServe deployment including embedded Triton Server |
| tritonserver_api.py | "Pythonic" wrapper on top of low level bindings

## Building Docker Environments

The prototype is designed to be run within a docker container using
volume mounting for interactive local development.

### Triton 23.09 + RayServe

#### Build Image
```bash
   ./build.sh
```

