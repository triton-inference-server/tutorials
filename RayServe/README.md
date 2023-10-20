# RayServe + Triton Inference Server Prototype

The goal of this project is to demonstrate an early prototype
integration with RayServe based on Triton's low level python bindings.

A thin partial "pythonic wrapper" is also provided as an early preview
of the type of high level api that will be provided in a coming Triton
release.

>**Note** This code is early prototype only and is not meant for
>production and is subject to change.

# Description of Files

|File|Description|
|----|-----------|
|build.sh| builds a docker image based on Triton 23.09 + RayServe |
|build_tensorrt_llm.sh| builds a docker image based on Triton 23.09 + Tensor-RT-LLM + Ray Server |


