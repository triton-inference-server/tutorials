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
|build_gpt_engine.sh | builds the engine file for gpt 2 |
|run.sh | runs basic image |
|run_trt_llm.sh| runs TensorRT-LLM enabled image |
| triton_deployment.py | RayServe deployment including embedded Triton Server |
| tritonserver_api.py | "Pythonic" wrapper on top of low level bindings

## Building and Running within Docker

The prototype is designed to be run within a docker container using
volume mounting for interactive local development.

### Triton 23.09 + RayServe

#### Build Image
```bash
   ./build.sh
```

#### Supported Backends
```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt
```

#### Run
```bash
  ./run.sh
```

#### Within Docker Container

```bash
python3 triton_deployment.py
```

##### Expected Result
```
(ServeReplica:default:TritonDeployment pid=2736) {'name': 'test', 'version': 1, 'state': 'READY'}
<SNIP>
Hello Theodore!
<SNIP>
{'text_output': 'Theodore', 'fp16_output': [0.5]}
<SNIP>
```

#### Within Docker Container
```bash
serve run triton_deployment:triton_app
```

##### Expected Result
```bash
<SNIP>
ServeReplica:default:TritonDeployment pid=10347) {'name': 'test', 'version': 1, 'state': 'READY'}
<SNIP>
```

#### Interact
```bash
curl "localhost:8000/test?text_input="who%20is%20Triton%20Inference%20Server?"&fp16_input=0.5"
```

##### Expected Result
```bash
<SNIP>
{"text_output":"who is Triton Inference Server?","fp16_output":[0.5]}
```

### Triton 23.09 + TensorRT-LLM + RayServe

#### Build Image
```bash
   ./build_tensorrt_llm.sh
```

#### Supported Backends
```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt  tensorrtllm
```

#### Run
```bash
  ./run_trt_llm.sh
```

#### Build TensorRT-LLM Engine
```bash
./
```

#### Within Docker Container

```bash
python3 triton_deployment.py
```

##### Expected Result
```
(ServeReplica:default:TritonDeployment pid=2736) {'name': 'test', 'version': 1, 'state': 'READY'}
<SNIP>
Hello Theodore!
<SNIP>
{'text_output': 'Theodore', 'fp16_output': [0.5]}
<SNIP>
```

#### Within Docker Container
```bash
serve run triton_deployment:triton_app
```

##### Expected Result
```bash
<SNIP>
ServeReplica:default:TritonDeployment pid=10347) {'name': 'test', 'version': 1, 'state': 'READY'}
<SNIP>
```

#### Interact
```bash
curl "localhost:8000/test?text_input="who%20is%20Triton%20Inference%20Server?"&fp16_input=0.5"
```

##### Expected Result
```bash
<SNIP>
{"text_output":"who is Triton Inference Server?","fp16_output":[0.5]}
```


