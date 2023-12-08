# RayServe + Triton Inference Server Prototype

The goal of this project is to demonstrate integration with RayServe
based on Triton's BETA in process Python API and bindings.

>**Note** This code is not meant for production and is subject to
>change.

## Included Files

|File|Description|
|----|-----------|
|build.sh| builds a docker image based on Triton 23.11 + RayServe |
|Dockerfile | Dockerfile for Triton 23.11 + RayServe |
|models_test | Simple model repository with simple custom python model |
|run.sh | runs basic image |
| triton_deployment.py | RayServe deployment including embedded Triton Server |
| ./deps/tritonserver-2.41.0.dev0-py3-none-any.whl | BETA Python API

## Python Binding References

| File | Description |
| ---- | ----------- |
| [tritonserver_pybind.cc](https://github.com/triton-inference-server/core/blob/main/python/tritonserver/_c/tritonserver_pybind.cc) | Low level Python API |
| [test_binding.py](https://github.com/triton-inference-server/core/blob/main/python/test/test_binding.py) | Unit tests and example usage |
| [tritonserver/_api/wrapper.py](https://github.com/triton-inference-server/core/blob/nnshah1-python-api/python/tritonserver/_api/wrapper.py) | High level Python API |
| [test_api.py](https://github.com/triton-inference-server/core/blob/nnshah1-python-api/python/test/test_api.py) | Unit test and example usage |

## Building and Running within Docker

The prototype is designed to be run within a docker container using
volume mounting for interactive local development.

### Triton 23.11 + RayServe 2.8

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

## Limitations

* Partial API support 

* Non graceful exit on server.stop()

* Performance not tested
