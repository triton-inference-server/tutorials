# Triton Inference Server In-Process Python API [BETA]

Starting with release r24.01 Triton Inference Server will include a
Python package enabling developers to embed Triton Inference Server
instances in their Python applications. The in-process Python API is
designed to match the functionality of the in-process C API while
providing a higher level abstraction. At its core in fact the API
relies on a 1:1 python binding of the C API created with pybind-c++
and thus provides all the flexibility and power of the C API with a
simpler to use interface. 

This tutorial repository includes a preview of the API based on the
r23.12 release of Triton.

> [!Note]
> As the API is in BETA please expect some changes as we 
> test out different features and get feedback.
> All feedback is weclome and we look forward to hearing from you!

## Requirements

The following instructions require a linux system with bash and Docker
installed. For GPU support a CUDA device compatible with CUDA 12.x is
required.

## Build / Installation

The tutorial and Python API package are designed to be installed and
run within the `nvcr.io/nvidia/tritonserver:23.12-py3` docker image.

A set of convenience scripts are provided to create a docker image
based on the `nvcr.io/nvidia/tritonserver:23.12-py3` image with the
Python API installed plus additional dependencies required for the
examples.

### Trition Inference Server 23.12 + Python API

#### Build Image
```bash
   ./build.sh
```

#### Example Output
```bash
#18 naming to docker.io/library/triton-python-api:r23.12 0.0s done
#18 DONE 0.2s
+ [[ TEST == TRT_LLM ]]
+ [[ TEST == TEST ]]
+ mkdir -p /home/user/tutorials/Triton_Inference_Server_Python_API/models
+ cp -rf /home/user/tutorials/Triton_Inference_Server_Python_API/deps/test/test_api_models/test /home/user/tutorials/Triton_Inference_Server_Python_API/models/.
```

#### Supported Backends
```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt
```

#### Included Models

The default build includes a `test` model that can be used for
exercising basic operations including sending input tensors of
different data types. The `test` model copies provided inputs of
`shape [-1, -1]` to outputs of shape `[-1, -1]`. Inputs are named
`data_type_input` and outputs are named `data_type_output`
(e.g. `string_input`, `string_outpu`).


## Hello World

### Start Container and Python Shell

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
   ./run.sh
   python3
```

### Create and Start a Server Instance

```python
import tritonserver

server = tritonserver.Server(model_repository="/workspace/models")
server.start()
```

### List Models

```
server.models()
```

#### Example Output
```python
{('test', 1): {'name': 'test', 'version': 1, 'state': 'READY'}}
```

### Send an Inference

```python
model = server.model("test")
responses = model.infer(inputs={"string_input":[["hello world!"]]})
```

### Iterate through Responses

```python
for response in responses:
    print(response.outputs["string_output"].to_string_array())
```

#### Example Output
```python
[['hello world!']]
```
