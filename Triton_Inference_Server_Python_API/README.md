<!--
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Triton Inference Server In-Process Python API [BETA]

Starting with release 24.01 Triton Inference Server will include a
Python package enabling developers to embed Triton Inference Server
instances in their Python applications. The in-process Python API is
designed to match the functionality of the in-process C API while
providing a higher level abstraction. At its core the API relies on a
1:1 python binding of the C API and provides all the flexibility and
power of the C API with a simpler to use interface.

This tutorial repository includes a preview of the API based on the
23.12 release of Triton.

> [!Note]
> As the API is in BETA please expect some changes as we
> test out different features and get feedback.
> All feedback is weclome and we look forward to hearing from you!

| [Requirements](#requirements) | [Installation](#installation) | [Hello World](#hello-world) | [Stable Diffusion](#stable-diffusion) | [Ray Serve Deployment](examples/rayserve) |

## Requirements

The following instructions require a linux system with Docker
installed. For CUDA support, make sure your CUDA driver meets the
requirements in "NVIDIA Driver" section of Deep Learning Framework
support matrix:
https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

## Installation

The tutorial and Python API package are designed to be installed and
run within the `nvcr.io/nvidia/tritonserver:23.12-py3` docker image.

A set of convenience scripts are provided to create a docker image
based on the `nvcr.io/nvidia/tritonserver:23.12-py3` image with the
Python API installed plus additional dependencies required for the
examples.

### Trition Inference Server 23.12 + Python API

#### Clone Repository
```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials/Triton_Inference_Server_Python_API
```

#### Build `triton-python-api:r23.12` Image
```bash
./build.sh
```

#### Supported Backends

The built image includes all the backends shipped by default in the
tritonserver `nvcr.io/nvidia/tritonserver:23.12-py3` container.

```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt
```

#### Included Models

The `default` build includes an `identity` model that can be used for
exercising basic operations including sending input tensors of
different data types. The `identity` model copies provided inputs of
`shape [-1, -1]` to outputs of shape `[-1, -1]`. Inputs are named
`data_type_input` and outputs are named `data_type_output`
(e.g. `string_input`, `string_output`, `fp16_input`, `fp16_output`).


## Hello World

### Start `triton-python-api:r23.12` Container

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
./run.sh
```

### Enter Python Shell

```bash
python3
```

### Create and Start a Server Instance

```python
import tritonserver

server = tritonserver.Server(model_repository="/workspace/identity-models")
server.start()
```

### List Models

```
server.models()
```

#### Example Output

`server.models()` returns a dictionary of the available models with
their current state.

```python
{('identity', 1): {'name': 'identity', 'version': 1, 'state': 'READY'}}
```

### Send an Inference Request

```python
model = server.model("identity")
responses = model.infer(inputs={"string_input":[["hello world!"]]})
```

### Iterate through Responses
`model.infer()` returns an iterator that can be used to process the
results of an inference request.

```python
for response in responses:
    print(response.outputs["string_output"].to_string_array())
```

#### Example Output
```python
[['hello world!']]
```


## Stable Diffusion

Please note in order to run the stable diffusion example you will need
a hugging face token and need to set the environment variable
`HF_TOKEN` before running the container or set the token by using the
`huggingface-cli login` command after running the container.

This example is based on the
[building_complex_pipelines](/Conceptual_Guide/Part_6-building_complex_pipelines)
tutorial.


#### Build `triton-python-api:r23.12-diffusers` Image and Stable Diffusion Models

Please note the following command will take many minutes depending on
your hardware configuration and network connection.

```bash
   ./build.sh --framework diffusers --build-models
```

#### Supported Backends

The built image includes all the backends shipped by default in the
tritonserver `nvcr.io/nvidia/tritonserver:23.12-py3` container.

```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt
```

#### Included Models

The `diffusers` build includes a `stable_diffustion` pipeline that
takes a text prompt and returns a generated image. For more details on
the models and pipeline please see the
[building_complex_pipelines](/Conceptual_Guide/Part_6-building_complex_pipelines)
tutorial.

### Start Container

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
./run.sh --framework diffusers
```

### Enter Python Shell

```bash
python3
```

### Create and Start a Server Instance

```python
import tritonserver
import numpy
from PIL import Image

server = tritonserver.Server(model_repository="/workspace/diffuser-models")
server.start()
```

### List Models

```
server.models()
```

#### Example Output
```python
{('stable_diffusion', 1): {'name': 'stable_diffusion', 'version': 1, 'state': 'READY'}, ('text_encoder', 1): {'name': 'text_encoder', 'version': 1, 'state': 'READY'}, ('vae', 1): {'name': 'vae', 'version': 1, 'state': 'READY'}}
```

### Send an Inference Request

```python
model = server.model("stable_diffusion")
responses = model.infer(inputs={"prompt":[["butterfly in new york, realistic, 4k, photograph"]]})
```

### Iterate through Responses and save image


```python
for response in responses:
	generated_image = numpy.from_dlpack(response.outputs["generated_image"])
	generated_image = generated_image.squeeze().astype(numpy.uint8)
	image_ = Image.fromarray(generated_image)
	image_.save("sample_generated_image.jpg")
```

#### Example Output

![sample_generated_image](./docs/sample_generated_image.jpg)

