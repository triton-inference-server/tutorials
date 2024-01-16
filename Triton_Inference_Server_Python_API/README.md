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

## Requirements

The following instructions require a linux system with Docker
installed. For CUDA support, make sure your CUDA driver meets the
requirements in "NVIDIA Driver" section of Deep Learning Framework
support matrix:
https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

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

The default build includes a `identity` model that can be used for
exercising basic operations including sending input tensors of
different data types. The `identity` model copies provided inputs of
`shape [-1, -1]` to outputs of shape `[-1, -1]`. Inputs are named
`data_type_input` and outputs are named `data_type_output`
(e.g. `string_input`, `string_output`).


## Hello World

### Start Container

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
   ./run.sh
```

#### Example Output

```bash
=============================
== Triton Inference Server ==
=============================

NVIDIA Release 23.12 (build 77457706)
Triton Server Version 2.41.0

Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

NOTE: CUDA Forward Compatibility mode ENABLED.
  Using CUDA 12.3 driver version 545.23.08 with kernel driver version 525.85.12.
  See https://docs.nvidia.com/deploy/cuda-compatibility/ for details.

root@user-machine:/workspace#
```

### Python Shell

```bash
python3
```

#### Example Output

```bash
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
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

### Send an Inference Request

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


## Stable Diffusion

Please note in order to run the stable diffusion example you will need
a hugging face token and need to set the environment variable
`HF_TOKEN` before running the container or set the token by using the
`huggingface-cli login` command after running the container.


#### Build Image and Models

Please note the following command will take many minutes depending on
your hardware configuration and network connection.

```bash
   ./build.sh --framework hf_diffusers --build-models
```

### Start Container

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
   ./run.sh --framework hf_diffusers
```

#### Example Output

```bash
=============================
== Triton Inference Server ==
=============================

NVIDIA Release 23.12 (build 77457706)
Triton Server Version 2.41.0

Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

NOTE: CUDA Forward Compatibility mode ENABLED.
  Using CUDA 12.3 driver version 545.23.08 with kernel driver version 525.85.12.
  See https://docs.nvidia.com/deploy/cuda-compatibility/ for details.

root@user-machine:/workspace#
```

### Python Shell

```bash
python3
```

#### Example Output

```bash
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
### Create and Start a Server Instance

```python
import tritonserver
import numpy
from PIL import Image

server = tritonserver.Server(model_repository="/workspace/models")
server.start()
```

### List Models

```
server.models()
```

#### Example Output
```python
{('stable_diffusion', 1): {'name': 'stable_diffusion', 'version': 1, 'state': 'READY'}, ('test', 1): {'name': 'test', 'version': 1, 'state': 'READY'}, ('text_encoder', 1): {'name': 'text_encoder', 'version': 1, 'state': 'READY'}, ('vae', 1): {'name': 'vae', 'version': 1, 'state': 'READY'}}
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

