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

# Triton Inference Server Ray Serve Deployment

Using the Triton Inference Server In-Process Python API you can
integrate triton server based models into any Python framework
including FastAPI and Ray Serve.

This directory contains an example Triton Inference Server Ray Serve
deployment based on FastAPI.

| [Installation](#installation) | [Run Deployment](#run-ray-serve-deployment) | [Send Requests](#send-requests-to-deployment) |


## Installation

The stable diffusion pipeline is based on the
[Popular_Models_Guide/StableDiffusion](../../../Popular_Models_Guide/StableDiffusion)
tutorial.

### Clone Repository
```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials/Triton_Inference_Server_Python_API
```

### Build Tritonserver Image and Stable Diffusion Models

Please note the following command will take many minutes depending on
your hardware configuration and network connection.

```bash
./build.sh --framework diffusion --build-models
```

## Run Ray Serve Deployment

### Start Container

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
./run.sh --framework diffusion

```

### Run Deployment
```bash
cd examples/rayserve
serve run tritonserver_deployment:tritonserver_deployment
```

## Send Requests to Deployment

The deployment includes two endpoints:

### `/identity`

The identity endpoint accepts a string and returns the same string.

#### Example Request
```
curl --request GET "http://127.0.0.1:8000/identity?string_input=hello_world!"
```

#### Example Output
```bash
"hello_world!"
```


### `/generate`
The generate endpoint accepts a prompt, generates an image based on
the prompt using stable diffusion, and saves the image to a file.

#### Example Request
```
curl --request GET "http://127.0.0.1:8000/generate?prompt=car,model-t,realistic,4k&filename=car_sample.jpg"
```

#### Example Output

![car_sample](../../docs/car_sample.jpg)



