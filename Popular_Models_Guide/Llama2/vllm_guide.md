<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

The vLLM Backend uses vLLM to do inference. Read more about vLLM [here](https://blog.vllm.ai/2023/06/20/vllm.html) and the vLLM Backend [here](https://github.com/triton-inference-server/vllm_backend).

## Pre-build instructions

For this tutorial, we are using the Llama2-7B HuggingFace model with pre-trained weights. Please follow the [README.md](README.md) for pre-build instructions and links for how to run Llama with other backends.

## Installation

The triton vLLM container can be pulled from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) with

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v $PWD/llama2vllm:/opt/tritonserver/model_repository/llama2vllm \
    nvcr.io/nvidia/tritonserver:23.11-vllm-python-py3
```
This will create a `/opt/tritonserver/model_repository` folder that contains the `llama2vllm` model. The model itself will be pulled from the HuggingFace

Once in the container, install the `huggingface-cli` and login with your own credentials.
```bash
pip install --upgrade huggingface_hub
huggingface-cli login --token <your huggingface access token>
```


## Serving with Triton

Then you can run the tritonserver as usual
```bash
tritonserver --model-repository model_repository
```
The server has launched successfully when you see the following outputs in your console:

```
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Sending requests via the `generate` endpoint

As a simple example to make sure the server works, you can use the `generate` endpoint to test. More about the generate endpoint [here](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md).

```bash
$ curl -X POST localhost:8000/v2/models/llama2vllm/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
# returns (formatted for better visualization)
> {
    "model_name":"llama2vllm",
    "model_version":"1",
    "text_output":"What is Triton Inference Server?\nTriton Inference Server is a lightweight, high-performance"
  }
```

## Sending requests via the Triton client

The Triton vLLM Backend repository has a [samples folder](https://github.com/triton-inference-server/vllm_backend/tree/main/samples) that has an example client.py to test the Llama2 model.

```bash
pip3 install tritonclient[all]
# Assuming Tritonserver server is running already
$ git clone https://github.com/triton-inference-server/vllm_backend.git
$ cd vllm_backend/samples
$ python3 client.py -m llama2vllm

```
The following steps should result in a `results.txt` that has the following content
```bash
Hello, my name is
I am a 20 year old student from the Netherlands. I am currently

=========

The most dangerous animal is
The most dangerous animal is the one that is not there.
The most dangerous

=========

The capital of France is
The capital of France is Paris.
The capital of France is Paris. The

=========

The future of AI is
The future of AI is in the hands of the people who use it.

=========
```