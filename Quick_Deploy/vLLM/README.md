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


# Deploying a vLLM model in Triton

The following tutorial demonstrates how to deploy a simple
[facebook/opt-125m](https://huggingface.co/facebook/opt-125m) model on
Triton Inference Server using Triton's [Python backend](https://github.com/triton-inference-server/python_backend) and the
[vLLM](https://github.com/vllm-project/vllm) library.

*NOTE*: The tutorial is a work in progress with known limitations and
is expected to change and improve over time. It is not intended to be
used in production.


## Step 1: Build a Triton Container Image with vLLM


We will build a new container image derived from `tritonserver:22.12-py3` with all the dependencies needed to run the model.
The dependencies that will be installed in the container image are listed in `requirements.txt`.

```
docker build -t tritonserver_vllm:22.12-py3 .
```

The above command should create `tritonserver_vllm:22.12-py3` image with all the dependencies.

*NOTE*: A [custom execution environment](https://github.com/triton-inference-server/python_backend#creating-custom-execution-environments) with vLLM and all the dependencies
specified in `requirement.txt` can be created and shipped along with the model instead of
building a new container image.

## Step 2: Set Up Triton Inference Server

To use Triton, we need to build a model repository. The structure of the repository as follows:
```
model_repository
|
+-- vllm
    |
    +-- config.pbtxt
    +-- vllm_engine_args.json
    +-- 1
        |
        +-- model.py
```

A sample model repository for deploying `facebook/opt-125m` using vLLM in Triton is included with this demo as `model_repository` directory. The content of `vllm_engine_args.json` is:

```json
{
    "model":"facebook/opt-125m"
}
```
This file can be modified to provide further settings to the vLLM engine. See vLLM EngineArgs for options..

Read through the documentation in [`config.pbtxt`](model_repository/vllm/config.pbtxt) and [`model.py`](model_repository/vllm/1/model.py) to
understand how to configure this sample for your use-case.


```
docker run --gpus all --rm --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models tritonserver_vllm:22.12-py3 tritonserver --model-repository=/models
```



## Step 3: Using a Triton Client to Query the Server

We will run the client within Triton's SDK container to issue multiple async requests using
[gRPC asyncio client](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/aio/__init__.py)
library.

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.12-py3-sdk bash
```

Within the container, run the [`client.py`](client.py) as:

```
python3 client.py

```

The output of the client should look like below:
```
===========
prompt => 'The future of AI is'
===========
response => ' not as simple as you think, and you have to understand it in order to'
=========== 
...

<SNIP>
...

PASS: vLLM example
```

When you run the client in verbose mode - with `--verbose` flag, the client will print more details about current and average inflight request counts. This demonstrates that Triton was able to transfer all the four request prompts to vLLM engine.

```
[VERBOSE RESPONSE]: {'request_id': '4_1', 'finished': True, 'prompt': 'The future of AI is', 'prompt_token_count': 6, 'completions': [{'index': 0, 'text': ' an exciting project, but it is still in its infancy\nWhen the world of', 'gen_token_count': 16, 'cumulative_logprob': -37.039882481098175, 'finish_reason': 'length'}], 'current_inflight_count': 4, 'average_inflight_count': 4.0}
...

```

## Limitations

- We use decoupled streaming protocol even if there is exactly 1 response for each request.
- We are explicitly serializing/deserializing the request/response json objects.
- The asyncio implementation is exposed to model.py.
- Does not support multi-GPU systems.
- Can not use latest Triton containers as vLLM only supports cuda 11.8.
