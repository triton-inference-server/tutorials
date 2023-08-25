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


# Deploying a vLLM models in Triton

*NOTE*: This tutorial is just for demonstration purpose. There is more follow-up work
that needs to be performed in Triton for more streamline integration with vLLM. 
These implementations can change in future for a better developer experience.

This README showcases how to deploy a simple [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) model on Triton Inference Server with [vLLM](https://github.com/vllm-project/vllm). We will be using Triton's [Python backend](https://github.com/triton-inference-server/python_backend) to host vLLM integration.

## Step 1: Build a Triton Container Image with vLLM

vLLM with all its dependencies is quite large hence it is not feasible to ship these dependencies in a [custom environment](https://github.com/triton-inference-server/python_backend#creating-custom-execution-environments).
We will build a new container image derived from tritonserver:22.12-py3 with all the dependencies needed to run the model.
The dependencies that will be installed in the container image are listed in requirements.txt

```
docker build -t tritonserver_vllm:22.12-py3 .
```

The above command should create tritonserver_vllm:22.12-py3 image with all the dependencies.

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
This file can be modified to provide further settings to the vLLM engine. Look at VLLMAsyncEngineConfig in [model.py](model_repository/vllm/1/model.py) for supported fields.

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

The output of the same should look like below:
```
===========
prompt => 'T h e   f u t u r e   o f   A I   i s'
===========
response => ' not as simple as you think, and you have to understand it in order to'
=========== 

===========
prompt => 'T h e   p r e s i d e n t   o f   t h e   U n i t e d   S t a t e s   i s'
===========
response => ' about to be arrested in Europe for allegedly meddling in the 2016 election.\n\n'
=========== 

===========
prompt => 'T h e   c a p i t a l   o f   F r a n c e   i s'
===========
response => ' becoming a state of chaos with a significant urban and industrial boom. France’'
=========== 

===========
prompt => 'H e l l o ,   m y   n a m e   i s'
===========
response => " Joel. I'm from Massachusetts and live in Melbourne, Australia.\nI'm"
=========== 

PASS: vLLM example

```

When you run the client in verbose mode - with --verbose flag, the client will print more details about current and average inflight request counts. This demonstrate that Triton was able to transfer the full request
load to vLLM engine.

```
[VERBOSE RESPONSE]: {'request_id': '4_1', 'finished': True, 'prompt': 'The future of AI is', 'prompt_token_count': 6, 'completions': [{'index': 0, 'text': ' changing the future.\n\nMicrosoft, Facebook and Google have all pledged to create', 'gen_token_count': 16, 'cumulative_logprob': -38.57662500068545, 'finish_reason': 'length'}], 'current_inflight_count': 4, 'average_inflight_count': 4.0}
[VERBOSE RESPONSE]: {'request_id': '1_1', 'finished': True, 'prompt': 'Hello, my name is', 'prompt_token_count': 6, 'completions': [{'index': 0, 'text': ' Tyler and I have a family, I love to take care of people and to', 'gen_token_count': 16, 'cumulative_logprob': -37.73788612172939, 'finish_reason': 'length'}], 'current_inflight_count': 3, 'average_inflight_count': 3.857142857142857}
[VERBOSE RESPONSE]: {'request_id': '2_1', 'finished': True, 'prompt': 'The president of the United States is', 'prompt_token_count': 8, 'completions': [{'index': 0, 'text': " a psychopath.  I mean, he's a psychopath.  I'd rather", 'gen_token_count': 16, 'cumulative_logprob': -29.792226254940033, 'finish_reason': 'length'}], 'current_inflight_count': 3, 'average_inflight_count': 3.8333333333333335}
[VERBOSE RESPONSE]: {'request_id': '3_1', 'finished': True, 'prompt': 'The capital of France is', 'prompt_token_count': 6, 'completions': [{'index': 0, 'text': ' in danger of being swallowed up in the Parisian and British forces as the French', 'gen_token_count': 16, 'cumulative_logprob': -37.89636680483818, 'finish_reason': 'length'}], 'current_inflight_count': 1, 'average_inflight_count': 3.5714285714285716}

```

## Limitations

- We are restricted to use decoupled streaming protocol even if there is exactly 1 response for each request.
- We are explicitly serializing/deserializing the request/response json objects.
- The asyncio implementation is exposed to model.py.
- Does not support multi-GPU systems.
- Can not use latest Triton containers as vLLM only supports cuda 10.8
