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

*NOTE*: The tutorial is intended to be a reference example only. It is a work in progress with
[known limitations](#limitations).


## Step 1: Build a Triton Container Image with vLLM

We will build a new container image derived from tritonserver:23.08-py3 with vLLM.

```
docker build -t tritonserver_vllm .
```

The above command should create the tritonserver_vllm image with vLLM and all of its dependencies.


## Step 2: Start Triton Inference Server

A sample model repository for deploying `facebook/opt-125m` using vLLM in Triton is 
included with this demo as `model_repository` directory. 
The model repository should look like this:
```
model_repository/
`-- vllm
    |-- 1
    |   `-- model.py
    |-- config.pbtxt
    |-- vllm_engine_args.json
```

The content of `vllm_engine_args.json` is:

```json
{
    "model": "facebook/opt-125m",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.5
}
```
This file can be modified to provide further settings to the vLLM engine. See vLLM
[AsyncEngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L165)
and
[EngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L11)
for supported key-value pairs.

For multi-GPU support, EngineArgs like `tensor_parallel_size` can be specified in [`vllm_engine_args.json`](model_repository/vllm/vllm_engine_args.json).

*Note*: vLLM greedily consume upto 90% of the GPU's memory under default settings.
This tutorial updates this behavior by setting `gpu_memory_utilization` to 50%.
You can tweak this behavior using fields like `gpu_memory_utilization` and other settings
in [`vllm_engine_args.json`](model_repository/vllm/vllm_engine_args.json).

Read through the documentation in [`model.py`](model_repository/vllm/1/model.py) to understand how
to configure this sample for your use-case.

Run the following commands to start the server container:

```
docker run --gpus all -it --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work tritonserver_vllm tritonserver --model-store ./model_repository
```

Upon successful start of the server, you should see the following at the end of the output.

```
I0901 23:39:08.729123 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0901 23:39:08.729640 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0901 23:39:08.772522 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Step 3: Use a Triton Client to Query the Server

We will run the client within Triton's SDK container to issue multiple async requests using the
[gRPC asyncio client](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/aio/__init__.py)
library.

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
```

Within the container, run [`client.py`](client.py) with:

```
python3 client.py
```

The client reads prompts from the [prompts.txt](prompts.txt) file, sends them to Triton server for
inference, and stores the results into a file named `results.txt` by default.

The output of the client should look like below:

```
Loading inputs from `prompts.txt`...
Storing results into `results.txt`...
PASS: vLLM example
```

You can inspect the contents of the `results.txt` for the response from the server. The `--iterations`
flag can be used with the client to increase the load on the server by looping through the list of
provided prompts in [`prompts.txt`](prompts.txt).

When you run the client in verbose mode with the `--verbose` flag, the client will print more details
about the request/response transactions.

## Limitations

- We use decoupled streaming protocol even if there is exactly 1 response for each request.
- The asyncio implementation is exposed to model.py.
- Does not support providing specific subset of GPUs to be used.
