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

*NOTE*: The tutorial is intended to be a reference example only. It is a work in progress with [known limitations](#limitations).


## Step 1: Build a Custom Execution Environment with vLLM and other Dependencies

Running vLLM within Triton container requires us to provide [custom execution environment](https://github.com/triton-inference-server/python_backend#creating-custom-execution-environments) with vLLM and other package dependencies. The provided script should build the package environment for you which will be used to load the model in Triton.

```
docker run --gpus all -it --rm -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:23.08-py3 ./gen_vllm_env.sh
```

This step might take a while to build the environment packages. Once complete, the provided sample [model_repository](model_repository) will be populated with `triton_python_backend_stub` and `vllm_env.tar.gz`.

## Step 2: Set Up Triton Inference Server

The structure of the sample model repository as follows after the Step 1 above:
```
model_repository/
`-- vllm
    |-- 1
    |   `-- model.py
    |-- config.pbtxt
    |-- triton_python_backend_stub
    |-- vllm_engine_args.json
    `-- vllm_env.tar.gz

```

A sample model repository for deploying `facebook/opt-125m` using vLLM in Triton is included with this demo as `model_repository` directory. The content of `vllm_engine_args.json` is:

```json
{
    "model": "facebook/opt-125m",
    "disable_log_requests": "true"
}
```
This file can be modified to provide further settings to the vLLM engine. See vLLM [AsyncEngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L165) and [EngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L11) for supported key-value pairs.

*Note*: vLLM greedily consume upto 90% of the GPU's memory under default settings. You can provide appropriate fields like `gpu_memory_utilization` and other settings via [`vllm_engine_args.json`](model_repository/vllm/vllm_engine_args.json).

Read through the documentation in [`model.py`](model_repository/vllm/1/model.py) to understand how to configure this sample for your use-case.

Start the server like below:

```
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 --gpus all -it --rm --shm-size=8G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver --model-store=/models
```

Upon successful start of the server, you can see the following in the end.

```
I0901 23:39:08.729123 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0901 23:39:08.729640 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0901 23:39:08.772522 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Step 3: Using a Triton Client to Query the Server

We will run the client within Triton's SDK container to issue multiple async requests using
[gRPC asyncio client](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/aio/__init__.py)
library.

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
```

Within the container, run the [`client.py`](client.py) as:

```
python3 client.py
```

The client reads prompts from [prompts.txt](prompts.txt) file, sends them to Triton server for inference and stores the results into a file named `results.txt` by default.

The output of the client should look like below:

```
Loading inputs from `prompts.txt`...
Storing results into `results.txt`...
PASS: vLLM example
```

You can inspect the contents of the `results.txt` for the response from the server. `--iterations` flag can be used with the client to increase the load on the server by looping through the list of provided prompts in [`prompts.txt`](prompts.txt).

When you run the client in verbose mode - with `--verbose` flag, the client will print more details about the request/response transactions.

## Limitations

- We use decoupled streaming protocol even if there is exactly 1 response for each request.
- The asyncio implementation is exposed to model.py.
- Does not support multi-GPU systems.
