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
Triton Inference Server using the Triton's
[Python-based](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md#python-based-backends)
[vLLM](https://github.com/triton-inference-server/vllm_backend/tree/main)
backend.

*NOTE*: The tutorial is intended to be a reference example only and has [known limitations](#limitations).


## Step 1: Prepare your model repository

To use Triton, we need to build a model repository. For this tutorial we will
use the model repository, provided in the [samples](https://github.com/triton-inference-server/vllm_backend/tree/main/samples)
folder of the [vllm_backend](https://github.com/triton-inference-server/vllm_backend/tree/main)
repository.

The following set of commands will create a `model_repository/vllm_model/1`
directory and copy 2 files:
[`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json)
and
[`config.pbtxt`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/config.pbtxt),
required to serve the [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) model.
```
mkdir -p model_repository/vllm_model/1
wget -P model_repository/vllm_model/1 https://raw.githubusercontent.com/triton-inference-server/vllm_backend/r<xx.yy>/samples/model_repository/vllm_model/1/model.json
wget -P model_repository/vllm_model/ https://raw.githubusercontent.com/triton-inference-server/vllm_backend/r<xx.yy>/samples/model_repository/vllm_model/config.pbtxt
```
where <xx.yy> is the version of Triton that you want to use. Please note, that Triton's vLLM container has been introduced starting from 23.10 release.

The model repository should look like this:
```
model_repository/
└── vllm_model
    ├── 1
    │   └── model.json
    └── config.pbtxt
```

The content of `model.json` is:

```json
{
    "model": "facebook/opt-125m",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.5
}
```

This file can be modified to provide further settings to the vLLM engine. See vLLM
[AsyncEngineArgs](https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/engine/arg_utils.py#L615)
and
[EngineArgs](https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/engine/arg_utils.py#L21)
for supported key-value pairs. Inflight batching and paged attention is handled
by the vLLM engine.

For multi-GPU support, EngineArgs like `tensor_parallel_size` can be specified
in [`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json).

*Note*: vLLM greedily consume up to 90% of the GPU's memory under default settings.
This tutorial updates this behavior by setting `gpu_memory_utilization` to 50%.
You can tweak this behavior using fields like `gpu_memory_utilization` and other settings
in [`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json).

Read through the documentation in [`model.py`](https://github.com/triton-inference-server/vllm_backend/blob/main/src/model.py)
to understand how to configure this sample for your use-case.

## Step 2: Launch Triton Inference Server

Once you have the model repository setup, it is time to launch the triton server.
Starting with 23.10 release, a dedicated container with vLLM pre-installed
is available on [NGC.](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
To use this container to launch Triton, you can use the docker command below.
```
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 tritonserver --model-store ./model_repository
```
Throughout the tutorial, \<xx.yy\> is the version of Triton
that you want to use. Please note, that Triton's vLLM
container was first published in 23.10 release, so any prior version
will not work.

After you start Triton you will see output on the console showing
the server starting up and loading the model. When you see output
like the following, Triton is ready to accept inference requests.

```
I1030 22:33:28.291908 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1030 22:33:28.292879 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1030 22:33:28.335154 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

## Step 3: Use a Triton Client to Send Your First Inference Request

In this tutorial, we will show how to send an inference request to the
[facebook/opt-125m](https://huggingface.co/facebook/opt-125m) model in 2 ways:

* [Using the generate endpoint](#using-generate-endpoint)
* [Using the gRPC asyncio client](#using-grpc-asyncio-client)

### Using the Generate Endpoint
After you start Triton with the sample model_repository,
you can quickly run your first inference request with the
[generate](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)
endpoint.

Start Triton's SDK container with the following command:
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk bash
```

Now, let's send an inference request:
```
curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
```

Upon success, you should see a response from the server like this one:
```
{"model_name":"vllm_model","model_version":"1","text_output":"What is Triton Inference Server?\n\nTriton Inference Server is a server that is used by many"}
```

### Using the gRPC Asyncio Client
Now, we will see how to run the client within Triton's SDK container
to issue multiple async requests using the
[gRPC asyncio client](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/aio/__init__.py)
library.

This method requires a
[client.py](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/client.py)
script and a set of
[prompts](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/prompts.txt),
which are provided in the
[samples](https://github.com/triton-inference-server/vllm_backend/tree/main/samples)
folder of
[vllm_backend](https://github.com/triton-inference-server/vllm_backend/tree/main)
repository.

Use the following command to download `client.py` and `prompts.txt` to your
current directory:
```
wget https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/samples/client.py
wget https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/samples/prompts.txt
```

Now, we are ready to start Triton's SDK container:
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk bash
```

Within the container, run
[`client.py`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/client.py)
with:
```
python3 client.py
```

The client reads prompts from the
[prompts.txt](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/prompts.txt)
file, sends them to Triton server for
inference, and stores the results into a file named `results.txt` by default.

The output of the client should look like below:

```
Loading inputs from `prompts.txt`...
Storing results into `results.txt`...
PASS: vLLM example
```

You can inspect the contents of the `results.txt` for the response
from the server. The `--iterations` flag can be used with the client
to increase the load on the server by looping through the list of
provided prompts in
[prompts.txt](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/prompts.txt).

When you run the client in verbose mode with the `--verbose` flag,
the client will print more details about the request/response transactions.

## Limitations

- We use decoupled streaming protocol even if there is exactly 1 response for each request.
- The asyncio implementation is exposed to model.py.
- Does not support providing specific subset of GPUs to be used.
- If you are running multiple instances of Triton server with
a Python-based vLLM backend, you need to specify a different
`shm-region-prefix-name` for each server. See
[here](https://github.com/triton-inference-server/python_backend#running-multiple-instances-of-triton-server)
for more information.
