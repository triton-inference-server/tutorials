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

# Deploying a GPT-2 Model using Python Backend and Iterative Scheduling

In this tutorial, we will deploy a GPT-2 model using the Python backend and
demonstrate the
[iterative scheduling](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#iterative-sequences)
feature.

## Prerequisites

Before getting started with this tutorial, make sure you're familiar
with the following concepts:

* [Triton-Server Quick Start](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
* [Python Backend](https://github.com/triton-inference-server/python_backend)

## Iterative Scheduling

Iterative scheduling is a technique that allows the Triton Inference Server to
schedule the same request multiple times with the same input. This is useful for
models that have an auto-regressive loop. Iterative scheduling enables Triton
Server to implement inflight batching for your models and gives you the ability
to combine new sequences as they are arriving with inflight sequences.

## Tutorial Overview

In this tutorial we deploy two models:

* simple-gpt2: This model receives a batch of requests and proceeds to the next
batch only when it is done generating tokens for the current batch.

* iterative-gpt2: This model uses iterative scheduling to process
new sequences in a batch even when it is still generating tokens for the
previous sequences

### Demo

[![asciicast](https://asciinema.org/a/TUZtHwZsYrJzHuZF7XCOj1Avx.svg)](https://asciinema.org/a/TUZtHwZsYrJzHuZF7XCOj1Avx)

### Step 1: Prepare the Server Environment

* First, run the Triton Inference Server Container:

```
# Replace yy.mm with year and month of release. Please use 24.04 release upward.
docker run --gpus=all --name iterative-scheduling -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
```

* Next, install all the dependencies required by the models running in the
python backend and login with your [huggingface token](https://huggingface.co/settings/tokens)
(Account on [HuggingFace](https://huggingface.co/) is required).

```
pip install transformers[torch]
```

> [!NOTE]
> Optional: If you want to avoid installing the dependencies each time you run the
> container, you can run `docker commit iterative-scheduling iterative-scheduling-image` to save the container
> and use that for subsequent runs.

Then, start the server:

```
tritonserver --model-repository=/models
```

### Step 2: Install the client side dependencies

In another terminal install the client dependencies:

```
pip3 install tritonclient[grpc]
pip3 install tqdm
```

### Step 3: Run the client

The simple-gpt2 model doesn't use iterative scheduling and will proceed to the
next batch only when it is done generating tokens for the current batch.

Run the following command to start the client:

```
python3 client/client.py --model simple-gpt2
```

As you can see, the tokens for one request are processed first before proceeding
to the next request.

Run `Ctrl+C` to stop the client.


The iterative scheduler is able to incorporate new requests as they are arriving
in the server.

Run the following command to start the client:
```
python3 client/client.py --model iterative-gpt2
```

As you can see, the tokens for both prompts are getting generated simultaneously.

## Next Steps

We plan to integrate KV-Cache with these models for better performance. Currently,
the main goal of tutorial is to demonstrate how to use iterative scheduling with
Python backend.
