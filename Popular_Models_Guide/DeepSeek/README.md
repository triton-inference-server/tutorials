<!--
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Deploying DeepSeek-R1-Distill-Llama-8B model with Triton

In this tutorial we'll use vLLM Backend to deploy
[`DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).
Read more about vLLM [here](https://blog.vllm.ai/2023/06/20/vllm.html) and
the vLLM Backend [here](https://github.com/triton-inference-server/vllm_backend).

## Model Repository

Let's first set up a model repository. In this tutorial we'll use the sample
model repository, provided in the [Triton vLLM backend repository.](https://github.com/triton-inference-server/vllm_backend/tree/main/samples/model_repository/vllm_model)

You can clone the full repository with:
```bash
git clone -b r25.01 https://github.com/triton-inference-server/vllm_backend.git
```

The sample model repository uses [`facebook/opt-125m` model,](https://github.com/triton-inference-server/vllm_backend/blob/80dd0371e0301fabf79c57536e60700d016fcc76/samples/model_repository/vllm_model/1/model.json#L2)
let's replace it with `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"`.
Additionally, please note, that with the default parameters it's important to adjust `gpu_memory_utilization` appropriately to
your hardware. Please note, that with all default parameters
`"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"` needs about 35GB of memory to be
deployed via Triton + vLLM backend, make sure to adjust "gpu_memory_utilization"
accordingly. For example, for RTX 5880 the minimum value should be `0.69`, at
the same time `0.41` is sufficient for A100. For the simplicity of this
tutorial, we'll set this number to `0.9`. The resulting `model.json` should
look like:
```json
{
    "model":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "disable_log_requests": true,
    "gpu_memory_utilization": 0.9,
    "enforce_eager": true
}
```

## Serving with Triton

Then you can run the tritonserver as usual
```bash
LOCAL_MODEL_REPOSITORY=./vllm_backend/samples/model_repository/
docker run --rm -it --net host --shm-size=2g  --ulimit memlock=-1 \
--ulimit stack=67108864 --gpus all -v $LOCAL_MODEL_REPOSITORY:/opt/tritonserver/model_repository  \
nvcr.io/nvidia/tritonserver:25.01-vllm-python-py3 tritonserver --model-repository=model_repository/
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
$ curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0, "exclude_input_in_output": true, "max_tokens": 45}}' | jq
```
The expected output should look like:
```json
{
  "model_name": "vllm_model",
  "model_version": "1",
  "text_output": " It's a high-performance, scalable, and efficient inference server for AI models. It's designed to handle large numbers of requests quickly and efficiently, making it suitable for real-time applications like autonomous vehicles, smart homes, and more"
}
```

## Sending requests via the Triton client

The Triton vLLM Backend repository has a [samples folder](https://github.com/triton-inference-server/vllm_backend/tree/main/samples)
that has an example client.py to test the model.

```bash
LOCAL_WORKSPACE=./vllm_backend/samples
docker run -ti --gpus all --network=host --pid=host --ipc=host -v $LOCAL_WORKSPACE:/workspace nvcr.io/nvidia/tritonserver:25.01-py3-sdk
```
Then you can use client as follows:
```bash
python client.py -m vllm_model
```

The following steps should result in a `results.txt` that has the following content
```
Hello, my name is
I need to write a program that can read a text file and find all the names in the text. The names can be in any case (uppercase, lowercase, or mixed). Also, the names can be part of longer words or phrases, so I need to make sure that I'm extracting only the names and not parts of other words. Additionally, the names can be separated by various non-word characters, such as commas, periods, apostrophes, etc. So, I need to extract

=========

The most dangerous animal is
The most dangerous animal is the one that poses the greatest threat to human safety and well-being. This can vary depending on the region and the specific circumstances. For example, in some areas, large predators like lions or tigers might be considered the most dangerous, while in others, venomous snakes or dangerous marine animals might take precedence.

To determine the most dangerous animal, one would need to consider factors such as:
1. **Number of incidents**: How many people have been injured or killed by this

=========

The capital of France is
A) London
B) Paris
C) Marseille
D) Lyon

Okay, so I have this question here: "The capital of France is..." with options A) London, B) Paris, C) Marseille, D) Lyon. Hmm, I need to figure out the correct answer. Let me think about what I know regarding the capitals of different countries.

First off, I remember that France is a country in Western Europe. I've heard people talk about Paris before, especially in

=========

The future of AI is
AI is the future of everything. It's going to change how we live, work, and interact with the world. From healthcare to education, from transportation to entertainment, AI will play a crucial role in shaping our tomorrow. But what does that mean for us? How will AI impact our daily lives? Let's explore some possibilities.

First, in healthcare, AI can help diagnose diseases faster and more accurately than ever before. It can analyze medical data, recommend treatments, and even assist in surgery.

=========
```