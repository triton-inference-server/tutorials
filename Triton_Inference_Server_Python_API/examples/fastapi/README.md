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

# Triton Inference Server Open AI Compatible Server 

Using the Triton In-Process Python API you can integrat triton server
based models into any Python framework including FastAPI with an
OpenAI compatible interface.

This directory contains a FastAPI based Triton Inference Server
supporing `llama-3-8b-instruct` with both the vLLM and TRT-LLM
backends. 

The front end application was generated using a trimmed version of the
OpenAI OpenAPI [specification](api-spec/openai_trimmed.yml) and the
tool [`fastapi-codegen`](scripts/openai_trimmed.yml).

## Installation

The following instructions assume you have a huggingface token set in
the environment variable `HF_TOKEN`.

### Clone Repository
```
git clone https://github.com/triton-inference-server/tutorials.git -b nnshah1-meetup-04-2024
cd tutorials/Triton_Inference_Server_Python_API/examples/fastapi
```
## Triton + vLLM

### Build and Run Image
```
export HF_TOKEN=<hf_token>
../../build.sh --framework vllm
../../run.sh --framework vllm
cd examples/fastapi
```

### Import Model

Note: Model import only has to be done the first time running the server.

```
triton remove -m all --model-repository llama-3-8b-instruct-vllm
triton import -m llama-3-8b-instruct --backend vllm --model-repository llama-3-8b-instruct-vllm
```

### Run Server

```
python3 fastapi-codegen/openai-tritonserver.py --model-repository llama-3-8b-instruct-vllm
```

## Triton + TRT-LLM

### Build and Run Image
```
export HF_TOKEN=<hf_token>
../../build.sh --framework trt_llm
../../run.sh --framework trt_llm
cd examples/fastapi
```

### Import Model

Note: Model import only has to be done the first time running the server.

```
triton remove -m all --model-repository llama-3-8b-instruct-trt-llm
triton import -m llama-3-8b-instruct --backend tensorrtllm --model-repository llama-3-8b-instruct-trt-llm
```

### Run Server

```
python3 fastapi-codegen/openai-tritonserver.py --model-repository llama-3-8b-instruct-trt-llm
```

## Send OpenAI API Requests

#### Completions `/v1/completions`

```
curl -X 'POST' \
    'http://0.0.0.0:8000/v1/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "llama-3-8b-instruct",
    "prompt": "Once upon a time",
    "max_tokens": 16,
    "top_p": 1,
    "n": 1,
    "stream": false,
    "stop": "string",
    "frequency_penalty": 0.0
    }' | jq . 
```

#### Chat Completions `/v1/chat/completions`

```
curl -X 'POST' \
'http://0.0.0.0:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "llama-3-8b-instruct",
    "messages": [
        {
            "role":"user",
            "content":"Hello there how are you?"
        },
        {
            "role":"assistant",
            "content":"Good and you?"
        },
        {
            "role":"user",
            "content":"Whats your name?"
        }
    ],
    "max_tokens": 16,
    "top_p": 1,
    "n": 1,
    "stream": false,
    "stop": "string",
    "frequency_penalty": 0.0
    }' | jq .
```

#### Model List

```
curl -s http://localhost:8000/v1/models | jq .
```

#### Model Info

```
curl -s http://localhost:8000/v1/models/llama-3-8b-instruct | jq .
```

## Comparison to vllm 

The vLLM container can also be used to run the vLLM FastAPI Server

### Run the vLLM Open AI Server
```
python3 -m vllm.entrypoints.openai.api_server --model "meta-llama/Meta-Llama-3-8B-Instruct" --disable-log-requests
```

```
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"meta-llama/Meta-Llama-3-8B-Instruct","messages":[{"role":"system","content":"you are a helpful assistant."},{"role":"user","content":"Hello!"}]}' | jq .
```

## Running GenAI Perf

Note: the following command requires the 24.05 pre-release version of genai-perf.

Preliminary results show performance is on par with vLLM with concurrency 2

```
genai-perf -m meta-llama/Meta-Llama-3-8B-Instruct --endpoint v1/chat/completions --endpoint-type chat --service-kind openai -u http://localhost:8000 --num-prompts 100 --synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 50 --concurrency 2 --measurement-interval 40000 --extra-inputs max_tokens:512 --extra-input ignore_eos:true -- -v --max-threads=256 
erval 40000 --extra-inputs max_tokens:512 --extra-input ignore_eos:true -- -v --max-threads=256
```

## Known Limitations

* Concurrency leads to data corruption
* Max tokens is not processed by trt-llm backend correctly
* Usage information is not populated
* `finish_reason` is currently always set to `stop`
* Limited performance testing has been done 
* Using genai-perf to test streaming requires changes to genai-perf SSE handling
