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

# Triton Inference Server Fast API / Open API / Open AI Example

## Build Image
```
../../build.sh --framework vllm --build-arg TRITON_CLI_TAG=rmccormick-trtllm-0.9
```

## Import Model
```
triton remove -m all --model-repository llm-models
triton import -m llama-3-8b-instruct --backend vllm --model-repository llm-models
```

## Open AI API Specification

We use
https://raw.githubusercontent.com/openai/openai-openapi/25d9dacc86a94df1db98725fe87494564317cafa/openapi.yaml
as the base specification.

As this tutorial only covers LLM applications we use a trimmed specficiation (api-spec/openai_trimmed.yml).

## Generating the Fast API server using fastapi-codegen

```
./scripts/fastapi-codegen.sh "-i api-spec/openai_trimmed.yml -o fastapi-codegen --model-file openai_protocol_types"
```

### Modifications

1. Remove relative import

Before:

```
from .openapi_protocol_types
```

After:
```
from openapi_protocol_types
```


## Generating the Fast API server using openapi-code-generator


## curl examples

### Models

#### List

```
curl -s http://localhost:8000/models | jq .
```

```
{
  "object": "list",
  "data": [
    {
      "id": "llama-3-8b-instruct",
      "created": 1714952401,
      "object": "model",
      "owned_by": "ACME"
    }
  ]
```
#### Retrieve Model Info

```
curl -s http://localhost:8000/models/llama-3-8b-instruct | jq .
```

```
{
  "id": "llama-3-8b-instruct",
  "created": 1714953302,
  "object": "model",
  "owned_by": "ACME"
}
```
