<!--
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Deploying a Vision-Language Model (Qwen2.5-VL) in Triton with TensorRT-LLM

TensorRT-LLM is NVIDIA's recommended solution for running Large Language Models
(LLMs) and multimodal models on NVIDIA GPUs. Read more about TensorRT-LLM
[here](https://github.com/NVIDIA/TensorRT-LLM) and Triton's TensorRT-LLM Backend
[here](https://github.com/triton-inference-server/tensorrtllm_backend).

This tutorial shows how to serve the multimodal
[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
vision-language model with Triton Inference Server using the TensorRT-LLM
PyTorch backend (LLM API). The PyTorch backend works directly with Hugging Face
checkpoints — no TensorRT engine building required.

> [!NOTE]
> The legacy TensorRT engine-build workflow for multimodal models (building
> separate visual and LLM engines with `trtllm-build` and
> `build_visual_engine.py`, then wiring them through an `inflight_batcher_llm`
> model repository) is deprecated and is being removed from TensorRT-LLM. This
> tutorial uses the modern LLM API / PyTorch backend instead.

## Launch the Triton TensorRT-LLM container

Mount your Hugging Face cache so the model can be auto-downloaded at server
startup. Replace `<xx.yy>` with the version of Triton you want to use — the
latest Triton Server container is recommended and can be found
[here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

For gated models, set your token first: `export HF_TOKEN=hf_...`

## Prepare the model repository

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

Edit `TensorRT-LLM/triton_backend/all_models/llmapi/tensorrt_llm/1/model.yaml`
and set the model:

```yaml
model: Qwen/Qwen2.5-VL-7B-Instruct
backend: pytorch
```

All keys in `model.yaml` map directly to the
[`LLM()` constructor arguments](https://nvidia.github.io/TensorRT-LLM/llm-api/) —
this is where you configure KV cache, parallelism, and more. You can also point
`model` at a local filesystem path if you have pre-downloaded the checkpoint.

## Serving with Triton

Launch Triton Server with the
[launch_triton_server.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/scripts/launch_triton_server.py)
script, running from the parent of `TensorRT-LLM/`:

```bash
python3 TensorRT-LLM/triton_backend/scripts/launch_triton_server.py \
    --model_repo=TensorRT-LLM/triton_backend/all_models/llmapi/
```

> You should expect the following response once the server is ready:
> ```
> I0503 22:01:25.210518 1175 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
> I0503 22:01:25.211612 1175 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
> I0503 22:01:25.254914 1175 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002
> ```

To stop Triton Server inside the container, run:
```bash
pkill tritonserver
```

## Send an inference request

For a text-only prompt, use the
[generate endpoint](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html):

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
  -d '{"text_input": "Describe how vision-language models understand images.", "sampling_param_max_tokens": 100}' | jq
```

For image + text (multimodal) requests, Qwen2.5-VL follows the TensorRT-LLM
multimodal LLM API input format. See the
[multimodal LLM API examples](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/multimodal/README.md)
and the
[TensorRT-LLM Backend LLM API guide](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llmapi.md)
for the exact request schema for passing images alongside the prompt.

## References

- [TensorRT-LLM Backend README](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md)
- [TensorRT-LLM Backend LLM API guide](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llmapi.md)
- [End to end workflow to run multi-modal models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/multimodal/README.md)
