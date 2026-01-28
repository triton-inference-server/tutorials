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

# Speculative Decoding with TensorRT-LLM

- [About Speculative Decoding](#about-speculative-decoding)
- [EAGLE](#eagle)
- [MEDUSA](#medusa)
- [Draft Model-Based Speculative Decoding](#draft-model-based-speculative-decoding)

## About Speculative Decoding

This tutorial shows how to serve speculative decoding models in Triton Inference Server with [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend) using the PyTorch backend and LLMAPI. The LLMAPI backend provides a simplified deployment approach - **no engine building required**. Please go to [Speculative Decoding](../README.md) main page to learn more about other supported backends.

According to [Spec-Bench](https://sites.google.com/view/spec-bench), EAGLE is currently the top-performing approach for speeding up LLM inference across different tasks.
In this tutorial, we'll focus on [EAGLE](#eagle) and demonstrate how to make it work with Triton Inference Server. However, we'll also cover [MEDUSA](#medusa) and [Draft Model-Based Speculative Decoding](#draft-model-based-speculative-decoding) for those interested in exploring alternative methods. This way, you can choose the best fit for your needs.

## EAGLE

EAGLE ([paper](https://arxiv.org/pdf/2401.15077) | [github](https://github.com/SafeAILab/EAGLE) | [blog](https://sites.google.com/view/eagle-llm)) is a speculative decoding technique that accelerates Large Language Model (LLM) inference by predicting future tokens based on contextual features extracted from the LLM's second-top layer. It employs a lightweight Auto-regression Head to predict the next feature vector, which is then used to generate tokens through the LLM's frozen classification head, achieving significant speedups (2x-3x faster than vanilla decoding) while maintaining output quality and distribution consistency. EAGLE-2, an improved version, further enhances performance by using confidence scores from the draft model to dynamically adjust the draft tree structure, resulting in even faster inference speeds.

*NOTE: EAGLE-2 is not supported via Triton Inference Server using TensorRT-LLM backend yet.*

### EAGLE Model Information

In this example, we will be using [EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) with the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) base model. More EAGLE3 models can be found [here](https://huggingface.co/yuhuili). With the LLMAPI backend, models are downloaded automatically from HuggingFace when first used.

### Launch Triton TensorRT-LLM container

Launch Triton docker container with TensorRT-LLM backend.
Please, make sure to replace <xx.yy> with the version of Triton that you want
to use (must be >= 25.01). The latest Triton Server container is recommended and can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

### Prepare Model Repository

Copy the LLMAPI model template and configure it for EAGLE speculative decoding:

```bash
cp -R /app/all_models/llmapi/ llmapi_repo/
```

Edit `llmapi_repo/tensorrt_llm/1/model.yaml` with your EAGLE configuration:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

speculative_config:
  decoding_type: Eagle
  speculative_model: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
  max_draft_len: 4

triton_config:
  max_batch_size: 0
  decoupled: False
```

*NOTE: On the PyTorch backend, `decoding_type: Eagle` is treated as `Eagle3`. EAGLE (v1/v2) draft checkpoints are not compatible - you must use an Eagle3 draft model. See the [TensorRT-LLM speculative decoding documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md) for available Eagle3 models.*

### Serving with Triton

Launch Tritonserver with the [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/scripts/launch_triton_server.py) script:

```bash
python3 /app/scripts/launch_triton_server.py --model_repo=llmapi_repo/
```

> You should expect the following response:
> ```
> ...
> I0503 22:01:25.210518 1175 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
> I0503 22:01:25.211612 1175 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
> I0503 22:01:25.254914 1175 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002
> ```

To stop Triton Server inside the container, run:
```bash
pkill tritonserver
```
*NOTE: do not forget to run above command to stop Triton Server if launch Tritionserver failed due to various reasons. Otherwise, it could cause OOM or MPI issues.*

### Send Inference Requests

You can test the results of the run with the [generate endpoint](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html):

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
    -d '{"text_input": "What is ML?", "sampling_param_max_tokens": 50}' | jq
```
> You should expect the following response:
> ```json
> {
>   "model_name": "tensorrt_llm",
>   "model_version": "1",
>   "text_output": "ML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."
> }
> ```

### Evaluating Performance

You can benchmark the performance gain of EAGLE over the base model using [AIPerf](https://github.com/ai-dynamo/aiperf), NVIDIA's comprehensive benchmarking tool for generative AI models.

*NOTE: The experiments below are done on a single node with one GPU - RTX 5880 (48GB GPU memory). The numbers below are for reference only. Actual performance may vary due to different hardware and environment.*

1. Install AIPerf

Install AIPerf in the container (or run from a separate client machine):
```bash
pip install aiperf
```

2. Run Benchmark on EAGLE Model

Run AIPerf against the EAGLE model. AIPerf will generate synthetic prompts automatically:
```bash
aiperf profile \
    --model tensorrt_llm \
    --url http://localhost:8000/v2/models/tensorrt_llm/generate \
    --endpoint-type template \
    --extra-inputs payload_template:'{"text_input": {{ text|tojson }}, "sampling_param_max_tokens": {{ max_tokens }}}' \
    --extra-inputs response_field:'text_output' \
    --synthetic-input-tokens-mean 128 \
    --output-tokens-mean 256 \
    --request-count 50 \
    --concurrency 1
```

*NOTE: When benchmarking the speedup of speculative decoding versus the base model, use `--concurrency 1`. This setting is crucial because speculative decoding is designed to trade extra computation for reduced token generation latency. By limiting concurrency, we avoid saturating hardware resources with multiple requests, allowing for a more accurate assessment of the technique's latency benefits.*

AIPerf will output comprehensive metrics including:
- **Output Token Throughput (tokens/sec)**: Key metric for comparing EAGLE vs base model
- **Time to First Token (TTFT)**: Latency to receive the first token
- **Inter Token Latency (ITL)**: Average time between tokens
- **Request Latency**: End-to-end request latency

3. Run Benchmark on Base Model

To compare performance between EAGLE and the base model (i.e., vanilla LLM without speculative decoding), repeat the steps for the base model.

Kill the existing Triton Server:
```bash
pkill tritonserver
```

Create a model repository for the base model (without speculative decoding):
```bash
cp -R /app/all_models/llmapi/ llmapi_base_repo/
```

Edit `llmapi_base_repo/tensorrt_llm/1/model.yaml` for the base model (no speculative_config):

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

triton_config:
  max_batch_size: 0
  decoupled: False
```

Launch Triton Server with the base model:
```bash
python3 /app/scripts/launch_triton_server.py --model_repo=llmapi_base_repo/
```

Run AIPerf on the base model:
```bash
aiperf profile \
    --model tensorrt_llm \
    --url http://localhost:8000/v2/models/tensorrt_llm/generate \
    --endpoint-type template \
    --extra-inputs payload_template:'{"text_input": {{ text|tojson }}, "sampling_param_max_tokens": {{ max_tokens }}}' \
    --extra-inputs response_field:'text_output' \
    --synthetic-input-tokens-mean 128 \
    --output-tokens-mean 256 \
    --request-count 50 \
    --concurrency 1
```

4. Compare Performance

From the sample runs above, we can see that the EAGLE model has a lower latency and higher throughput than the base model. In our tests, EAGLE achieved approximately 2.2x speedup in output token throughput compared to the base model.

For more advanced benchmarking options, refer to the [AIPerf documentation](https://github.com/ai-dynamo/aiperf), including:
- Request rate testing with `--request-rate`
- Goodput measurement with `--goodput`
- GPU telemetry with `--enable-gpu-telemetry`

As stated above, the numbers are gathered from a single node with one GPU - RTX 5880 (48GB GPU memory). The actual number may vary due to different hardware and environment.


## MEDUSA

MEDUSA ([paper](https://arxiv.org/pdf/2401.10774)) is a speculative decoding technique that adds extra decoding heads to LLMs to predict multiple subsequent tokens in parallel. Here are the key differences between MEDUSA and EAGLE:

 - Architecture: MEDUSA adds extra decoding heads to LLMs to predict multiple subsequent tokens in parallel, while EAGLE extrapolates second-top-layer contextual feature vectors of LLMs.

 - Generation structure: MEDUSA generates a fully connected tree across adjacent layers through the Cartesian product, often resulting in nonsensical combinations. In contrast, EAGLE creates a sparser, more selective tree structure that is more context-aware.

 - Consistency: MEDUSA's non-greedy generation does not guarantee lossless performance, while EAGLE provably maintains consistency with vanilla decoding in the distribution of generated texts.

 - Accuracy: MEDUSA achieves an accuracy of about 0.6 in generating drafts, whereas EAGLE attains a higher accuracy of approximately 0.8 as claimed in the EAGLE paper.

 - Speed: EAGLE is reported to be 1.6x faster than MEDUSA for certain models as claimed in the EAGLE paper.

**NOTE:** MEDUSA is **not supported** on the PyTorch/LLMAPI backend. To use MEDUSA with Triton Inference Server, you must use the TensorRT engine-based workflow with `trtllm-build`. Please refer to the [TensorRT-LLM MEDUSA documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/speculative-decoding.md#medusa) for detailed instructions on building and deploying MEDUSA models.


## Draft Model-Based Speculative Decoding

Draft Model-Based Speculative Decoding ([paper](https://arxiv.org/pdf/2302.01318)) is another (and earlier) approach to accelerate LLM inference, distinct from both EAGLE and MEDUSA. Here are the key differences:

 - Draft Generation: it uses a smaller, faster LLM as a draft model to predict multiple tokens ahead. This contrasts with EAGLE's feature-level extrapolation and MEDUSA's additional decoding heads.

 - Verification Process: it employs a chain-like structure for draft generation and verification, unlike EAGLE and MEDUSA which use tree-based attention mechanisms.

 - Consistency: it maintains distribution consistency with the target LLM in both greedy and non-greedy settings, similar to EAGLE but different from MEDUSA.

 - Efficiency: While effective, it is generally slower than both EAGLE and MEDUSA.

 - Implementation: it requires a separate draft model, which can be challenging to implement effectively for smaller target models. EAGLE and MEDUSA, in contrast, modify the existing model architecture.

 - Accuracy: its draft accuracy can vary depending on the draft model used, while EAGLE achieves a higher draft accuracy (about 0.8) compared to MEDUSA (about 0.6).

### Draft Model Configuration

Edit `llmapi_repo/tensorrt_llm/1/model.yaml` with your draft model configuration:

```yaml
model: meta-llama/Llama-3.1-70B-Instruct
backend: pytorch

tensor_parallel_size: 4
pipeline_parallel_size: 1

speculative_config:
  decoding_type: DraftTarget
  speculative_model: meta-llama/Llama-3.1-8B-Instruct
  max_draft_len: 5

triton_config:
  max_batch_size: 0
  decoupled: False
```

For more details on draft model speculative decoding, please refer to the [TensorRT-LLM speculative decoding documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md).
