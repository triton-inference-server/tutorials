<!--
# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
- [EAGLE 3](#eagle-3)
- [MEDUSA](#medusa)
- [Draft Model-Based Speculative Decoding](#draft-model-based-speculative-decoding)

## About Speculative Decoding

This tutorial shows how to build and serve speculative decoding models in Triton Inference Server with [TensorRT-LLM LLM API / PyTorch backend](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llmapi.md) on a single node with one GPU. Please go to [Speculative Decoding](../README.md) main page to learn more about other supported backends.

> **Note:** This tutorial uses the modern **LLM API / PyTorch backend**, which works directly with HuggingFace model checkpoints and does not require building TensorRT engines. If you are looking for the legacy TRT engine-based approach, see the [engine backend archive](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/README-engine-backend-archive.md).

According to [Spec-Bench](https://sites.google.com/view/spec-bench), EAGLE is currently the top-performing approach for speeding up LLM inference across different tasks.
In this tutorial, we'll focus on [EAGLE 3](#eagle-3) and demonstrate how to make it work with Triton Inference Server. However, we'll also cover [Draft Model-Based Speculative Decoding](#draft-model-based-speculative-decoding) for those interested in exploring alternative methods. This way, you can choose the best fit for your needs.

## EAGLE 3

EAGLE-3 ([paper](https://arxiv.org/pdf/2503.01840) | [github](https://github.com/SafeAILab/EAGLE) | [blog](https://sites.google.com/view/eagle-llm)) is the latest generation of the EAGLE speculative decoding technique that accelerates Large Language Model (LLM) inference by predicting future tokens based on contextual features. It employs a lightweight draft head to predict the next feature vector, which is then used to generate tokens through the LLM's frozen classification head, achieving significant speedups (2x-3x faster than vanilla decoding) while maintaining output quality. Compared to EAGLE (v1/v2), EAGLE 3 further improves acceptance rates through training-time test enhancements.

### Download the Target and Draft Models (Optional)

In this example, we use [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) as the target model and [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) as the draft model. Both models can be auto-downloaded from HuggingFace at server startup by mounting your HuggingFace cache into the container. Alternatively, you can pre-download them:

```bash
# Authenticate first if needed (Llama-3.1 requires accepting the license on HuggingFace)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

More EAGLE 3 compatible draft model checkpoints can be found in the [Speculative Decoding Modules](https://huggingface.co/collections/nvidia/speculative-decoding-modules) collection from NVIDIA.

### Launch Triton TensorRT-LLM Container

Launch the Triton container with TensorRT-LLM backend. Mount your HuggingFace cache so models can be auto-downloaded. Replace `<xx.yy>` with the version of Triton you want to use (must be >= 25.01). The latest Triton Server container is recommended and can be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

### Prepare the Model Repository

Copy the LLM API model template inside the container:

```bash
cp -R /app/all_models/llmapi/ /opt/tritonserver/llmapi_repo/
```

Edit `/opt/tritonserver/llmapi_repo/tensorrt_llm/1/model.yaml` to configure EAGLE 3:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

speculative_config:
  decoding_type: Eagle3
  max_draft_len: 3
  speculative_model: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

triton_config:
  max_batch_size: 0
  decoupled: False
```

*NOTE: You can also specify a local filesystem path for `model` and `speculative_model` if you have pre-downloaded the models.*

### Serving with Triton

Launch Triton Server with the [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/scripts/launch_triton_server.py) script:

```bash
python3 /app/scripts/launch_triton_server.py --model_repo=/opt/tritonserver/llmapi_repo/
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
*NOTE: do not forget to run the above command to stop Triton Server if launching Triton Server failed due to various reasons. Otherwise, it could cause OOM or MPI issues.*

### Send Inference Requests

You can test the results of the run with the [generate endpoint](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html):

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
  -d '{"text_input": "What is ML?", "sampling_param_max_tokens": 50}'
```

> You should expect the following response:
> ```json
> {"model_name":"tensorrt_llm","model_version":"1","text_output":"What is ML?\nML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."}
> ```

Optionally, include speculative decoding performance metrics by adding `"sampling_param_return_perf_metrics": true`:

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
  -d '{"text_input": "What is ML?", "sampling_param_max_tokens": 50, "sampling_param_return_perf_metrics": true}' | jq
```

This adds fields like `acceptance_rate`, `total_accepted_draft_tokens`, and `total_draft_tokens` to the response, which are useful for evaluating speculative decoding effectiveness.

### Evaluating Performance with Gen-AI Perf

Gen-AI Perf is a command line tool for measuring the throughput and latency of generative AI models as served through an inference server.
You can read more about Gen-AI Perf [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html). We will use Gen-AI Perf to evaluate the performance gain of EAGLE 3 over the base model.

*NOTE: below experiment is done on a single node with one GPU - RTX 5880 (48GB GPU memory). The number below is only for reference. The actual number may vary due to the different hardware and environment.*

1. Prepare Dataset

We will be using the HumanEval dataset for our evaluation, which is used in the original EAGLE paper. The HumanEval dataset has been converted to the format required by EAGLE and is available [here](https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/humaneval/question.jsonl). To make it compatible for Gen-AI Perf, we need to do another conversion. You may use other datasets besides HumanEval as well, as long as it could be converted to the format required by Gen-AI Perf. Note that MT-bench could not be used since Gen-AI Perf does not support multiturn dataset as input yet. Follow the steps below to download and convert the dataset.

```bash
wget https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/data/humaneval/question.jsonl

# dataset-converter.py file can be found in the parent folder of this README.
python3 dataset-converter.py --input_file question.jsonl --output_file converted_humaneval.jsonl
```

2. Install GenAI-Perf (Ubuntu 24.04, Python 3.10+)

```bash
pip install genai-perf
```
*NOTE: you must already have CUDA 12 installed.*

3. Run Gen-AI Perf

Run the following command in the SDK container:
```bash
genai-perf \
  profile \
  -m tensorrt_llm \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file /path/to/converted/dataset/converted_humaneval.jsonl \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --concurrency 1
```
*NOTE: When benchmarking the speedup of speculative decoding versus the base model, use `--concurrency 1`. This setting is crucial because speculative decoding is designed to trade extra computation for reduced token generation latency. By limiting concurrency, we avoid saturating hardware resources with multiple requests, allowing for a more accurate assessment of the technique's latency benefits. This approach ensures that the benchmark reflects the true performance gains of speculative decoding in real-world, low-concurrency scenarios.*

4. Run Gen-AI Perf on Base Model

To compare performance between EAGLE 3 and the base model (i.e. vanilla LLM without speculative decoding), restart Triton Server with a `model.yaml` that omits the `speculative_config` block:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

triton_config:
  max_batch_size: 0
  decoupled: False
```

Then re-run the Gen-AI Perf command above.

5. Compare Performance

From sample runs, EAGLE 3 typically delivers 2x or greater token throughput improvement over the base model at low concurrency. The exact speedup varies by hardware, model, and dataset.

As stated above, the number above is gathered from a single node with one GPU - RTX 5880 (48GB GPU memory). The actual number may vary due to the different hardware and environment.

## MEDUSA

> **Important:** MEDUSA is **not supported** in the modern LLM API / PyTorch backend. It only works with the legacy TRT engine backend.
>
> For new deployments, we recommend using [EAGLE 3](#eagle-3) instead, which is fully supported on the LLM API / PyTorch backend and achieves higher draft accuracy.
>
> If you specifically need MEDUSA with the TRT engine backend, refer to the [engine backend archive](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/README-engine-backend-archive.md) for the legacy instructions.

## Draft Model-Based Speculative Decoding

Draft Model-Based Speculative Decoding ([paper](https://arxiv.org/pdf/2302.01318)) is another approach to accelerate LLM inference that uses a smaller, faster LLM as a draft model to predict multiple tokens ahead. This approach is distinct from EAGLE 3 and is supported in the modern LLM API / PyTorch backend. Here are the key differences compared to EAGLE 3:

 - Draft Generation: it uses a separate, independent LLM as a draft model to predict multiple tokens ahead. This contrasts with EAGLE 3's feature-level extrapolation using a lightweight draft head embedded into the target model.

 - Verification Process: it employs a chain-like (linear) structure for draft generation and verification, unlike EAGLE 3 which uses tree-based attention mechanisms.

 - Consistency: it maintains distribution consistency with the target LLM in both greedy and non-greedy settings, similar to EAGLE 3.

 - Efficiency: While effective, it is generally slower than EAGLE 3.

 - Implementation: it requires a separate draft model that shares the same tokenizer as the target model. The draft model can be any HuggingFace-compatible LLM.

To use Draft Model-Based Speculative Decoding with Triton via the LLM API, follow the same container setup and model repository preparation steps as in the [EAGLE 3](#eagle-3) section above, but configure `model.yaml` as follows:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

speculative_config:
  decoding_type: Draft_Target
  max_draft_len: 3
  speculative_model: /path/to/draft_model  # Must share the same tokenizer as the target model

triton_config:
  max_batch_size: 0
  decoupled: False
```

*NOTE: The draft and target models must be trained with the same tokenizer. If they are not compatible, the acceptance rate will be extremely low and performance will regress rather than improve.*
