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

# Speculative Decoding with vLLM

- [About Speculative Decoding](#about-speculative-decoding)
- [EAGLE](#eagle)
- [Draft Model-Based Speculative Decoding](#draft-model-based-speculative-decoding)

## About Speculative Decoding

This tutorial shows how to build and serve speculative decoding models in Triton Inference Server with [vLLM Backend](https://github.com/triton-inference-server/vllm_backend) on a single node with one GPU. Please go to [Speculative Decoding](../README.md) main page to learn more about other supported backends.

According to [Spec-Bench](https://sites.google.com/view/spec-bench), EAGLE is currently the top-performing approach for speeding up LLM inference across different tasks. In this tutorial, we'll focus on [EAGLE](#eagle) and demonstrate how to make it work with Triton Inference Server. We'll also cover [Draft Model-Based Speculative Decoding](#draft-model-based-speculative-decoding) for those interested in exploring alternative methods. If you are interested in how vLLM supports speculative decoding, more details [here](https://blog.vllm.ai/2024/10/17/spec-decode.html). By finishing this tutorial, you will be able to try other speculative decoding techniques provided by vLLM [here](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculative-decoding) with Triton Inference Server easily on your own.

## EAGLE

EAGLE ([paper](https://arxiv.org/pdf/2401.15077) | [github](https://github.com/SafeAILab/EAGLE) | [blog](https://sites.google.com/view/eagle-llm)) is a speculative decoding technique that accelerates Large Language Model (LLM) inference by predicting future tokens based on contextual features extracted from the LLM's second-top layer. It employs a lightweight Auto-regression Head to predict the next feature vector, which is then used to generate tokens through the LLM's frozen classification head, achieving significant speedups (2x-3x faster than vanilla decoding) while maintaining output quality and distribution consistency.

### Acquiring EAGLE Model and its Base Model

In this example, we will be using the [EAGLE-LLaMA3-Instruct-8B](https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B) model.
More types of EAGLE models can be found [here](https://huggingface.co/yuhuili). The base model [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) is also needed for EAGLE to work.

To download both models, run the following command:
```bash
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git clone https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```
*NOTE: you need to request access in Hugging Face and login to download and use Llama3 models.*

### Convert EAGLE Model

According to vLLM official [doc](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-using-eagle-based-draft-models):
> ... EAGLE models should be able to be loaded and used directly by vLLM after [PR 12304](https://github.com/vllm-project/vllm/pull/12304). If you are using vllm version before [PR 12304](https://github.com/vllm-project/vllm/pull/12304), please use the [script](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d) to convert the speculative model, and specify speculative_model="path/to/modified/eagle/model" ...

For Triton, if you are using Triton Server container version <= 25.02, you need to convert the EAGLE model by running above [script](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d), inside the folder than contains both EAGLE and base models. Triton Server container version >= 25.03 would use vLLM versions (>= 0.7.3) that contains PR 12304.

### Create Model Repository

A model repository is Triton’s way of reading your models and any associated metadata with each model (configurations, version files, etc.). See [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_1-model_deployment/README.html#setting-up-the-model-repository) for model details.

We have prepared a template of a model repository for EAGLE model and base model in [model_repository](model_repository). Please make a copy and modify the model.json to suit your needs. For example, we are setting `num_speculative_tokens` to 5 for eagle_model, according to the vLLM [example](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-with-a-draft-model). You can change it to other values and it might affect the performance.

### Serving with Triton

Let's serve the model by launching Triton docker container with vLLM backend.
Note that we're mounting the downloaded (and maybe converted) EAGLE and base models to `/hf-models` and the model repository acquired in the previous section to `/model_repository` in the docker container. Please, make sure to replace <xx.yy> with the version of Triton that you want
to use. The latest Triton Server container is recommended and could be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

```bash
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v </path/to/model_repository>:/model_repository \
    -v </path/to/eagle/and/base/model>:/hf-models \
    nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 \
    tritonserver --model-repository /model_repository \
    --model-control-mode explicit --load-model eagle_model
```

### Send Inference Requests

Let's send an inference request to the [generate endpoint](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html).

```bash
curl -X POST localhost:8000/v2/models/eagle_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}' | jq
```

> You should expect the following response:
> ```
> {
>  "model_name": "eagle_model",
>  "model_version": "1",
>  "text_output": "What is Triton Inference Server?¶\n\nTriton Inference Server is an open-source, high-performance,"
> }
> ```

### Evaluating Performance with Gen-AI Perf

Gen-AI Perf is a command line tool for measuring the throughput and latency of generative AI models as served through an inference server.
You can read more about Gen-AI Perf [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html). We will use Gen-AI Perf to evaluate the performance gain of EAGLE model over the base model.

1. Prepare Dataset

We will be using the HumanEval dataset for our evaluation, which is used in the original EAGLE paper. The HumanEval dataset has been converted to the format required by EAGLE and is available [here](https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/humaneval/question.jsonl). To make it compatible for Gen-AI Perf, we need to do another conversion. You may use other datasets besides HumanEval as well, as long as it could be converted to the
format required by Gen-AI Perf. Note that MT-bench could not be used since Gen-AI Perf does not support multiturn dataset as input yet. Follow the steps below to download and convert the dataset.
```bash
wget https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/data/humaneval/question.jsonl

# dataset-converter.py file can be found in the parent folder as this README.
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
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file /path/to/converted/dataset/converted_humaneval.jsonl \
  --tokenizer /path/to/hf-models/Meta-Llama-3-8B-Instruct/ \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --concurrency 1
```
*NOTE: When benchmarking the speedup of speculative decoding versus the base model, use `--concurrency 1`. This setting is crucial because speculative decoding is designed to trade extra computation for reduced token generation latency. By limiting concurrency, we avoid saturating hardware resources with multiple requests, allowing for a more accurate assessment of the technique's latency benefits. This approach ensures that the benchmark reflects the true performance gains of speculative decoding in real-world, low-concurrency scenarios.*

A sample output that looks like this:
```
                                    NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│              Request Latency (ms) │ 7,510.69 │ 6,534.94 │ 8,433.33 │ 8,409.31 │ 8,193.07 │ 7,832.68 │
│   Output Sequence Length (tokens) │   325.00 │   324.00 │   326.00 │   325.97 │   325.70 │   325.25 │
│    Input Sequence Length (tokens) │   112.50 │    79.00 │   137.00 │   136.55 │   132.50 │   125.75 │
│ Output Token Throughput (per sec) │    43.27 │      N/A │      N/A │      N/A │      N/A │      N/A │
│      Request Throughput (per sec) │     0.13 │      N/A │      N/A │      N/A │      N/A │      N/A │
│             Request Count (count) │     4.00 │      N/A │      N/A │      N/A │      N/A │      N/A │
└───────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

*NOTE: above sample output is done on a single node with one GPU - RTX 5880 (48GB GPU memory). The number below is only for reference. The actual number may vary due to the different hardware and environment.*

4. Run Gen-AI Perf on Base Model

To compare performance between EAGLE and base model (i.e. vanilla LLM w/o speculative decoding), we need to run Gen-AI Perf Tool on the base model as well. To serve base model, we only need to change the [Serving with Triton](#Serving-with-Triton) by switching the `--load-model` argument from `eagle_model` to `base_model`:

```bash
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v </path/to/model_repository>:/model_repository \
    -v </path/to/eagle/and/base/model>:/hf-models \
    nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 \
    tritonserver --model-repository /model_repository \
    --model-control-mode explicit --load-model base_model
```

Please use EAGLE with care, since according to vLLM [doc](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-using-eagle-based-draft-models):

> When using EAGLE-based speculators with vLLM, the observed speedup is lower than what is reported in the reference implementation [here](https://github.com/SafeAILab/EAGLE). This issue is under investigation and tracked here: [vllm-project/vllm#9565](https://github.com/vllm-project/vllm/issues/9565).

## Draft Model-Based Speculative Decoding

Draft Model-Based Speculative Decoding ([paper](https://arxiv.org/pdf/2302.01318)) is another (and earlier) approach to accelerate LLM inference, distinct from EAGLE. Here are the key differences:

 - Draft Generation: it uses a smaller, faster LLM as a draft model to predict multiple tokens ahead. This contrasts with EAGLE's feature-level extrapolation.

 - Verification Process: it employs a chain-like structure for draft generation and verification, unlike EAGLE which uses tree-based attention mechanisms.

 - Efficiency: While effective, it is generally slower than EAGLE.

 - Implementation: it requires a separate draft model, which can be challenging to implement effectively for smaller target models. EAGLE, in contrast, modifies the existing model architecture.

 - Accuracy: its draft accuracy can vary depending on the draft model used, while EAGLE achieves a higher draft accuracy (about 0.8).

To run Draft Model-Based Speculative Decoding with Triton Inference Server, it is very similar to the steps above for EAGLE. The only difference is that you need to use a different model repository. A template of model repository for Draft Model-Based Speculative Decoding is available in [model_repository/opt_model](model_repository/opt_model), following the example from vLLM [doc](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-with-a-draft-model). Please make a copy and modify the model.json to suit your needs. Then, you can start Triton server with the following command:

```bash
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v </path/to/model_repository>:/model_repository \
    nvcr.io/nvidia/tritonserver:25.02-vllm-python-py3 \
    tritonserver --model-repository /model_repository \
    --model-control-mode explicit --load-model opt_model
```