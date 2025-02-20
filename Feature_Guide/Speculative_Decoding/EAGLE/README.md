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

# EAGLE Speculative Decoding

This tutorial shows how to build and run a model using EAGLE speculative decoding ([paper](https://arxiv.org/pdf/2401.15077) | [github](https://github.com/SafeAILab/EAGLE/tree/main) | [blog](https://sites.google.com/view/eagle-llm)) in Triton Inference Server with TensorRT-LLM backend on a single node with one GPU.

TensorRT-LLM is NVIDIA's recommended solution of running Large Language Models(LLMs) on NVIDIA GPUs. Read more about TensoRT-LLM [here](https://github.com/NVIDIA/TensorRT-LLM) and Triton's TensorRT-LLM Backend [here](https://github.com/triton-inference-server/tensorrtllm_backend).

## Limitations
  * EAGLE-2 is not supported.
  * [mc_sim_7b_63](https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_choices.py#L1) would be used as EAGLE choices for each inference request and cannot be changed.
  * Pipeline parallelism is not supported.

## Acquiring EAGLE Model and its Base Model

Throughout the tutorial, we will be using the [EAGLE-Vicuna-7B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3) model, uploaded to HuggingFace by the authors of EAGLE. More types of EAGLE models could be found [here](https://sites.google.com/view/eagle-llm). The base model [Vicuna-7B-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3) is also needed for EAGLE to work.

Vicuna-7B-v1.3 model is a fine-tuned Llama. With some modifications, you can add EAGLE to other base models as well. Some TensorRT-LLM models might not work with EAGLE due to the missing head size in the speculative decoding XQA attention kernels.

To download both models, run the following command:
```bash
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
git clone https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3
```

## Acquiring TensorRT-LLM backend

This tutorial requires TensorRT-LLM Backend repository. Please note,
that for best user experience we recommend using the latest
[release tag](https://github.com/triton-inference-server/tensorrtllm_backend/tags)
of `tensorrtllm_backend`.

To clone TensorRT-LLM Backend repository, make sure to run the following set of commands:
```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch <release branch>
cd tensorrtllm_backend
git submodule update --init --recursive
```

## Launch Triton TensorRT-LLM container

Launch Triton docker container with TensorRT-LLM backend.
Note that we're mounting `tensorrtllm_backend` to `/tensorrtllm_backend`
and the downloaded EAGLE and base models to `/hf-models` in the docker container for simplicity.
Make an `engines` folder outside docker to reuse engines for future runs.
Please, make sure to replace <xx.yy> with the version of Triton that you want
to use. The latest Triton Server container could be found [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/tensorrtllm_backend>:/tensorrtllm_backend \
    -v </path/to/eagle/and/base/model/>:/hf-models \
    -v </path/to/engines>:/engines \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

## Create Engines for Each Model [skip this step if you already have an engine]

TensorRT-LLM requires each model to be compiled for the configuration
you need before running. To do so, before you run your model for the first time
on Triton Server you will need to create a TensorRT-LLM engine.

Starting with [24.04 release](https://github.com/triton-inference-server/server/releases/tag/v2.45.0),
Triton Server TensrRT-LLM container comes with
pre-installed TensorRT-LLM package, which allows users to build engines inside
the Triton container. Simply follow the next steps in the container:

```bash
BASE_MODEL=/hf-models/vicuna-7b-v1.3
EAGLE_MODEL=/hf-models/EAGLE-Vicuna-7B-v1.3
CKPT_PATH=/tmp/ckpt/vicuna/7b/
ENGINE_DIR=/engines/eagle-vicuna-7b/1-gpu/
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/eagle/convert_checkpoint.py
python3 ${CONVERT_CHKPT_SCRIPT} --model_dir ${BASE_MODEL} \
                                --eagle_model_dir ${EAGLE_MODEL} \
                                --output_dir ${CKPT_PATH} \
                                --dtype float16 \
                                --max_draft_len 63 \
                                --num_eagle_layers 4 \
                                --max_non_leaves_per_layer 10
trtllm-build --checkpoint_dir ${CKPT_PATH} \
            --output_dir ${ENGINE_DIR} \
            --gemm_plugin float16 \
            --use_paged_context_fmha enable \
            --speculative_decoding_mode eagle \
            --max_batch_size 4
```

To verify that the engine is built correctly, run the following command:
```bash
python3 /tensorrtllm_backend/tensorrt_llm/examples/run.py --engine_dir ${ENGINE_DIR} \
                 --tokenizer_dir ${BASE_MODEL} \
                 --max_output_len=100 \
                 --input_text "Once upon"
```
Sample output:
```
> Input [Text 0]: "<s> Once upon"
> Output [Text 0 Beam 0]: "a time, there was a young girl who loved to read. She would spend hours in the library, devouring books of all genres. She had a special love for fairy tales, and would often dream of living in a magical world where she could meet princes and princesses, and have adventures with talking animals.
> One day, while she was reading a book, she came across a passage that spoke to her heart. It said, "You are the author of"
> [TensorRT-LLM][INFO] Refreshed the MPI local session
```

## Serving with Triton

The last step is to create a Triton readable model. You can find a template of a model that uses inflight batching in
[tensorrtllm_backend/all_models/inflight_batcher_llm](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main/all_models/inflight_batcher_llm). To run EAGLE model, you will need to:

1. Copy over the inflight batcher models repository
```bash
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
```

2. Modify config.pbtxt for the preprocessing, postprocessing and processing steps.

```bash
TOKENIZER_DIR=/hf-models/vicuna-7b-v1.3
TOKENIZER_TYPE=auto
ENGINE_DIR=/engines/eagle-vicuna-7b/1-gpu/
DECOUPLED_MODE=false
MODEL_FOLDER=/opt/tritonserver/inflight_batcher_llm
MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=10000
TRITON_BACKEND=tensorrtllm
LOGITS_DATATYPE="TYPE_FP32"
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE}
```

3. Launch Tritonserver

Launch Tritonserver with the [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) script. Here, we launch a single instance of `tritonserver` with MPI by setting `--world_size=1`.

```bash
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=1 --model_repo=/opt/tritonserver/inflight_batcher_llm
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
Note: do not forget to run above command to stop Triton Server if launch Tritionserver failed due to various reasons. Otherwise, it could cause OOM or MPI issues.

curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is ML?", "max_tokens": 50, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'

## Send an Inference Request

You can test the results of the run with:
1. The [inflight_batcher_llm_client.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/client/inflight_batcher_llm_client.py) script.

```bash
# Using the SDK container as an example. <xx.yy> is the version of Triton Server you are using.
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/tensorrtllm_backend/inflight_batcher_llm/client>:/tensorrtllm_client \
    -v </path/to/eagle/and/base/model/>:/hf-models \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
# Install extra dependencies for the script
pip3 install transformers sentencepiece
python3 /tensorrtllm_client/inflight_batcher_llm_client.py --request-output-len 50 --tokenizer-dir /hf-models/vicuna-7b-v1.3 --text "What is ML?"
```
> You should expect the following response:
> ```
> ...
> Input: What is ML?
> Output beam 0:
> ML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation.
> ...
> ```

2. The [generate endpoint](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint).

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is ML?", "max_tokens": 50, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```
> You should expect the following response:
> ```
> {"model_name":"ensemble","model_version":"1","sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"What is ML?\nML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."}
> ```

## Evaluating Performance with Gen-AI Perf

Gen-AI Perf is a command line tool for measuring the throughput and latency of generative AI models as served through an inference server.
You can read more about Gen-AI Perf [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html). We will use Gen-AI Perf to evaluate the performance gain of EAGLE model over the base model.

*NOTE: below experiment is done on a single node with one GPU - RTX 5880 (48GB GPU memory). The number below is only for reference. The actual number may vary due to the different hardware and environment.*

1. Prepare Dataset

We will be using the HumanEval dataset for our evaluation, which is used in the original EAGLE paper. The HumanEval dataset has been converted to the format required by EAGLE and is available [here](https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/humaneval/question.jsonl). To make it compatible for Gen-AI Perf, we need to do another conversion. You may use other datasets besides HumanEval as well, as long as it could be converted to the
format required by Gen-AI Perf. Note that MT-bench could not be used since Gen-AI Perf does not support multiturn dataset as input yet. Follow the steps below to download and convert the dataset.
```bash
wget https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/data/humaneval/question.jsonl

python3 dataset-converter.py --input_file question.jsonl --output_file converted_humaneval.jsonl
```

2. Install GenAI-Perf (Ubuntu 24.04, Python 3.10+)

```bash
pip install genai-perf
```
NOTE: you must already have CUDA 12 installed

3. Run Gen-AI Perf

Run the following command in the SDK container:
```bash
genai-perf \
  profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file /path/to/converted/dataset/converted_humaneval.jsonl \
  --tokenizer /path/to/hf-models/vicuna-7b-v1.3/ \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --request-rate 2
```
NOTE: you may need to change the input-file name according to your converted dataset. Above is using converted_humaneval.jsonl as an example.
A sample output that looks like this:
```
                                     NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃    min ┃       max ┃       p99 ┃       p90 ┃       p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│              Request Latency (ms) │ 7,667.61 │ 940.38 │ 16,101.65 │ 15,439.36 │ 14,043.33 │ 10,662.31 │
│   Output Sequence Length (tokens) │   319.87 │ 133.00 │    485.00 │    472.08 │    441.60 │    404.00 │
│    Input Sequence Length (tokens) │   153.05 │  63.00 │    278.00 │    259.38 │    190.20 │    183.50 │
│ Output Token Throughput (per sec) │   360.53 │    N/A │       N/A │       N/A │       N/A │       N/A │
│      Request Throughput (per sec) │     1.13 │    N/A │       N/A │       N/A │       N/A │       N/A │
│             Request Count (count) │    39.00 │    N/A │       N/A │       N/A │       N/A │       N/A │
└───────────────────────────────────┴──────────┴────────┴───────────┴───────────┴───────────┴───────────┘
```

4. Run Gen-AI Perf on Base Model

To compare performance between EAGLE and base model, we need to run Gen-AI Perf Tool on the base model as well. To do so, we need to repeat the steps above for the base model with minor changes.

Kill the existing Triton Server and run the following command in the Triton Server container:
```bash
pkill tritonserver
```

Build the TRT-LLM engine for the base model:
```bash
BASE_MODEL=/hf-models/vicuna-7b-v1.3
CKPT_PATH=/tmp/ckpt/vicuna-base/7b/
ENGINE_DIR=/engines/vicuna-7b/1-gpu/
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
python3 ${CONVERT_CHKPT_SCRIPT} --model_dir ${BASE_MODEL} \
                                --output_dir ${CKPT_PATH} \
                                --dtype float16
trtllm-build --checkpoint_dir ${CKPT_PATH} \
            --output_dir ${ENGINE_DIR} \
            --remove_input_padding enable \
            --gpt_attention_plugin float16 \
            --context_fmha enable \
            --gemm_plugin float16 \
            --paged_kv_cache enable \
            --max_batch_size 4
```

Create a Triton readable model for the base model:
```bash
mkdir -p /opt/tritonserver/vicuna_base
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/vicuna_base/.

TOKENIZER_DIR=/hf-models/vicuna-7b-v1.3
TOKENIZER_TYPE=auto
ENGINE_DIR=/engines/vicuna-7b/1-gpu/
DECOUPLED_MODE=false
MODEL_FOLDER=/opt/tritonserver/vicuna_base/inflight_batcher_llm
MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=10000
TRITON_BACKEND=tensorrtllm
LOGITS_DATATYPE="TYPE_FP32"
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE}
```

Launch Triton Server with the base model:
```bash
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=1 --model_repo=/opt/tritonserver/vicuna_base/inflight_batcher_llm
```

Run Gen-AI Perf Tool on Base Model:
```bash
genai-perf \
  profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file /path/to/converted/dataset/converted_humaneval.jsonl \
  --tokenizer /path/to/hf-models/vicuna-7b-v1.3/ \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --request-rate 2
```

Sample performance output for base model:
```
                                      NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃      min ┃       max ┃       p99 ┃       p90 ┃       p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│              Request Latency (ms) │ 8,730.66 │ 1,792.94 │ 16,376.18 │ 16,054.17 │ 14,780.51 │ 12,529.04 │
│   Output Sequence Length (tokens) │   353.32 │   153.00 │    534.00 │    508.65 │    445.30 │    428.25 │
│    Input Sequence Length (tokens) │   156.62 │    63.00 │    296.00 │    288.98 │    196.60 │    185.00 │
│ Output Token Throughput (per sec) │   410.03 │      N/A │       N/A │       N/A │       N/A │       N/A │
│      Request Throughput (per sec) │     1.16 │      N/A │       N/A │       N/A │       N/A │       N/A │
│             Request Count (count) │    40.00 │      N/A │       N/A │       N/A │       N/A │       N/A │
└───────────────────────────────────┴──────────┴──────────┴───────────┴───────────┴───────────┴───────────┘
```

5. Compare Performance

```bash
genai-perf \
  profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file /path/to/converted/dataset/converted_gsm8k.jsonl \
  --tokenizer /path/to/hf-models/vicuna-7b-v1.3/ \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --request-rate 5
```

EAGLE model performance output on GSM8K:
```
                                   NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃   min ┃       max ┃       p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│              Request Latency (ms) │ 5,633.32 │ 34.94 │ 13,317.99 │ 12,415.99 │ 9,931.24 │ 8,085.09 │
│   Output Sequence Length (tokens) │   116.02 │ 23.00 │    353.00 │    348.77 │   305.30 │   126.00 │
│    Input Sequence Length (tokens) │    66.70 │ 23.00 │    148.00 │    144.39 │   102.10 │    81.00 │
│ Output Token Throughput (per sec) │   389.08 │   N/A │       N/A │       N/A │      N/A │      N/A │
│      Request Throughput (per sec) │     3.35 │   N/A │       N/A │       N/A │      N/A │      N/A │
│             Request Count (count) │   120.00 │   N/A │       N/A │       N/A │      N/A │      N/A │
└───────────────────────────────────┴──────────┴───────┴───────────┴───────────┴──────────┴──────────┘
```

Base model performance output on GSM8K:
```
                                  NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃   min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│              Request Latency (ms) │ 4,327.16 │ 32.04 │ 9,253.56 │ 9,033.99 │ 7,175.71 │ 6,257.44 │
│   Output Sequence Length (tokens) │   116.09 │ 23.00 │   353.00 │   330.00 │   289.00 │   127.00 │
│    Input Sequence Length (tokens) │    65.24 │ 23.00 │   148.00 │   139.83 │    98.40 │    79.00 │
│ Output Token Throughput (per sec) │   472.50 │   N/A │      N/A │      N/A │      N/A │      N/A │
│      Request Throughput (per sec) │     4.07 │   N/A │      N/A │      N/A │      N/A │      N/A │
│             Request Count (count) │   144.00 │   N/A │      N/A │      N/A │      N/A │      N/A │
└───────────────────────────────────┴──────────┴───────┴──────────┴──────────┴──────────┴──────────┘
```

