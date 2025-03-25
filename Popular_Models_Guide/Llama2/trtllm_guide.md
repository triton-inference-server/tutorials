<!--
# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Deploying Hugging Face Llama2-7b Model in Triton

TensorRT-LLM is Nvidia's recommended solution of running Large Language
Models(LLMs) on Nvidia GPUs. Read more about TensoRT-LLM [here](https://github.com/NVIDIA/TensorRT-LLM)
and Triton's TensorRT-LLM Backend [here](https://github.com/triton-inference-server/tensorrtllm_backend).

*NOTE:* If some parts of this tutorial doesn't work, it is possible that there
are some version mismatches between the `tutorials` and `tensorrtllm_backend`
repository. Refer to [llama.md](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
for more detailed modifications if necessary. And if you are familiar with
python, you can also try using
[High-level API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/high-level-api/README.md)
for LLM workflow.


## Acquiring Llama2-7B model

For this tutorial, we are using the Llama2-7B HuggingFace model with pre-trained
weights. Clone the repo of the model with weights and tokens
[here](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main).
You will need to get permissions for the Llama2 repository as well as get access
to the huggingface cli. To get access to the huggingface cli,
go here: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Deploying with Triton CLI

[Triton CLI](https://github.com/triton-inference-server/triton_cli) is
an open source command line interface that enables users to create,
deploy, and profile models served by the Triton Inference Server.

### Launch Triton TensorRT-LLM container

Launch Triton docker container with TensorRT-LLM backend.
Note that we're mounting the acquired Llama2-7b model to `/root/.cache/huggingface`
in the docker container so that Triton CLI could use it and skip the download
step.

Make an `engines` folder outside docker to reuse engines for future runs.
Please, make sure to replace <xx.yy> with the version of Triton that you want
to use.

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/Llama2/repo>:/root/.cache/huggingface \
    -v </path/to/engines>:/engines \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```
### Install Triton CLI

Install [the latest release](https://github.com/triton-inference-server/triton_cli/releases)
of Triton CLI:
```bash
GIT_REF=<LATEST_RELEASE>
pip install git+https://github.com/triton-inference-server/triton_cli.git@${GIT_REF}
```

### Prepare Triton model repository
Triton CLI has a single command `triton import` that automatically converts HF
checkpoint into TensorRT-LLM checkpoint format, builds TensorRT-LLM engines,
and prepares a Triton model repository:
```bash
ENGINE_DEST_PATH=/engines triton import -m llama-2-7b --backend tensorrtllm
```

Please, note that specifying `ENGINE_DEST_PATH` is optional, but recommended
if you want to re-use compiled engines in the future.

After successful run of `triton import`, you should see the structure of
a model repository printed in the console:
```
...
triton - INFO - Current repo at /root/models:
models/
├── llama-2-7b/
│   ├── 1/
│   │   ├── lib/
│   │   │   ├── decode.py
│   │   │   └── triton_decoder.py
│   │   └── model.py
│   └── config.pbtxt
├── postprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
└── tensorrt_llm/
    ├── 1/
    └── config.pbtxt

```

### Start Triton Inference Server

Start server pointing at the default model repository:
```
triton start
```

### Send an inference request
Use the [generate endpoint](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html).
to send an inference request to the deployed model.

```bash
curl -X POST localhost:8000/v2/models/llama-2-7b/generate -d '{"text_input": "What is ML?", "max_tokens": 50, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```
> You should expect the following response:
> ```
> {"context_logits":0.0,...,"text_output":"What is ML?\nML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."}
> ```

## Deploying with Triton Inference Server

If you would like to hava a better control over the deployment process,
next steps will guide you over the process of TensorRT-LLM engine building
process and Triton model repository set up.

### Prerequisite: TensorRT-LLM backend

This tutorial requires TensorRT-LLM Backend repository. Please note,
that for best user experience we recommend using the latest
[release tag](https://github.com/triton-inference-server/tensorrtllm_backend/tags)
of `tensorrtllm_backend` and
the latest [Triton Server container.](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)

To clone TensorRT-LLM Backend repository, make sure to run the following
set of commands.
```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch <release branch>
# Update the submodules
cd tensorrtllm_backend
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git submodule update --init --recursive
```

### Launch Triton TensorRT-LLM container

Launch Triton docker container with TensorRT-LLM backend.
Note that we're mounting `tensorrtllm_backend` to `/tensorrtllm_backend`
and the Llama2 model to `/Llama-2-7b-hf` in the docker container for simplicity.
Make an `engines` folder outside docker to reuse engines for future runs.
Please, make sure to replace <xx.yy> with the version of Triton that you want
to use.

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/tensorrtllm_backend>:/tensorrtllm_backend \
    -v </path/to/Llama2/repo>:/Llama-2-7b-hf \
    -v </path/to/engines>:/engines \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

Alternatively, you can follow instructions
[here](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#build-the-docker-container)
to build Triton Server with Tensorrt-LLM Backend if you want
to build a specialized container.

Don't forget to allow gpu usage when you launch the container.

> Optional: For simplicity, we've condensed all following steps into
> a [deploy_trtllm_llama.sh](tutorials/Popular_Models_Guide/Llama2/deploy_trtllm_llama.sh).
> Make sure to clone tutorials repo to your machine and start the docker
> container with the tutorial repo mounted to `/tutorials` by adding
> `-v /path/to/tutorials/:/tutorials` to docker run command, listed above.
> Then, when container has started, simply run the script via
> ```bash
> /tutorials/Popular_Models_Guide/Llama2/deploy_trtllm_llama.sh <WORLD_SIZE>
> ```
> For how to run an inference request, refer to the [Client](#client) section
> of this tutorial.

### Create Engines for each model [skip this step if you already have an engine]

TensorRT-LLM requires each model to be compiled for the configuration
you need before running. To do so, before you run your model for the first time
on Triton Server you will need to create a TensorRT-LLM engine.

Starting with [24.04 release](https://github.com/triton-inference-server/server/releases/tag/v2.45.0),
Triton Server TensrRT-LLM container comes with
pre-installed TensorRT-LLM package, which allows users to build engines inside
the Triton container. Simply follow the next steps:

```bash
HF_LLAMA_MODEL=/Llama-2-7b-hf
UNIFIED_CKPT_PATH=/tmp/ckpt/llama/7b/
ENGINE_DIR=/engines/llama-2-7b/1-gpu/
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
python3 ${CONVERT_CHKPT_SCRIPT} --model_dir ${HF_LLAMA_MODEL} --output_dir ${UNIFIED_CKPT_PATH} --dtype float16
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
            --remove_input_padding enable \
            --gpt_attention_plugin float16 \
            --context_fmha enable \
            --gemm_plugin float16 \
            --output_dir ${ENGINE_DIR} \
            --paged_kv_cache enable \
            --max_batch_size 4
```
> Optional: You can check test the output of the model with `run.py`
> located in the same llama examples folder.
>
>   ```bash
>    python3 /tensorrtllm_backend/tensorrt_llm/examples/run.py --engine_dir=/engines/llama-2-7b/1-gpu/ --max_output_len 50 --tokenizer_dir /Llama-2-7b-hf --input_text "What is ML?"
>    ```
> You should expect the following response:
> ```
> [TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
> ...
> Input [Text 0]: "<s> What is ML?"
> Output [Text 0 Beam 0]: "
> ML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."
> ```

### Serving with Triton

The last step is to create a Triton readable model. You can
find a template of a model that uses inflight batching in
[tensorrtllm_backend/all_models/inflight_batcher_llm](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main/all_models/inflight_batcher_llm).
To run our Llama2-7B model, you will need to:


1. Copy over the inflight batcher models repository

```bash
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
```

2. Modify config.pbtxt for the preprocessing, postprocessing and processing
steps. The following script do a minimized configuration to run tritonserver,
but if you want optimal performance or custom parameters, read details in
[documentation](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
and [perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md):
Note: `TRITON_BACKEND` has two possible options: `tensorrtllm` and `python`. If `TRITON_BACKEND=python`, the python backend will deploy [`model.py`](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/inflight_batcher_llm/tensorrt_llm/1/model.py).
```bash
# preprocessing
TOKENIZER_DIR=/Llama-2-7b-hf/
TOKENIZER_TYPE=auto
ENGINE_DIR=/engines/llama-2-7b/1-gpu/
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

3.  Launch Tritonserver

Use the [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) script. This launches multiple instances of `tritonserver` with MPI.
```bash
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/opt/tritonserver/inflight_batcher_llm
```
`<world size of the engine>` is the number of GPUs you want to use to run the engine. Set it to 1 for single GPU deployment.
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

### Send an inference request

You can test the results of the run with:
1. The [inflight_batcher_llm_client.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/client/inflight_batcher_llm_client.py) script.

```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend/inflight_batcher_llm/client:/tensorrtllm_client \
    -v /path/to/Llama2/repo:/Llama-2-7b-hf \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
# Install extra dependencies for the script
pip3 install transformers sentencepiece
python3 /tensorrtllm_client/inflight_batcher_llm_client.py --request-output-len 50 --tokenizer-dir /Llama-2-7b-hf/ --text "What is ML?"
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

### Evaluating performance with Gen-AI Perf
Gen-AI Perf is a command line tool for measuring the throughput and latency of generative AI models as served through an inference server.
You can read more about Gen-AI Perf [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/README.html).

To use Gen-AI Perf, run the following command in the same Triton docker container (i.e. nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk):
```bash
genai-perf \
  profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer /Llama-2-7b-hf/ \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```
You should expect an output that looks like this:
```
                                                  LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃              Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│   Request latency (ms) │ 1,630.23 │ 1,616.37 │ 1,644.65 │ 1,644.05 │ 1,638.70 │ 1,635.64 │
│ Output sequence length │   300.00 │   300.00 │   300.00 │   300.00 │   300.00 │   300.00 │
│  Input sequence length │   200.00 │   200.00 │   200.00 │   200.00 │   200.00 │   200.00 │
└────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
Output token throughput (per sec): 184.02
Request throughput (per sec): 0.61
2024-08-08 19:45 [INFO] genai_perf.export_data.json_exporter:56 - Generating artifacts/ensemble-triton-tensorrtllm-concurrency1/profile_export_genai_perf.json
2024-08-08 19:45 [INFO] genai_perf.export_data.csv_exporter:69 - Generating artifacts/ensemble-triton-tensorrtllm-concurrency1/profile_export_genai_perf.csv
```


## References

For more examples feel free to refer to [End to end workflow to run llama.](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
