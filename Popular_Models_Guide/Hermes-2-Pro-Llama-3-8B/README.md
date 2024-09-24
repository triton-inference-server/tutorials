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

# Deploying Hermes-2-Pro-Llama-3-8B Model with Triton Inference Server

The [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
is an advanced language model developed by [NousResearch](https://nousresearch.com/).
This model is an enhancement of the Meta-Llama-3-8B finetuned in-house using the
OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and
JSON Mode dataset developed by NousResearch. These advancements enable the model
to excel in both general conversational tasks and specialized functions like
structured JSON outputs and function calling, making it a versatile tool for
various applications.

The model is available for download through [huggingface](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B).

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

## Prerequisite: TensorRT-LLM backend

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

## Launch Triton TensorRT-LLM container

Launch Triton docker container with TensorRT-LLM backend.
Note that we're mounting `tensorrtllm_backend` to `/tensorrtllm_backend`
and the Hermes model to `/Hermes-2-Pro-Llama-3-8B` in the docker container for
simplicity. Make an `engines` folder outside docker to reuse engines for future
runs. Please, make sure to replace <xx.yy> with the version of Triton that you
want to use.

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/tensorrtllm_backend>:/tensorrtllm_backend \
    -v </path/to/Hermes/repo>:/Hermes-2-Pro-Llama-3-8B \
    -v </path/to/engines>:/engines \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

Alternatively, you can follow instructions
[here](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#build-the-docker-container)
to build Triton Server with Tensorrt-LLM Backend if you want to build
a specialized container.

Don't forget to allow gpu usage when you launch the container.

## Create Engines for each model [skip this step if you already have an engine]

TensorRT-LLM requires each model to be compiled for the configuration
you need before running. To do so, before you run your model for the first time
on Triton Server you will need to create a TensorRT-LLM engine.

Triton Server TensrRT-LLM container comes with pre-installed TensorRT-LLM
package, which allows users to build engines inside the Triton container.
Simply follow the next steps:

```bash
HF_LLAMA_MODEL=/Hermes-2-Pro-Llama-3-8B
UNIFIED_CKPT_PATH=/tmp/ckpt/hermes/8b/
ENGINE_DIR=/engines
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
>    python3 /tensorrtllm_backend/tensorrt_llm/examples/run.py --engine_dir=${ENGINE_DIR} --max_output_len 28 --tokenizer_dir ${HF_LLAMA_MODEL} --input_text "What is ML?"
>    ```
> You should expect the following response:
> ```
> Input [Text 0]: "<|begin_of_text|>What is ML?"
> Output [Text 0 Beam 0]: "
> Machine learning is a type of artificial intelligence (AI) that allows software applications to become more accurate in predicting outcomes without being explicitly programmed."
> ```

## Serving with Triton

The last step is to create a Triton readable model. You can
find a template of a model that uses inflight batching in
[tensorrtllm_backend/all_models/inflight_batcher_llm](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main/all_models/inflight_batcher_llm).
To run our model, you will need to:


1. Copy over the inflight batcher models repository

```bash
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
```

2. Modify `config.pbtxt` for the preprocessing, postprocessing and processing
steps. The following script do a minimized configuration to run tritonserver,
but if you want optimal performance or custom parameters, read details in
[documentation](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
and [perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md):

```bash
# preprocessing
TOKENIZER_DIR=/Hermes-2-Pro-Llama-3-8B/
TOKENIZER_TYPE=auto
DECOUPLED_MODE=false
MODEL_FOLDER=/opt/tritonserver/inflight_batcher_llm
MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=10000
TRTLLM_BACKEND=python
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRTLLM_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching
```

3.  Launch Tritonserver

> [!NOTE]
> This tutorial was prepared for serving a TensorRT-LLM model on a single GPU.
> Thus, in the following command use `--world_size=1` if the engine was built
> for a single GPU. Alternatively, if the engine requires multiple GPUs
> make sure to specify the exact number of GPUs required by the engine
> in `--world_size`.

Use the [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) script. This launches multiple instances of `tritonserver` with MPI.
```bash
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/opt/tritonserver/inflight_batcher_llm
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

## Send an inference request

You can test the results of the run with:
1. The [inflight_batcher_llm_client.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/client/inflight_batcher_llm_client.py) script.

First, let's start Triton SDK container:
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend/inflight_batcher_llm/client:/tensorrtllm_client \
    -v /path/to/Hermes-2-Pro-Llama-3-8B/repo:/Hermes-2-Pro-Llama-3-8B \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

Additionally, please install extra dependencies for the script:
```bash
pip3 install transformers sentencepiece
python3 /tensorrtllm_client/inflight_batcher_llm_client.py --request-output-len 28 --tokenizer-dir /Hermes-2-Pro-Llama-3-8B --text "What is ML?"
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
> {"context_logits":0.0,...,"text_output":"What is ML?\nMachine learning is a type of artificial intelligence (AI) that allows software applications to become more accurate in predicting outcomes without being explicitly programmed."}
> ```


## References

For more examples feel free to refer to [End to end workflow to run llama.](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)