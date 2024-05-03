<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TensorRT-LLM is Nvidia's recommended solution of running Large Language
Models(LLMs) on Nvidia GPUs. Read more about TensoRT-LLM [here](https://github.com/NVIDIA/TensorRT-LLM)
and Triton's TensorRTLLM Backend [here](https://github.com/triton-inference-server/tensorrtllm_backend).

*NOTE:* If some parts of this tutorial doesn't work, it is possible that there
are some version mismatches between the `tutorials` and `tensorrt_backend` repository.
Refer to [llama.md](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
for more detailed modifications if necessary. And if you are familier with python, you can also try using [High-level API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/high-level-api/README.md) for LLM workflow.


## Pre-build instructions

For this tutorial, we are using the Llama2-7B HuggingFace model with pre-trained weights.
Clone the repo of the model with weights and tokens [here](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main).
You will need to get permissions for the Llama2 repository as well as get access to the huggingface cli. To get access to the huggingface cli, go here: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Installation

1. The installation starts with cloning the TensorRT-LLM Backend and update the TensorRT-LLM submodule:
```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch <release branch>
# Update the submodules
cd tensorrtllm_backend
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git submodule update --init --recursive
```

2. Launch Triton docker container with TensorRT-LLM backend. Note I'm mounting `tensorrtllm_backend` to `/tensorrtllm_backend` and the Llama2 model to `/Llama-2-7b-hf` in the docker container for simplicity. Make an `engines` folder outside docker to reuse engines for future runs.
```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend:/tensorrtllm_backend \
    -v /path/to/Llama2/repo:/Llama-2-7b-hf \
    -v /path/to/engines:/engines \
    nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3
```

Alternatively, you can follow instructions [here](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md) to build Triton Server with Tensorrt-LLM Backend if you want to build a specialized container.

Don't forget to allow gpu usage when you launch the container.

## Create Engines for each model [skip this step if you already have an engine]
TensorRT-LLM requires each model to be compiled for the configuration you need before running. To do so, before you run your model for the first time on Triton Server you will need to create a TensorRT-LLM engine for the model for the configuration you want with the following steps:

1. Install Tensorrt-LLM python package
   ```bash
    # Install CMake
    bash /tensorrtllm_backend/tensorrt_llm/docker/common/install_cmake.sh
    export PATH="/usr/local/cmake/bin:${PATH}"

    # PyTorch needs to be built from source for aarch64
    ARCH="$(uname -i)"
    if [ "${ARCH}" = "aarch64" ]; then TORCH_INSTALL_TYPE="src_non_cxx11_abi"; \
    else TORCH_INSTALL_TYPE="pypi"; fi && \
    (cd /tensorrtllm_backend/tensorrt_llm &&
        bash docker/common/install_pytorch.sh $TORCH_INSTALL_TYPE &&
        python3 ./scripts/build_wheel.py --trt_root=/usr/local/tensorrt &&
        pip3 install ./build/tensorrt_llm*.whl)
    ```

2.  Compile model engines

    The script to build Llama models is located in [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples). We use the one located in the docker container as `/tensorrtllm_backend/tensorrt_llm/examples/llama/build.py`.
    This command compiles the model with inflight batching and 1 GPU. To run with more GPUs, you will need to change the build command to use `--world_size X`.
    More details for the scripting please see the documentation for the Llama example [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama/README.md).

    ```bash
    python /tensorrtllm_backend/tensorrt_llm/examples/llama/build.py --model_dir /Llama-2-7b-hf/ \
                    --dtype bfloat16 \
                    --use_gpt_attention_plugin bfloat16 \
                    --use_inflight_batching \
                    --paged_kv_cache \
                    --remove_input_padding \
                    --use_gemm_plugin bfloat16 \
                    --output_dir /engines/1-gpu/ \
                    --world_size 1
    ```

    > Optional: You can check test the output of the model with `run.py`
    > located in the same llama examples folder.
    >
    >   ```bash
    >    python3 /tensorrtllm_backend/tensorrt_llm/examples/llama/run.py --engine_dir=/engines/1-gpu/ --max_output_len 100 --tokenizer_dir /Llama-2-7b-hf --input_text "How do I count to ten in French?"
    >    ```

## Serving with Triton

The last step is to create a Triton readable model. You can
find a template of a model that uses inflight batching in [tensorrtllm_backend/all_models/inflight_batcher_llm](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main/all_models/inflight_batcher_llm).
To run our Llama2-7B model, you will need to:


1. Copy over the inflight batcher models repository

 ```bash
 cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
 ```

2. Modify config.pbtxt for the preprocessing, postprocessing and processing steps. The following script do a minimized configuration to run tritonserver, but if you want optimal performance or custom parameters, read details in [documentation](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md) and [perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/perf_best_practices.md):

    ```bash
    # preprocessing
    TOKENIZER_DIR=/Llama-2-7b-hf/
    TOKENIZER_TYPE=auto
    DECOUPLED_MODE=false
    ENGINE_DIR=/engines/1-gpu/
    MODEL_FOLDER=/opt/tritonserver/inflight_batcher_llm
    MAX_BATCH_SIZE=64
    FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
    python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE}
    python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE}
    python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE}
    python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
    python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR}
    ```
    Also, ensure that the `gpt_model_type` parameter is set to `inflight_fused_batching`.

3.  Launch Tritonserver

    Use the [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) script. This launches multiple instances of `tritonserver` with MPI.
    ```bash
    python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/opt/tritonserver/inflight_batcher_llm
    ```

## Client

You can test the results of the run with:
1. The [inflight_batcher_llm_client.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/client/inflight_batcher_llm_client.py) script.

```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend:/tensorrtllm_backend \
    -v /path/to/Llama2/repo:/Llama-2-7b-hf \
    -v /path/to/engines:/engines \
    nvcr.io/nvidia/tritonserver:23.10-py3-sdk
# Install extra dependencies for the script
pip3 install transformers sentencepiece
python3 /tensorrtllm_backend/inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer_type llama --tokenizer_dir /Llama-2-7b-hf
```

2. The [generate endpoint](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint) if you are using the Triton TensorRT-LLM Backend container with versions greater than `r23.10`.


