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

Note: This tutorial is for TensorRT-LLM Backend which is currently under development.

## Pre-build instructions

For this tutorial, we are using the Llama2-7B HuggingFace model with pre-trained weights.
Clone the repo of the model with weights and tokens [here](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main). You will need to get permissions for the Llama2 repository as well as get access to the huggingface cli. To get access to the huggingface cli, go here: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Installation

Launch Triton docker container with TensorRT-LLM backend 
```docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-py3 bash```

Alternatively, you can follow instructions [here](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md) to build Tritonserver with Tensorrt-LLM Backend if you want to build a specialized container. 

Don't forget to allow gpu usage when you launch the container.

## Create Engines for each model [skip this step if you already have a engine]
TensorRT-LLM requires each model to be compiled for the configuration you need before running. 
To do so, before you run your model for the first time on Tritonserver you will need to create a TensorRT-LLM engine for the model for the configuration you want. 
To do so, you will need to complete the following steps:

1. Install Tensorrt-LLM python package
   ```bash
# TensorRT-LLM is required for generating engines. 
pip install git+https://github.com/NVIDIA/TensorRT-LLM.git
mkdir /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/
cp /opt/tritonserver/backends/tensorrtllm/* /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/
```

2.  Log in to huggingface-cli

    ```bash
    huggingface-cli login --token hf_*****
    ```

3.  Compile model (3 min)

    <!-- ```bash
    python3 examples/llama/build.py \
        --model_dir meta-llama/Llama-2-7b-chat-hf \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --output_dir ../tensorrt_llm_backend/all_models/gpt/tensorrt_llm/1 \
        --world_size 1
    ``` -->

    ```bash
    python3 examples/llama/build.py \
        --model_dir meta-llama/Llama-2-7b-chat-hf \
        --dtype float16 \
        --use_gpt_attention_plugin float16 \
        --use_gemm_plugin float16 \
        --output_dir ../tensorrt_llm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1 \
        --world_size 1
    ```


python build.py --model_dir /mnt/nvdl/usr/katheriney/Llama-2-7b-chat-hf/ \
                --dtype bfloat16 \
                --use_gpt_attention_plugin bfloat16 \
                --use_inflight_batching \
                --paged_kv_cache \
                --remove_input_padding \
                --use_gemm_plugin bfloat16 \
                --output_dir /mnt/nvdl/usr/katheriney/engines/bf16/1-gpu/

    ```bash
    python3 examples/llama/build.py 
        --model_dir /data/meta-llama/Llama-2-7b-chat-hf/ \
        --dtype float16  --use_gpt_attention_plugin bfloat16 \
        --use_gemm_plugin bfloat16 \
        --output_dir /data/meta-llama/gen/7B/trt_engines/bf16/1-gpu/ 
    ```
    
    > Optional: You can check test the output of the model with the following command:
    >
    >   ```bash
    >    python3 examples/llama/run.py --engine_dir=../tensorrt_llm_backend/all_models/gpt/tensorrt_llm/1/ --max_output_len 100 --tokenizer_dir meta-llama/Llama-2-7b-chat-hf --input_text "How do I count to ten in French?"
    >    ```

## Serving with Triton

> Note: WIP, this part doesnt work yet because it uses the wrong tokenizer

13. Launch Triton Docker container

    ```bash
    cd ..
    docker run -it --rm --gpus all --network host --shm-size=1g -v $(pwd)/all_models:/app/all_models triton_trt_llm
    ```

    <!-- ```bash
    docker run -it --rm --gpus all --network host --shm-size=1g -v $(pwd)/../llama_quickstart/model_repository:/app/all_models/gpt triton_trt_llm
    ``` -->

14. Install TensorRT-LLM into container (6 sec)

    ```bash
    pip install tensorrt_llm/build/tensorrt_llm-0.1.3-py3-none-any.whl
    ```

15. Modify model config file

    <!-- ```bash
    sed -i 's#${engine_dir}#/app/all_models/gpt/tensorrt_llm/1#' all_models/gpt/tensorrt_llm/config.pbtxt
    ``` -->

    ```bash
    sed -i 's#${decoupled_mode}#true#' all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt
    sed -i 's#${engine_dir}#/app/all_models/inflight_batcher_llm/tensorrt_llm/1#' all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt
    ```

16. Launch Tritonserver (20 mins)

    ```bash
    tritonserver --model-repository /app/all_models/gpt --log-verbose 1
    ```

    ```bash
    tritonserver --model-repository /app/all_models/inflight_batcher_llm --log-verbose 1
    ```

## Client

WIP


