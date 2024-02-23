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

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=24.01-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as triton-python-api

RUN apt-get update; apt-get install -y gdb

COPY ./deps/requirements.txt /tmp/requirements.txt

RUN pip install --timeout=2000 -r /tmp/requirements.txt

# Finish pyright install

RUN pyright --help

ARG FRAMEWORK=DIFFUSERS

COPY ./deps/requirements_torch.txt /tmp/requirements_torch.txt
RUN if [[ "$FRAMEWORK" == "DIFFUSERS" ]] ; then pip install --timeout=2000 -r /tmp/requirements_torch.txt ; fi

RUN if [[ "$FRAMEWORK" == "DIFFUSERS_XL" ]] ; then pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt ; fi

RUN if [[ "$FRAMEWORK" == "DIFFUSERS_XL" ]] ; then git clone https://github.com/NVIDIA/TensorRT.git -b release/9.2 --single-branch /tmp/TensorRT ; fi

RUN if [[ "$FRAMEWORK" == "DIFFUSERS_XL" ]] ; then pip3 install -r /tmp/TensorRT/demo/Diffusion/requirements.txt; fi

COPY ./deps/requirements_trtllm.txt /tmp/requirements_trtllm.txt
RUN if [[ "$FRAMEWORK" == "TRT_LLM" ]] ; then pip install --timeout=2000 -r /tmp/requirements_trtllm.txt ; fi

COPY ./deps/triton_cli /tmp/triton_cli
RUN if [[ "$FRAMEWORK" == "TRT_LLM" ]] ; then cd /tmp/triton_cli && pip install . --timeout=2000 ; fi

COPY ./deps/tritonserver-2.41.0.dev0-py3-none-any.whl /tmp/tritonserver-2.41.0.dev0-py3-none-any.whl

RUN find /opt/tritonserver/python -maxdepth 1 -type f -name \
    "tritonserver-*.whl" | xargs -I {} pip3 install --force-reinstall --upgrade {}[all]

RUN pip3 show tritonserver 1>/dev/null || \
    if [ $? != 0 ]; then \
       pip3 install /tmp/tritonserver-2.41.0.dev0-py3-none-any.whl[all] ;\
    fi

RUN ln -sf /bin/bash /bin/sh

COPY . /workspace

ARG RUN_TESTS=FALSE

RUN if [[ "$RUN_TESTS" == "TRUE" ]] ; then cd /tmp && git clone -b r24.01 https://github.com/triton-inference-server/core.git && cp -rf /tmp/core/python/test /workspace/deps/ ; fi

RUN if [[ "$RUN_TESTS" == "TRUE" ]] ; then pytest /workspace/deps ; fi

