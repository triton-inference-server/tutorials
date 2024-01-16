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
ARG BASE_IMAGE_TAG=23.12-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as triton-python-api

RUN apt-get update; apt-get install -y gdb

COPY ./deps/requirements.txt /tmp/requirements.txt

RUN pip install --timeout=2000 -r /tmp/requirements.txt

# Finish pyright install

 RUN pyright --help

ARG FRAMEWORK=HF_DIFFUSERS

COPY ./deps/requirements_torch.txt /tmp/requirements_torch.txt
RUN if [[ "$FRAMEWORK" == "DIFFUSERS" ]] ; then pip install --timeout=2000 -r /tmp/requirements_torch.txt ; fi

COPY ./deps/tritonserver-2.41.0.dev0-py3-none-any.whl /tmp/tritonserver-2.41.0.dev0-py3-none-any.whl

RUN pip3 install --force-reinstall --upgrade /tmp/tritonserver-2.41.0.dev0-py3-none-any.whl[all]

RUN ln -sf /bin/bash /bin/sh

COPY . /workspace

ARG RUN_TESTS=FALSE

RUN if [[ "$RUN_TESTS" == "TRUE" ]] ; then cd /tmp && git clone -b r23.12-python-api https://github.com/triton-inference-server/core.git && cp -rf /tmp/core/python/test /workspace/deps/ ; fi

RUN if [[ "$RUN_TESTS" == "TRUE" ]] ; then pytest /workspace/deps ; fi

