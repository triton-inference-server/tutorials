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
ARG BASE_IMAGE_TAG=24.08-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as triton-python-api

RUN apt-get update; apt-get install -y gdb

RUN --mount=type=bind,source=./deps/requirements.txt,target=/tmp/requirements.txt \
    pip install --timeout=2000 --requirement /tmp/requirements.txt

# Finish pyright install

RUN pyright --help

RUN find /opt/tritonserver/python -maxdepth 1 -type f -name \
    "tritonserver-*.whl" | xargs -I {} pip3 install --upgrade {}[all]

# grafana
RUN apt-get install -y adduser libfontconfig1 musl && \
    wget https://dl.grafana.com/enterprise/release/grafana-enterprise_11.2.0_amd64.deb && \
    dpkg -i grafana-enterprise_11.2.0_amd64.deb && \
    rm -rf grafana-enterprise_11.2.0_amd64.deb

RUN ln -sf /bin/bash /bin/sh

