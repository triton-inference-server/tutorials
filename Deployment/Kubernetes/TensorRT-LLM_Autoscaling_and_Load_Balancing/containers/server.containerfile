# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_CONTAINER_IMAGE=nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
ARG ENGINE_DEST_PATH=/var/run/engines
ARG HF_HOME=/var/run/cache

FROM ${BASE_CONTAINER_IMAGE}

# Set a set of useful labels.
LABEL "base"="${BASE_CONTAINER_IMAGE}"
LABEL "role"="server"

# Stop APT (Debian package manager) from complaining about interactivity.
ENV DEBIAN_FRONTEND=noninteractive
# Set additional environment values that make usage more pleasant.
ENV TERM=xterm-256color

# Set Triton CLI environment variables which control where
# TRTLLM engine and model files are downloaded to; and where
# the path to the Huggingface cache.
ENV ENGINE_DEST_PATH ${ENGINE_DEST_PATH}
ENV HF_HOME ${HF_HOME}

# Set the active working directory.
WORKDIR /workspace

# Install a custom version of Triton CLI that support Tensor parallelism and
# the 70B version of Llama models.
RUN pip --verbose install \
    --no-cache-dir \
    --no-color \
    --no-input \
    git+https://github.com/triton-inference-server/triton_cli.git@jwyman/aslb-mn

# Copy the server script.
COPY server.py .

RUN apt list --installed \
 && pip list --version

ENTRYPOINT [ "/bin/bash" ]
