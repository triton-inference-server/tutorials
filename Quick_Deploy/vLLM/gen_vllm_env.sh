#!/bin/bash
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


#
# This script creates a conda environment for Triton with vllm
# dependencies.
#

# Pick the release tag from the container environment variable
RELEASE_TAG="r${NVIDIA_TRITON_SERVER_VERSION}"

# Install and setup conda environment
file_name="Miniconda3-latest-Linux-x86_64.sh"
rm -rf ./miniconda  $file_name
wget https://repo.anaconda.com/miniconda/$file_name

# install miniconda in silent mode
bash $file_name -p ./miniconda -b

# activate conda
eval "$(./miniconda/bin/conda shell.bash hook)"

# Installing cmake and dependencies
apt update && apt install software-properties-common rapidjson-dev libarchive-dev zlib1g-dev -y
# Using CMAKE installation instruction from:: https://apt.kitware.com/
apt install -y gpg wget && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
        gpg --dearmor - |  \
        tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    . /etc/os-release && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | \
    tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake cmake-data

conda create -n vllm_env python=3.10 -y
conda activate vllm_env
export PYTHONNOUSERSITE=True
conda install -c conda-forge libstdcxx-ng=12 -y
conda install -c conda-forge conda-pack -y

# vllm needs cuda 11.8 to run properly
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

pip install numpy vllm

rm -rf python_backend
git clone https://github.com/triton-inference-server/python_backend -b $RELEASE_TAG
(cd python_backend/ && mkdir builddir && cd builddir && \
cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=$RELEASE_TAG -DTRITON_COMMON_REPO_TAG=$RELEASE_TAG -DTRITON_CORE_REPO_TAG=$RELEASE_TAG ../ && \
make -j18 triton-python-backend-stub)

conda-pack -n vllm_env -o vllm_env.tar.gz
chmod 755 vllm_env.tar.gz

mv vllm_env.tar.gz ./model_repository/vllm/
mv python_backend/builddir/triton_python_backend_stub ./model_repository/vllm/

conda deactivate


# Clean-up
rm -rf ./miniconda  $file_name
rm -rf python_backend
