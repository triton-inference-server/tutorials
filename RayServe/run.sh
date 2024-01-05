#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#export AWS_DEFAULT_REGION=<REPLACE_ME>
#export AWS_ACCESS_KEY_ID=<REPLACE_ME>
#export AWS_SECRET_ACCESS_KEY=<REPLACE_ME>

export S3_BUCKET_URL="s3://model-repo-example"

INSTALL_DIR=/usr/local/lib/python3.10/dist-packages/tritonserver
LOCAL_DIR=~/python_beta_api/core/python/tritonserver

docker run --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -eHF_TOKEN -eAWS_DEFAULT_REGION -eAWS_ACCESS_KEY_ID -eAWS_SECRET_ACCESS_KEY -eS3_BUCKET_URL -v ${PWD}:/workspace -v${PWD}/models_test:/workspace/models -w /workspace --name rayserve-triton rayserve-triton

# docker run --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -eHF_TOKEN -eAWS_DEFAULT_REGION -eAWS_ACCESS_KEY_ID -eAWS_SECRET_ACCESS_KEY -eS3_BUCKET_URL -v ${PWD}:/workspace -v${PWD}/models_test:/workspace/models -w /workspace --name rayserve-triton -v${LOCAL_DIR}/__init__.py:${INSTALL_DIR}/__init__.py -v${LOCAL_DIR}/_api:${INSTALL_DIR}/_api rayserve-triton


# docker run --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -eHF_TOKEN -eAWS_DEFAULT_REGION -eAWS_ACCESS_KEY_ID -eAWS_SECRET_ACCESS_KEY -eS3_BUCKET_URL -v ${PWD}:/workspace -v${PWD}/models_test:/workspace/models -w /workspace --name rayserve-triton -v${LOCAL_DIR}/__init__.py:${INSTALL_DIR}/__init__.py -v${LOCAL_DIR}/_api:${INSTALL_DIR}/_api -v${LOCAL_DIR}/_c/__init__.pyi:${INSTALL_DIR}/_c/__init__.pyi -v${LOCAL_DIR}/__init__.pyi:${INSTALL_DIR}/__init__.pyi -v${LOCAL_DIR}/_c/triton_bindings.pyi:${INSTALL_DIR}/_c/triton_bindings.pyi rayserve-triton

#docker run --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -eHF_TOKEN -eAWS_DEFAULT_REGION -eAWS_ACCESS_KEY_ID -eAWS_SECRET_ACCESS_KEY -eS3_BUCKET_URL -v ${PWD}:/workspace -v${PWD}/models_test:/workspace/models -w /workspace --name rayserve-triton rayserve-triton
