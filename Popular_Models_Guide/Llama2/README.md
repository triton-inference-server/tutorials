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

# Pre-build instructions

For this tutorial, we are using the Llama2-7B HuggingFace model with pre-trained weights.
Clone the repo of the model with weights and tokens [here](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main).
You will need to get permissions for the Llama2 repository as well as get access to the huggingface cli. To get access to the huggingface cli, go here: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

There are multiple ways to run Llama2 with Tritonserver.
1. Infer with [TensorRT-LLM Backend](trtllm_guide.md#infer-with-tensorrt-llm-backend)
2. Infer with [vLLM Backend](vllm_guide.md#infer-with-vllm-backend)
3. Infer with [Python Backend as a HuggingFace model](../Quick_Deploy/HuggingFaceTransformers/README.md#deploying-hugging-face-transformer-models-in-triton)
