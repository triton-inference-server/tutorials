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


# Conceptual Guides

| Related Pages | [Server Docs](https://github.com/triton-inference-server/server/tree/main/docs#triton-inference-server-documentation) |
| ------------ | --------------- |

Conceptual guides have been designed as an onboarding experience to Triton Inference Server. These guides will cover:
* [Part 1: Model Deployment](Part_1-model_deployment/): This guide talks about deploying and managing multiple models.
* [Part 2: Improving Resource Utilization](Part_2-improving_resource_utilization/): This guide discusses two popular features/techniques used to maximize a GPU's utilization whilst deploying models.
* [Part 3: Optimizing Triton Configuration](Part_3-optimizing_triton_configuration/): Each deployment has requirements specific to the use case. This guide walks users through the process of tailoring deployment configurations to match the SLAs.
* [Part 4: Accelerating Models](Part_4-inference_acceleration/): Another path towards achieving higher throughput is to accelerate the underlying models. This guide covers SDKs and tools which can be used to accelerate the models.
* [Part 5: Building Model Ensembles](./Part_5-Model_Ensembles/): Models are rarely used standalone. This guide will cover "how to build a deep learning inference pipeline?"
* [Part 6: Using the BLS API to build complex pipelines](Part_6-building_complex_pipelines/): Often times there are scenarios where the pipeline requires control flows. Learn how to work with complex pipelines with models deployed on different backends.
* [Part 7: Iterative Scheduling Tutorial](./Part_7-iterative_scheduling): Shows how to use the Triton Iterative Scheduler with a GPT2 model using HuggingFace Transformers.
* [Part 8: Semantic Caching](./Part_8-semantic_caching/): Shows benefits of adding semantic caching to you LLM-based workflow.
