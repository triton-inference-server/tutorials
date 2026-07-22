<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# HSTU Generative Recommenders on Triton

[Hierarchical Sequential Transduction Units (HSTU)](https://arxiv.org/abs/2402.17152)
power **Generative Recommenders (GRs)**: recommendation workloads reformulated as
generative modeling over high-cardinality, non-stationary event streams. HSTU
supports both retrieval and ranking style tasks.

Triton Inference Server can serve HSTU models through the
[PyTorch backend](https://github.com/triton-inference-server/pytorch_backend)
using ahead-of-time (AOT) Inductor packages (`platform: "torch_aoti"`). Training,
export, KV-cache runtime, and end-to-end examples live in NVIDIA's
[recsys-examples](https://github.com/NVIDIA/recsys-examples) repository rather
than in this tutorials tree.

## Where to go next

| Resource | Description |
| -------- | ----------- |
| [HSTU overview](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/README.md) | Architecture, training, and inference entry points |
| [HSTU inference](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/inference/README.md) | Inference features, KV-cache, AOTInductor export, and KuaiRand examples |
| [PyTorch AOTI on Triton](https://github.com/triton-inference-server/pytorch_backend#aot-inductor-support-beta) | `torch_aoti` model repository layout and configuration |

> [!NOTE]
> Use the recsys-examples guides for building, exporting, and validating HSTU
> models. This page only points Triton users at that workflow and the Torch AOTI
> serving path.
