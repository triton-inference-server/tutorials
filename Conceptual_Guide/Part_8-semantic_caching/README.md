<!--
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
-->

# Semantic caching

When deploying large language models (LLMs) or LLM-based workflows
there are two key factors to consider: the performance and cost-efficiency
of your application. Generating language model outputs requires significant
computational resources, for example GPU time, memory usage, and other
infrastructure costs. These resource-intensive requirements create a
pressing need for optimization strategies that can maintain
high-quality outputs while minimizing operational expenses.

Semantic caching emerges as a powerful solution to reduce computational costs
for LLM-based applications. Unlike traditional caching, it considers
the content and context of incoming requests.

## Definition and Main Benefits

**_Semantic caching_** is a caching mechanism that takes into account
the semantics of the incoming request, rather than just the raw data itself.
It goes beyond simple key-value pairs and considers the content or
context of the data.

This approach offers several benefits including, but not limited to:

+ **Cost Optimization**

    - Semantic caching can substantially reduce operational expenses associated
    with LLM deployments. By storing and reusing responses for semantically
    similar queries, it minimizes the number of actual LLM calls required.

+ **Reduced Latency**

    - One of the primary benefits of semantic caching is its ability to
    significantly improve response times. By retrieving cached responses for
    similar queries, the system can bypass the need for full model inference,
    resulting in the reduced latency.

+ **Increased Throughput**

    - Semantic caching allows for more efficient utilization of computational
    resources. By serving cached responses for similar queries, it reduces the
    load on infrastructure components. This efficiency enables the system
    to handle a higher volume of requests with the same hardware, effectively
    increasing throughput.

+ **Scalability**

    - The improved resource efficiency and reduced computational demands allows
    applications to serve more users without a proportional increase in
    infrastructure costs.

+ **Consistency in Responses**

    - For certain applications, maintaining consistency in responses to
    similar queries can be beneficial. Semantic caching ensures that analogous
    questions receive uniform answers, which can be particularly useful
    in scenarios like customer service or educational applications.

## Sample Reference Implementation

In this tutorial we provide a reference implementation for Semantic Cache in
[semantic_caching.py](tutorials/Conceptual_Guide/Part_8-semantic_caching/artifacts/semantic_caching.py)

## Further optimisations

## Interested in This Feature?

While this reference implementation provides a glimpse into the potential
of semantic caching, it's important to note that it's not an officially
supported feature in Triton Inference Server.

We value your input! If you're interested in seeing semantic caching as a
supported feature in future releases, we encourage you to [FILL IN]

Provide details about why you think semantic caching would be valuable for
your use case. Your feedback helps shape our product roadmap,
and we appreciate your contributions to making our software better for everyone.