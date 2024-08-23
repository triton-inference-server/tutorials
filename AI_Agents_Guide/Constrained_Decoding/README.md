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

# Constrained Decoding with Triton Inference Server

This tutorial focuses on constrained decoding, an important technique for
ensuring that large language models (LLMs) generate outputs that adhere
to strict formatting requirements—requirements that may be challenging or
expensive to achieve solely through fine-tuning.

## Table of Contents

- [Introduction to Constrained Decoding](#introduction-to-constrained-decoding)
- [Structured Generation via Prompt Engineering](#structured-generation-via-prompt-engineering)
- [Enforcig Output Format via External Libraries](#enforcig-output-format-via-external-libraries)
    * [LM Format Enforcer](#lm-format-enforcer)
    * [Outlines](#outlines)

## Pre-requisites

In this tutorial we'll use the [Hermes-2-Pro-Llama-3-8B.](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
Please, follow the [Deploying Hermes-2-Pro-Llama-3-8B Model with Triton Inference Server](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)
tutorial to prepare the set up, needed for this tutorial.

## Introduction to Constrained Decoding

Constrained decoding is a powerful technique used in natural language processing
and various AI applications to guide and control the output of a model.
By imposing specific constraints, this method ensures that generated outputs
adhere to predefined criteria, such as length, format, or content restrictions.
This capability is essential in contexts where compliance with rules
is non-negotiable, such as producing valid code snippets, structured data,
or grammatically correct sentences.

In recent advancements, some models are already fine-tuned to incorporate
these constraints inherently. These models are designed
to seamlessly integrate constraints during the generation process, reducing
the need for extensive post-processing. By doing so, they enhance the efficiency
and accuracy of tasks that require strict adherence to predefined rules.
This built-in capability makes them particularly valuable in applications
like automated content creation, data validation, and real-time language
translation, where precision and reliability are paramount.

This tutorial is based on [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B),
, which already supports JSON Structured Outputs. An extensive instruction stack
on deploying Hermes-2-Pro-Llama-3-8B model with Triton Inference Server and
TensorRT-LLM backend can be found in [this](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)
tutorial. The structure and quality of a produced output in such cases can be
controlled through prompt engineering. To explore this path, please refer to
[Structured Generation via Prompt Engineering](#structured-generation-via-prompt-engineering)
section on the tutorial.

For scenarios where models are not inherently fine-tuned for
constrained decoding, or when more precise control over the output is desired,
dedicated libraries like lm-format-enforcer and outlines offer robust solutions.
These libraries provide tools to enforce specific constraints on model outputs,
allowing developers to tailor the generation process to meet precise
requirements. By leveraging such libraries, users can achieve greater
control over the output, ensuring it aligns perfectly with the desired criteria,
whether that involves maintaining a certain format, adhering to content
guidelines, or ensuring grammatical correctness. In this tutorial we'll show
how to use [*LM Format Enforcer*](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file)
and [*Outlines*](https://github.com/outlines-dev/outlines?tab=readme-ov-file)
in your workflow.

## Prerequisite: Hermes-2-Pro-Llama-3-8B

Before proceeding, please make sure that you've successfully deployed
Hermes-2-Pro-Llama-3-8B model with Triton Inference Server and
TensorRT-LLM backend following [these steps](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)

## Structured Generation via Prompt Engineering

First, let's start Triton SDK container:
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend/inflight_batcher_llm/client:/tensorrtllm_client \
    -v /path/to/Hermes-2-Pro-Llama-3-8B/repo:/Hermes-2-Pro-Llama-3-8B \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```
### Example 1

```bash
python3 client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200
```
You should expect the following response:

> ```
> ...
>assistant
>{
>  "title": "Harry Potter and the Order of Phoenix",
>  "book_number": 5,
>  "author": "J.K. Rowling",
>  "series": "Harry Potter",
>  "publication_date": "June 21, 2003",
>  "page_count": 766,
>  "publisher": "Arthur A. Levine Books",
>  "genre": [
>    "Fantasy",
>    "Adventure",
>    "Young Adult"
>  ],
>  "awards": [
>    {
>      "award_name": "British Book Award",
>      "category": "Children's Book of the Year",
>      "year": 2004
>    }
>  ],
>  "plot_summary": "Harry Potter and the Order of Phoenix is the fifth book in the Harry Potter series. In this installment, Harry returns to Hogwarts School of Witchcraft and Wizardry for his fifth year. The Ministry of Magic is in denial about the return of Lord Voldemort, and Harry finds himself battling against the
>
> ``

### Example 2

```bash
python3 client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200 --use-schema
```
You should expect the following response:

> ```
> ...
>assistant
>{
>  "title": "Harry Potter and the Order of Phoenix",
>  "year": 2007,
>  "director": "David Yates",
>  "producer": "David Heyman",
>  "plot": "Harry Potter and his friends must protect Hogwarts from a threat when the Ministry of Magic is taken over by Lord Voldemort's followers."
>}
>
>
> ``
