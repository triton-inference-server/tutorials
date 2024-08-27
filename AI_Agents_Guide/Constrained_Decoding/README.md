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
dedicated libraries like
[*LM Format Enforcer*](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file)
and [*Outlines*](https://github.com/outlines-dev/outlines?tab=readme-ov-file)
offer robust solutions. These libraries provide tools to enforce specific
constraints on model outputs, allowing developers to tailor the generation
process to meet precise requirements. By leveraging such libraries,
users can achieve greater control over the output, ensuring it aligns perfectly
with the desired criteria, whether that involves maintaining a certain format,
adhering to content guidelines, or ensuring grammatical correctness.
In this tutorial we'll show how to use *LM Format Enforcer* and *Outlines*
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

For fine-tuned model we can enable JSON mode by simply composing a system prompt
as:

```
You are a helpful assistant that answers in JSON.
```
Please, refer to [`client.py`](./artifacts/client.py) for full `prompt`
composition logic.

```bash
python3 artifacts/client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200
```
You should expect the following response:

 ```
 ...
assistant
{
  "title": "Harry Potter and the Order of Phoenix",
  "book_number": 5,
  "author": "J.K. Rowling",
  "series": "Harry Potter",
  "publication_date": "June 21, 2003",
  "page_count": 766,
  "publisher": "Arthur A. Levine Books",
  "genre": [
    "Fantasy",
    "Adventure",
    "Young Adult"
  ],
  "awards": [
    {
      "award_name": "British Book Award",
      "category": "Children's Book of the Year",
      "year": 2004
    }
  ],
  "plot_summary": "Harry Potter and the Order of Phoenix is the fifth book in the Harry Potter series. In this installment, Harry returns to Hogwarts School of Witchcraft and Wizardry for his fifth year. The Ministry of Magic is in denial about the return of Lord Voldemort, and Harry finds himself battling against the

```

### Example 2

Optionally, we can also restrict an output to a specific schema. For example,
in [`client.py`](./artifacts/client.py) we use a `pydentic` library to define the
following answer format:

```python
from pydantic import BaseModel

class AnswerFormat(BaseModel):
    title: str
    year: int
    director: str
    producer: str
    plot: str

...

prompt += "Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>".format(
                schema=AnswerFormat.model_json_schema())

```

```bash
python3 artifacts/client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200 --use-schema
```
You should expect the following response:

```
 ...
assistant
{
  "title": "Harry Potter and the Order of Phoenix",
  "year": 2007,
  "director": "David Yates",
  "producer": "David Heyman",
  "plot": "Harry Potter and his friends must protect Hogwarts from a threat when the Ministry of Magic is taken over by Lord Voldemort's followers."
}

```

## Enforcig Output Format via External Libraries

In this section of the tutorial, we'll show how to impose constrains on LLMs,
which are not inherently fine-tuned for constrained decoding. We'll
[*LM Format Enforcer*](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file)
and [*Outlines*](https://github.com/outlines-dev/outlines?tab=readme-ov-file)
offer robust solutions.

### Pre-requisite: Common set-up

Make sure you've successfully deployed Hermes-2-Pro-Llama-3-8B model
with Triton Inference Server and TensorRT-LLM backend following
[these steps](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md).
> [!IMPORTANT]
> Make sure that the `tutorials` folder is mounted to `/tutorials`, when you
> start the docker container.


Upon successful setup you should have `/opt/tritonserver/inflight_batcher_llm`
folder and try a couple of inference requests (e.g. those, provided in
[example 1](#example-1) or [example 2](#example-2)).

We'll do some adjusments to model files, thu if you have a running server, you
can stop it via:
```bash
pkill tritonserver
```

#### Logits Post-Processor

Both of the libraries limit the set of allowed tokens at every generation stage.
In TensorRT-LLM, user can define a custom
[logits post-processor](https://nvidia.github.io/TensorRT-LLM/advanced/batch-manager.html#logits-post-processor-optional)
to mask logits, which should never be used in the current generation step.

For TensorRT-LLM models, deployed via `python` backend (i.e. when
[`triton_backend`](https://github.com/triton-inference-server/tensorrtllm_backend/blob/8aaf89bcf723dad112839fd36cbbe09e2e439c63/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt#L28C10-L28C29)
is set to `python` in `tensorrt_llm/config.pbtxt`, Triton's python backend will
use
[`model.py`](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/inflight_batcher_llm/tensorrt_llm/1/model.py)
to serve your TensorRT-LLM model.), custom logits processor should be specified
during model's initialization as a part of
[Executor's](https://nvidia.github.io/TensorRT-LLM/executor.html#executor-api)
configuration
([`logits_post_processor_map`](https://github.com/NVIDIA/TensorRT-LLM/blob/32ed92e4491baf2d54682a21d247e1948cca996e/tensorrt_llm/hlapi/llm_utils.py#L205)).
Below is the sample for reference.

```diff
...

executor_config = self.get_executor_config(model_config)
+ executor_config.logits_post_processor_map = {
+            "<custom_logits_processor_name>": custom_logits_processor
+           }
self.executor = trtllm.Executor(gpt_model_path,
                                trtllm.ModelType.DECODER_ONLY,
                                executor_config)
...
```

Additionally, if you want to enable logits pos-processor for every request
individually, you can do so via an additional `input` parameter.
For example, in this tutorial we will add `logits_post_processor_name` in
`inflight_batcher_llm/tensorrt_llm/config.pbtxt`:
```diff
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
    allow_ragged_batch: true
  },
  ...
  {
    name: "lora_config"
	data_type: TYPE_INT32
	dims: [ -1, 3 ]
	optional: true
	allow_ragged_batch: true
- }
+ },
+ {
+   name: "logits_post_processor_name"
+   data_type: TYPE_STRING
+   dims: [ -1 ]
+   optional: true
+ }
]
...
```
and process it in `inflight_batcher_llm/tensorrt_llm/1/model.py`:
```diff
def execute(self, requests):
    """`execute` must be implemented in every Python model. `execute`
    function receives a list of pb_utils.InferenceRequest as the only
    argument. This function is called when an inference is requested
    for this model.
    Parameters
    ----------
    requests : list
      A list of pb_utils.InferenceRequest
    Returns
    -------
    list
      A list of pb_utils.InferenceResponse. The length of this list must
      be the same as `requests`
    """
    ...

    for request in requests:
        response_sender = request.get_response_sender()
        if get_input_scalar_by_name(request, 'stop'):
            self.handle_stop_request(request.request_id(), response_sender)
        else:
            try:
                converted = convert_request(request,
                                            self.exclude_input_from_output,
                                            self.decoupled)
+               logits_post_processor_name = get_input_tensor_by_name(request, 'logits_post_processor_name')
+               if logits_post_processor_name is not None:
+                   converted.logits_post_processor_name = logits_post_processor_name.item().decode('utf-8')
            except Exception as e:
            ...
```
If you follow along with this tutorial, make sure same changes are incorporated
into corresponding files of `/opt/tritonserver/inflight_batcher_llm` repository.

#### Tokenizer

Both [*LM Format Enforcer*](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file)
and [*Outlines*](https://github.com/outlines-dev/outlines?tab=readme-ov-file)
require tokenizer access at initialization time. In this tutorial,
we'll be exposing tokenizer via `inflight_batcher_llm/tensorrt_llm/config.pbtxt`
parameter:

```txt
parameters: {
  key: "tokenizer_dir"
  value: {
    string_value: "/mnt/Code/tickets/function_calling_models/Hermes-2-Pro-Llama-3-8B"
  }
}
```
Simply append to the end on the `inflight_batcher_llm/tensorrt_llm/config.pbtxt`.

#### Repository set up

We've provided a sample implementation for *LM Format Enforcer* and *Outlines*
in [`artifacts/utils.py`](./artifacts/utils.py). Make sure you've copied

### LM Format Enforcer

```diff
...

executor_config = self.get_executor_config(model_config)
+ tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
+ logits_lmfe_processor_answer_format = LMFELogitsProcessor(tokenizer_dir, AnswerFormat.model_json_schema())
+ executor_config.logits_post_processor_map = {
+            str(LMFELogitsProcessor.PROCESSOR_NAME"): logits_lmfe_processor_answer_format
+           }
self.executor = trtllm.Executor(gpt_model_path,
                                trtllm.ModelType.DECODER_ONLY,
                                executor_config)
...
```


