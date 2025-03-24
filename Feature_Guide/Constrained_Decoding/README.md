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
- [Prerequisite: Hermes-2-Pro-Llama-3-8B](#prerequisite-hermes-2-pro-llama-3-8b)
- [Structured Generation via Prompt Engineering](#structured-generation-via-prompt-engineering)
    * [Example 1](#example-1)
    * [Example 2](#example-2)
- [Enforcig Output Format via External Libraries](#enforcig-output-format-via-external-libraries)
    * [Pre-requisite: Common set-up](#pre-requisite-common-set-up)
        + [Logits Post-Processor](#logits-post-processor)
        + [Tokenizer](#tokenizer)
        + [Repository set up](#repository-set-up)
    * [LM Format Enforcer](#lm-format-enforcer)
    * [Outlines](#outlines)

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
[Hermes-2-Pro-Llama-3-8B.](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
model with Triton Inference Server and TensorRT-LLM backend
following [these steps.](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)

## Structured Generation via Prompt Engineering

First, let's start Triton SDK container:
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials:/tutorials \
    -v /path/to/Hermes-2-Pro-Llama-3-8B/repo:/Hermes-2-Pro-Llama-3-8B \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

The provided client script uses `pydantic` library, which we do not ship with
the sdk container. Make sure to install it, before proceeding:

```bash
pip install pydantic
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
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200 --use-system-prompt
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
in [`client.py`](./artifacts/client.py) we use a `pydantic` library to define the
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
Let's try it out:

```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200 --use-system-prompt --use-schema
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

## Enforcing Output Format via External Libraries

In this section of the tutorial, we'll show how to impose constrains on LLMs,
which are not inherently fine-tuned for constrained decoding. We'll
[*LM Format Enforcer*](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file)
and [*Outlines*](https://github.com/outlines-dev/outlines?tab=readme-ov-file)
offer robust solutions.

The reference implementation for both libraries is provided in
[`utils.py`](./artifacts/utils.py) script, which also defines the output
format `AnswerFormat`:

```python
class WandFormat(BaseModel):
        wood: str
        core: str
        length: float

class AnswerFormat(BaseModel):
        name: str
        house: str
        blood_status: str
        occupation: str
        alive: str
        wand: WandFormat
```

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

+ executor_config.logits_post_processor_map = {
+            "<custom_logits_processor_name>": custom_logits_processor
+           }
self.executor = trtllm.Executor(model_path=...,
                                model_type=...,
                                executor_config=executor_config)
...
```

Additionally, if you want to enable logits post-processor for every request
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
and process it in `execute` function in
`inflight_batcher_llm/tensorrt_llm/1/model.py`:

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
In this tutorial, we're deploying Hermes-2-Pro-Llama-3-8B model as a part of an
ensemble. This means that the request is processed by the `ensemble` model
first, and then it is sent to `pre-processing model`, `tensorrt-llm model`, and
finally `post-processing`. This sequence defined in
`inflight_batcher_llm/ensemble/config.pbtxt` as well as input and output
mappings. Thus, we would need to update
`inflight_batcher_llm/ensemble/config.pbtxt` as well, so that `ensemble` model
properly passes additional input parameter to `tensorrt-llm model`:

```diff
input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  ...
  {
      name: "embedding_bias_weights"
      data_type: TYPE_FP32
      dims: [ -1 ]
      optional: true
- }
+ },
+ {
+   name: "logits_post_processor_name"
+   data_type: TYPE_STRING
+   dims: [ -1 ]
+   optional: true
+ }
]
output [
    ...
]
ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
    ...
    },
    {
      model_name: "tensorrt_llm"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "_INPUT_ID"
      }
      ...
      input_map {
        key: "bad_words_list"
        value: "_BAD_WORDS_IDS"
      }
+     input_map {
+       key: "logits_post_processor_name"
+       value: "logits_post_processor_name"
+     }
      output_map {
        key: "output_ids"
        value: "_TOKENS_BATCH"
      }
      ...
    }
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
    string_value: "/Hermes-2-Pro-Llama-3-8B"
  }
}
```
Simply append to the end on the `inflight_batcher_llm/tensorrt_llm/config.pbtxt`.

#### Repository set up

We've provided a sample implementation for *LM Format Enforcer* and *Outlines*
in [`artifacts/utils.py`](./artifacts/utils.py). Make sure you've copied it into
`/opt/tritonserver/inflight_batcher_llm/tensorrt_llm/1/lib` via

```bash
mkdir -p inflight_batcher_llm/tensorrt_llm/1/lib
cp /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/utils.py inflight_batcher_llm/tensorrt_llm/1/lib/
```
Finally, let's install all required libraries:

```bash
pip install pydantic lm-format-enforcer outlines setuptools
```

### LM Format Enforcer

To use LM Format Enforcer, make sure
`inflight_batcher_llm/tensorrt_llm/1/model.py` contains the following changes:

```diff
...
import tensorrt_llm.bindings.executor as trtllm

+ from lib.utils import LMFELogitsProcessor, AnswerFormat

...

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    ...

    def get_executor_config(self, model_config):
+       tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
+       logits_processor = LMFELogitsProcessor(tokenizer_dir, AnswerFormat.model_json_schema())
        kwargs = {
            "max_beam_width":
            get_parameter(model_config, "max_beam_width", int),
            "scheduler_config":
            self.get_scheduler_config(model_config),
            "kv_cache_config":
            self.get_kv_cache_config(model_config),
            "enable_chunked_context":
            get_parameter(model_config, "enable_chunked_context", bool),
            "normalize_log_probs":
            get_parameter(model_config, "normalize_log_probs", bool),
            "batching_type":
            convert_batching_type(get_parameter(model_config,
                                                "gpt_model_type")),
            "parallel_config":
            self.get_parallel_config(model_config),
            "peft_cache_config":
            self.get_peft_cache_config(model_config),
            "decoding_config":
            self.get_decoding_config(model_config),
+            "logits_post_processor_map":{
+                LMFELogitsProcessor.PROCESSOR_NAME: logits_processor
+            }
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.ExecutorConfig(**kwargs)
...
```

#### Send an inference request

First, let's start Triton SDK container:
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials/:/tutorials \
    -v /path/to/tutorials/repo:/tutorials \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

The provided client script uses `pydantic` library, which we do not ship with
the sdk container. Make sure to install it, before proceeding:

```bash
pip install pydantic
```

##### Option 1. Use provided [client script](./artifacts/client.py)

Let's first send a standard request, without enforcing the JSON answer format:
```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100
```

You should expect the following response:

```bash
Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and
```

Now, let's specify `logits_post_processor_name`  in our request:

```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100 --logits-post-processor-name "lmfe"
```

This time, the expected response looks like:
```bash
Who is Harry Potter?
		{
			"name": "Harry Potter",
			"occupation": "Wizard",
			"house": "Gryffindor",
			"wand": {
				"wood": "Holly",
				"core": "Phoenix feather",
				"length": 11
			},
			"blood_status": "Pure-blood",
			"alive": "Yes"
		}
```
As we can see, the schema, defined in [`utils.py`](./artifacts/utils.py) is
respected. Note, LM Format Enforcer lets LLM to control the order of generated
fields, thus re-ordering of fields is allowed.

##### Option 2. Use [generate endpoint](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint).

Let's first send a standard request, without enforcing the JSON answer format:
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```

You should expect the following response:

```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and"}
```

Now, let's specify `logits_post_processor_name`  in our request:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2, "logits_post_processor_name": "lmfe"}'
```

This time, the expected response looks like:
```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter?  \t\t\t\n\t\t{\n\t\t\t\"name\": \"Harry Potter\",\n\t\t\t\"occupation\": \"Wizard\",\n\t\t\t\"house\": \"Gryffindor\",\n\t\t\t\"wand\": {\n\t\t\t\t\"wood\": \"Holly\",\n\t\t\t\t\"core\": \"Phoenix feather\",\n\t\t\t\t\"length\": 11\n\t\t\t},\n\t\t\t\"blood_status\": \"Pure-blood\",\n\t\t\t\"alive\": \"Yes\"\n\t\t}\n\n\t\t\n\n\n\n\t\t\n"}
```

### Outlines

To use Outlines, make sure
`inflight_batcher_llm/tensorrt_llm/1/model.py` contains the following changes:

```diff
...
import tensorrt_llm.bindings.executor as trtllm

+ from lib.utils import OutlinesLogitsProcessor, AnswerFormat

...

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    ...

    def get_executor_config(self, model_config):
+       tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
+       logits_processor = OutlinesLogitsProcessor(tokenizer_dir, AnswerFormat.model_json_schema())
        kwargs = {
            "max_beam_width":
            get_parameter(model_config, "max_beam_width", int),
            "scheduler_config":
            self.get_scheduler_config(model_config),
            "kv_cache_config":
            self.get_kv_cache_config(model_config),
            "enable_chunked_context":
            get_parameter(model_config, "enable_chunked_context", bool),
            "normalize_log_probs":
            get_parameter(model_config, "normalize_log_probs", bool),
            "batching_type":
            convert_batching_type(get_parameter(model_config,
                                                "gpt_model_type")),
            "parallel_config":
            self.get_parallel_config(model_config),
            "peft_cache_config":
            self.get_peft_cache_config(model_config),
            "decoding_config":
            self.get_decoding_config(model_config),
+            "logits_post_processor_map":{
+                OutlinesLogitsProcessor.PROCESSOR_NAME: logits_processor
+            }
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.ExecutorConfig(**kwargs)
...
```

#### Send an inference request

First, let's start Triton SDK container:
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials/:/tutorials \
    -v /path/to/tutorials/repo:/tutorials \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

The provided client script uses `pydantic` library, which we do not ship with
the sdk container. Make sure to install it, before proceeding:

```bash
pip install pydantic
```

##### Option 1. Use provided [client script](./artifacts/client.py)

Let's first send a standard request, without enforcing the JSON answer format:
```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100
```

You should expect the following response:

```bash
Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and
```

Now, let's specify `logits_post_processor_name`  in our request:

```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100 --logits-post-processor-name "outlines"
```

This time, the expected response looks like:
```bash
Who is Harry Potter?{ "name": "Harry Potter","house": "Gryffindor","blood_status": "Pure-blood","occupation": "Wizards","alive": "No","wand": {"wood": "Holly","core": "Phoenix feather","length": 11 }}
```
As we can see, the schema, defined in [`utils.py`](./artifacts/utils.py) is
respected. Note, LM Format Enforcer lets LLM to control the order of generated
fields, thus re-ordering of fields is allowed.

##### Option 2. Use [generate endpoint](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint).

Let's first send a standard request, without enforcing the JSON answer format:
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```

You should expect the following response:

```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and"}
```

Now, let's specify `logits_post_processor_name`  in our request:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2, "logits_post_processor_name": "outlines"}'
```

This time, the expected response looks like:
```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter?{ \"name\": \"Harry Potter\",\"house\": \"Gryffindor\",\"blood_status\": \"Pure-blood\",\"occupation\": \"Wizards\",\"alive\": \"No\",\"wand\": {\"wood\": \"Holly\",\"core\": \"Phoenix feather\",\"length\": 11 }}"}
```
