<!--
# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Deploying E5 Text Embedding models with Triton Inference Server

[E5](https://arxiv.org/abs/2212.03533) is a family of text embedding models that can be used for several different purposes, including text retrieval and classification. In this example, we'll be deploying the [`e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2) model with Triton Inference Server, using the [TensorRT backend](https://github.com/triton-inference-server/tensorrt_backend). While this example is specific to the `e5-large-v2` model, it can be used as a baseline for other embedding models.

## Creating Model Repository

To deploy our e5 model, we'll need to create our model engine in a format that can be recognized by Triton, and place it in the proper directory structure.

We'll do this in two steps:

1. Exporting the model as an ONNX file
2. Compiling the exported ONNX file to a TensorRT plan

> [!TIP]
> You'll need to have [PyTorch](https://pytorch.org/) and [TensorRT](https://developer.nvidia.com/tensorrt) installed for this section.
> We recommend executing the steps in this section inside the [NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), which has the prerequisites needed.
> You can do this by running the following command
>
> ```bash
> docker run -it --rm --gpus all -v $(pwd):/workspace -v /tmp:/tmp nvcr.io/nvidia/pytorch:24.09-py3
> ```


### Exporting to ONNX

For exporting, we'll use the [Hugging Face optimum package](https://github.com/huggingface/optimum?tab=readme-ov-file), which has built in support for [exporting Hugging Face models to ONNX](https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model).

Note that here we're explicitly setting the batch size as `64`. Depending on your use case and hardware capacity, you may want to increase or decrease that number.

```bash
pip install optimum[exporters] sentence_transformers
optimum-cli export onnx --task sentence-similarity --model intfloat/e5-large-v2 /tmp/e5_onnx --batch_size 64
```

### Compile TRT Engine

Once the model is exported to ONNX, we can compile it into a [TensorRT Engine](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#ecosystem) by using [`trtexec`](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec). We'll also create our `model_repository` directory to save our engine into.

Note that we must explicitly set the minimum and maximum shapes for our model inputs here. The minimum shapes should be `1x1` for both the `input_ids` and `attention_mask` inputs, corresponding to a batch size and sequence length of 1. The maximum shapes should be `64x512`, where `64` the batch size matches the batch size set in the previous step, and `512` is the [maximum sequence length for the `e5-large-v2` model](https://huggingface.co/intfloat/e5-large-v2#limitations).

```bash
mkdir -p model_repository/e5/1

trtexec \
    --onnx=/tmp/e5_onnx/model.onnx \
    --saveEngine=/tmp/model_repository/e5/1/model.plan \
    --minShapes=input_ids:1x1,attention_mask:1x1 \
    --maxShapes=input_ids:64x512,attention_mask:64x512
```

## Deploy Triton

> [!TIP]
> If you used the NGC PyTorch container for the previous section, exit the container environment before executing the rest of the commands

With our model compiled and placed into our model repository, we can deploy our triton server by mounting it to and running the [tritonserver docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

```bash
docker run --gpus=1 --rm --net=host -v $(PWD)/model_repository:/models nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver --model-repository=/models
```

It may take some time to load the model and start the server. You should see a log message saying `"Started GRPCInferenceService at 0.0.0.0:8001"` when the server is ready.

## Send Request

Once our model is successfully deployed, we can start sending requests to it using the [`tritonclient`](https://github.com/triton-inference-server/client/tree/main#download-using-python-package-installer-pip) library.

You can use the following code snippet to begin using your deployed model. The model in Triton will expect text to be pre-tokenized, so we use the `transformers.AutoTokenizer` class to create our tokenizer.

> [!NOTE]
> For this model, you should include `query` or `passage` respectively at the beginning of your text when encoding for best retrieval performance.

```python
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

from transformers import AutoTokenizer

def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
input_texts = [
    "query: are judo throws allowed in wrestling?",
    "passage: Judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
]
tokenized_text = tokenizer(
    input_texts, max_length=512, padding=True, truncation=True, return_tensors="np"
)

triton_inputs = [
    prepare_tensor("input_ids", tokenized_text["input_ids"]),
    prepare_tensor("attention_mask", tokenized_text["attention_mask"]),
]

with grpcclient.InferenceServerClient(url="localhost:8001") as client:
    out = client.infer("e5", triton_inputs)

sentence_embedding = out.as_numpy("sentence_embedding")
token_embeddings = out.as_numpy("token_embeddings")

print(sentence_embedding)
```