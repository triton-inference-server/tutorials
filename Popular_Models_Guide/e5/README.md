# Deploying E5 Text Embedding models with Triton Inference Server


## Create model repo

```bash
mkdir -p model_repository/e5/1
```

```bash
docker run -it --rm --gpus all -v $(pwd):/workspace -v /tmp:/tmp nvcr.io/nvidia/pytorch:24.09-py3
```

## Download and Export to ONNX
```bash
pip install optimum[exporters] sentence_transformers
optimum-cli export onnx --task sentence-similarity --model intfloat/e5-large-v2 /tmp/e5_onnx --batch_size 64
```

## Compile TRT Engine
```bash
trtexec \
    --onnx=/tmp/e5_onnx/model.onnx \
    --saveEngine=/tmp/model_repository/e5/1/model.plan \
    --minShapes=input_ids:1x1,attention_mask:1x1 \
    --maxShapes=input_ids:64x512,attention_mask:64x512
```

## Deploy Triton
```bash
docker run --gpus=1 --rm --net=host -v /tmp/model_repository:/models nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver --model-repository=/models
```

## Send Request

Note that for this model, you need to include `query` or `passage` when encoding text for best performance.

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