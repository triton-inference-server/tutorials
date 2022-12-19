# Deploying HuggingFace models

Developers often work with open source models. HuggingFace is a popular source of many open source models. The discussion in this guide will focus on how a user can deploy almost any model from HuggingFace with the Triton Inference Server. For this example, the [CLIP](https://openai.com/blog/clip/) model available on [HuggingFace](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel.forward.example) is being used.

There are two primary methods of deploying a model on Triton:
* **Approach 1:** Deploy the pipeline without explicitly breaking apart model from a pipeline. The core advantage of this approach is that users can quickly deploy their pipeline.

* **Approach 2:** Break apart the pipeline, use a different backends for pre/post processing and deploying the core model on a framework backend. The advantage in this case is that running the core network on a dedicated framework backend provides higher performance. Additionally, many framework specific optimizations can be leveraged. See [Part 4](../Conceptual_Guide/Part_4-inference_acceleration/README.md) of the conceptual guide for more information.
![multiple models](./img/Approach.PNG)

## Approach 1

This approach will make use of Triton's python backend.

### What is a Python Backend?

The Triton Inference Server has a ["Python Backend"](https://github.com/triton-inference-server/python_backend). This backend is designed to let users serve code/models written in python to be deployed using the Triton Inference Server without writing any C++ code. In the Triton Ecosystem, any code deployed on a Python backend is often referred to as a "Python model".

### Example explanation

In this example, we make use of two functions of Python backend's functions: `initialize()` and `execute()`.


```
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```