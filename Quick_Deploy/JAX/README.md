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

# Deploying a JAX Model

This README showcases how to deploy a simple ResNet model on Triton Inference Server. While Triton doesn't yet have a dedicated JAX backend, JAX/Flax models can be deployed using [Python Backend](https://github.com/triton-inference-server/python_backend). If you are new to Triton, it is recommended to watch this [getting started video](https://www.youtube.com/watch?v=NQDtfSi5QF4) and review [Part 1](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment) of the conceptual guide before proceeding. For the purposes of demonstration, we are using a pre-trained model provided by [flaxmodels](https://github.com/matthias-wright/flaxmodels).

Before diving into the specifics execution, an understanding of the underlying structure is needed. To use a JAX or a Flax model, the recommended path for this is using a ["Python Model"](https://github.com/triton-inference-server/python_backend#python-backend). Python models in Triton are classes with three Triton-specific functions: `initialize`, `execute` and `finalize`. Users can customize this class to serve any python function they write or any model they want as long as it can be loaded in python runtime. The `initialize` function runs when the python model is loaded into memory, and the `finalize` function runs when the model is unloaded from memory. Both of these functions are optional to define. For the purposes of this example, we will use the `initialize` and the `execute` functions to load and run(respectively) a `resnet18` model.

We use the initialize method to load in the model weights and create our Flax model object. Here, we load a pretrained model from the flaxmodels library. You could also load weights from another pretrained model library, or from a file located in the model directory. Note that with JAX, our model parameters are automatically loaded onto any available accelerator, like a GPU.

In the execute function, we perform the actual model inference. Note that the input to the `execute` method is an arbitrary length _list_ of request objects that may have been dynamically batched together. In this example, we loop through and execute each request individually and append each response into the `responses` list. If your model supports batched inputs, you may find it more efficient to execute all of the requests in one function call.

```python
import triton_python_backend_utils as pb_utils
import jax
import flaxmodels as fm

import numpy as np

class TritonPythonModel:

    def initialize(self, args):

        self.key = jax.random.PRNGKey(0)
        self.resnet18 = fm.ResNet18(output='logits', pretrained='imagenet')


    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "image")
            input_image = inp.as_numpy()

            params = self.resnet18.init(self.key, input_image)
            out = self.resnet18.apply(params, input_image, train=False)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "fc_out",
                    np.array(out),
                )
            ])
            responses.append(inference_response)
        return responses
```

## Step 1: Set Up Triton Inference Server

To use Triton, we need to build a model repository. The structure of the repository is as follows:

```text
model_repository/
└── resnet50
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

For this example, we have pre-built the model repository. Next, we install the required dependencies and launch the Triton Inference Server.

```bash
# Replace the yy.mm in the image name with the release year and month
# of the Triton version needed, eg. 22.12
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3 bash

# Note: See JAX install guide for more details on installing JAX: https://github.com/google/jax#installation
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install --upgrade git+https://github.com/matthias-wright/flaxmodels.git

tritonserver --model-repository=/models
```

## Step 2: Using a Triton Client to Query the Server

Let's breakdown the client application. First, we setup a connection with the Triton Inference Server.

```python
client = httpclient.InferenceServerClient(url="localhost:8000")
```

Then we set the input and output arrays.

```python
# Set Inputs
input_tensors = [
    httpclient.InferInput("image", image.shape, datatype="FP32")
]
input_tensors[0].set_data_from_numpy(image)

# Set outputs
outputs = [
    httpclient.InferRequestedOutput("fc_out")
]
```

Lastly, we query send a request to the Triton Inference Server.

```python
# Query
query_response = client.infer(model_name="resnet50",
                                inputs=input_tensors,
                                outputs=outputs)

# Output
out = query_response.as_numpy("fc_out")
```
