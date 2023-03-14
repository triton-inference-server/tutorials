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

## Step 1: Set Up Triton Inference Server

To use Triton, we need to build a model repository. The structure of the repository as follows:
```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.py
```
For this example, we have pre-built the model repository. Next, we install the required dependencies and launch the Triton Inference Server.

```
# Replace the yy.mm in the image name with the release year and month
# of the Triton version needed, eg. 22.12
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3 bash

pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade git+https://github.com/matthias-wright/flaxmodels.git

```

## Step 2: Using a Triton Client to Query the Server

Let's breakdown the client application. First, we setup a connection with the Triton Inference Server.
```
client = httpclient.InferenceServerClient(url="localhost:8000")
```
Then we set the input and output arrays.
```
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

```
# Query
query_response = client.infer(model_name="resnet50",
                                inputs=input_tensors,
                                outputs=outputs)

# Output
out = query_response.as_numpy("fc_out")
```

