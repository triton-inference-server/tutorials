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

# Deploying Hugging Face Transformer Models in Triton

The following tutorial demonstrates how to deploy an arbitrary hugging face transformer
model on the Triton Inference Server using Triton's [Python backend](https://github.com/triton-inference-server/python_backend). For the purposes of this example, three transformer
models will be deployed:
- [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base)
- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

These models were selected because of their popularity and consistent response quality.
However, this tutorial is also generalizable for any transformer model provided 
sufficient infrastructure. 

*NOTE*: The tutorial is intended to be a reference example only. It is a work in progress.

## Step 1: Create a Model Repository

The first step is to create a model repository containing the models we want the Triton 
Inference Server to load and use for inference processing. For this tutorial an empty
directory named `model_repository` has been provided, into which we will copy the 
`falcon7b` model folder:

```
cp -r falcon7b/ model_repository/ 
```

The `falcon7b/` folder we copied is organized in the way Triton expects and contains 
two important files needed to serve models in Triton:
- **config.pbtxt** - Outlines the backend to use, model input/output details, and custom
parameters to use for execution. More information on the full range of model configuration
properties Triton supports can be found [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).
- **model.py** - Implements how Triton should handle the model during the initialization, 
execution, and finalization stages. More information regarding python backend usage 
can be found [here](https://github.com/triton-inference-server/python_backend#usage).

## Step 2: Build a Triton Container Image

The second step is to create an image that includes all the dependencies necessary
to deploy hugging face transformer models on the Triton Inference Server. This can be done
by building an image from the provided Dockerfile:

```
docker build -t triton_transformer_server .
```

## Step 3: Launch the Triton Inference Server

Once the ```triton_transformer_server``` image is created, you can launch the Triton Inference
Server in a container with the following command:

```bash
docker run --gpus all -it --rm -p 8000:8000 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server tritonserver --model-repository=model_repository
```

The server has launched successfully when you see the following outputs in your console:

```
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Step 4: Use a Triton Client to Query the Server

Now we can query the server using curl, specifying the server address and input details:

```json
curl -X POST localhost:8000/v2/models/falcon7b/infer -d '{"inputs": [{"name":"prompt","datatype":"BYTES","shape":[1],"data":["I am going"]}]}'
```
In our testing, the server returned the following result (formatted for legibility):
```json
{
  "model_name": "falcon7b",
  "model_version": "1",
  "outputs": [
    {
      "name": "text",
      "datatype": "BYTES",
      "shape": [
        1
      ],
      "data": [
        "I am going to be in the market for a new laptop soon. I"
      ]
    }
  ]
}
```

## Step 5: Host Multiple Models in Triton

So far in this tutorial, we have only loaded a single model. However, Triton is capable
of hosting many models, simultaneously. To accomplish this, first ensure you have
exited the docker container by invoking `Ctrl+C` and waiting for the container to exit.

Next copy the remaining models provided into the model repository:
```
cp -r mistral7b/ model_repository/
cp -r persimmon8b/ model_repository/
```
*NOTE*: The combined size of these three models is large. If your current hardware cannot
support hosting all three models simultaneously, consider copying only a single additional
model.

Again, launch the server by invoking the `docker run` command from above and wait for confirmation
that the server has launched successfully.

Query the server making sure to change the host address for each model:
```json
curl -X POST localhost:8000/v2/models/falcon7b/infer -d '{"inputs": [{"name":"prompt","datatype":"BYTES","shape":[1],"data":["How can you be"]}]}'
curl -X POST localhost:8000/v2/models/mistral7b/infer -d '{"inputs": [{"name":"prompt","datatype":"BYTES","shape":[1],"data":["Where are you going"]}]}'
curl -X POST localhost:8000/v2/models/persimmon8b/infer -d '{"inputs": [{"name":"prompt","datatype":"BYTES","shape":[1],"data":["Where is the nearest"]}]}'
```
In our testing, these queries return the following parsed results:
```bash
# falcon7b
"How can you be sure that you are getting the best deal on your car"

# mistral7b
"Where are you going? I’m going to the beach."

# persimmon8b
"Where is the nearest starbucks?"
```
## 'Hour Zero' Support

At the time of writing this tutorial, transformers version 4.34.0 was not yet released, meaning
very new models such as Persimmon-8B and Mistral 7B were not yet fully supported by the latest
transformers releases (4.33.3). Triton is not limited to waiting for official releases and can
load the latest models by building from source. This can be done by replacing:
```docker
RUN pip install transformers==4.34.0
```
with:
```docker
RUN pip install git+https://github.com/huggingface/transformers.git
```
in the provided Dockerfile. Using this technique, we were able to load Mistral 7B into Triton
within minutes of hearing about its release.

## Next Steps

The `model.py` files have been kept minimal in order to maximize generalizability. Should you wish 
to modify the behavior of the transformer models, such as increasing the number of generated sequences 
to return, be sure to modify the corresponding `config.pbtxt` and `model.py` files and copy them 
into the `model_repository`.

The transformers used in this tutorial were all suited for text-generation tasks, however, this 
is not a limitation. The principles of this tutorial can be applied to server models suited for
any other transformer task.

For a more custom deployment, please see our [model configuration guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) to see how the scope of this tutorial can be expanded to fit your needs.
