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

# Deploying a Hugging Face Transformer Models in Triton

The following tutorial demonstrates how to deploy an arbitrary hugging face transformer
model on the Triton Inference Server using Triton's [Python backend](https://github.com/triton-inference-server/python_backend). For the purposes of this example, three transformer
models will be deployed:
- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) for text generation requests.
- [gpt2](https://huggingface.co/gpt2) for text generation requests.
- [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) for text classification requests.

These models were selected because of their small size and consistent response quality, 
enabling the tutorial to run on most devices with sensible results. However, this tutorial
is also generalizable for much larger models provided sufficient infrastructure. 

*NOTE*: The tutorial is intended to be a reference example only. It is a work in progress.

## Step 1: Build a Model Repository

The first step is to create a model repository that the Triton Inference Server will use
for inference processing. We can create a model repository from the provided base files by 
executing the following python script:

```
python3 create_repository.py
```

Without specifying any arguments, this script will create a model repository containing
the details necessary for Triton to load and serve each of the aforementioned models.
For the purposes of this tutorial, we provide Triton with simple ```config.pbtxt``` files
for each model that describe:
- The the backend to use.
- Model input and output details.
- Custom parameters to use for execution.

## Step 2: Build a Triton Container Image

The second step is to create an image that includes all the dependencies necessary
to deploy hugging face transformer models on the Triton Inference Server. This can be done
by building an image from the provided Dockerfile.

```
docker build -t triton_transformer_server .
```

## Step 3: Run the Triton Inference Server

Once the ```triton_transformer_server``` image is created, you can run a container with
the following command:

```bash
docker run --gpus all -it --rm -p 8000:8000 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server tritonserver --model-repository=model_repository
```

The server has launched successfully when you see the following outputs in your console:

```bash
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Step 4: Use a Triton Client to Query the Server

Now we can query the server using the client script provided.

```bash
python3 client.py -m [model_name] --prompt [prompt_message]
```
*NOTE*: You may need to install packages for the client to run. See [client.py](client.py)
for details and run the script in an environment manager if you are concerned about 
version collisions.

In the case of this tutorial, we have included models that are specialized for one of 
two tasks: text classification or text generation. The ```distilbert-base-uncased-finetuned-sst-2-english```
model is specialized for text classification and we can query the model in the following way:

```bash
python3 client.py -m distilbert-base-uncased-finetuned-sst-2-english --prompt "I feel great!"
```

Which will return a result similar to:
```
| Name        | Shape       | Data              |
| ----------- | ----------- | ----------------- |
| prompt      | [1]         | ['I feel great!'] |
| label       | (1,)        | POSITIVE          |
| score       | (1,)        | [0.99986434]      |
```

Similarly, we can query our models specialized for text generation in this way:

```bash
python3 client.py -m opt-125m --prompt "Who am I?"
python3 client.py -m gpt2 --prompt "Who am I?"
```
Which will return a result similar to:
```
# facebook/opt-125m response

| Name        | Shape       | Data                  |
| ----------- | ----------- | --------------------- |
| prompt      | [1]         | ['Who am I?']         |
| text        | (1,)        | I am a writer, artist,|
|             |             | and photographer      |

# gpt2 response

| Name        | Shape       | Data                  |
| ----------- | ----------- | --------------------- |
| prompt      | [1]         | ['Who am I?']         |
| text        | (1,)        | Who am I? What am I?  |
|             |             | Why am I here?        |
```

## Next Steps

The base model files used to create the model repository can be found in the ```text_classification```
and ```text_generation``` directories. These base models have been kept minimal in order to maximize 
generalizability. Should you wish to modify the behavior of the transformer models, such 
as increasing the number of sequences a text generator model returns, you should modify these
files directly and re-run the ```create_repository.py``` script.

For a more custom deployment, please see our [model configuration guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) to see how the scope of this tutorial can be expanded to fit your needs.
