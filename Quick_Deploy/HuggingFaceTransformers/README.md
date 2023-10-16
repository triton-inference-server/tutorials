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
model on the Triton Inference Server using Triton's [Python backend](https://github.com/triton-inference-server/python_backend). For the purposes of this example, two transformer
models will be deployed:
- [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base)

These models were selected because of their popularity and consistent response quality.
However, this tutorial is also generalizable for any transformer model provided
sufficient infrastructure.

*NOTE*: The tutorial is intended to be a reference example only. It may not be tuned for
optimal performance.

## Step 1: Create a Model Repository

The first step is to create a model repository containing the models we want the Triton
Inference Server to load and use for inference processing. To accomplish this, create a
directory called `model_repository` and copy the `falcon7b` model folder into it:

```
mkdir -p model_repository
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
docker run --gpus all -it --rm --net=host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server tritonserver --model-repository=model_repository
```

The server has launched successfully when you see the following outputs in your console:

```
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Step 4: Query the Server

Now we can query the server using curl, specifying the server address and input details:

```json
curl -X POST localhost:8000/v2/models/falcon7b/infer -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["I am going"]}]}'
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

Next copy the remaining model provided into the model repository:
```
cp -r persimmon8b/ model_repository/
```
*NOTE*: The combined size of these two models is large. If your current hardware cannot
support hosting both models simultaneously, consider loading a smaller model, such as
[opt-125m](https://huggingface.co/facebook/opt-125m), by creating a folder for it
using the templates provided and copying it into `model_repository`.

Again, launch the server by invoking the `docker run` command from above and wait for confirmation
that the server has launched successfully.

Query the server making sure to change the host address for each model:
```json
curl -X POST localhost:8000/v2/models/falcon7b/infer -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["How can you be"]}]}'
curl -X POST localhost:8000/v2/models/persimmon8b/infer -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["Where is the nearest"]}]}'
```
In our testing, these queries returned the following parsed results:
```bash
# falcon7b
"How can you be sure that you are getting the best deal on your car"

# persimmon8b
"Where is the nearest starbucks?"
```

## 'Day Zero' Support

The latest transformer models may not always be supported in the most recent, official
release of the `transformers` package. In such a case, you should still be able to
load these 'bleeding edge' models in Triton by building `transformers` from source.
This can be done by replacing the transformers install directive in the provided
Dockerfile with:
```docker
RUN pip install git+https://github.com/huggingface/transformers.git
```
Using this technique you should be able to serve any transformer models supported by
hugging face with Triton.


# Next Steps
The following sections expand on the base tutorial and provide guidance for future sandboxing.

## Loading Cached Models
In the previous steps, we downloaded the falcon-7b model from hugging face when we
launched the Triton server. We can avoid this lengthy download process  in subsequent runs
by loading cached models into Triton. To do so, we can follow the hugging face [tutorial
for downloading models](https://huggingface.co/docs/hub/models-downloading). Once the model
is downloaded, we can mount it to the Triton container by adding the following mount option to our
`docker run` command from earlier (making sure to replace {USER} with your username on
Linux):

```bash
# Option to mount a specific cached model (falcon-7b in this case)
-v /home/{USER}/.cache/huggingface/hub/models--tiiuae--falcon-7b:/root/.cache/huggingface/hub/models--tiiuae--falcon-7b

# Option to mount all cached models on the host system
-v /home/{USER}/.cache/huggingface:/root/.cache/huggingface
```

## Triton Tool Ecosystem
Deploying models in Triton also comes with the benefit of access to a fully-supported suite
of deployment analyzers to help you better understand and tailor your systems to fit your
needs. Triton currently has two options for deployment analysis:
- [Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2310/user-guide/docs/user_guide/perf_analyzer.html): An inference performance optimizer.
- [Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html) A GPU memory and compute utilization optimizer.

### Performance Analyzer
To use the performance analyzer, please ensure the Triton server is still active with the falcon7b
model or restart it using the docker run command from above.

Once Triton launches successfully, start a Triton SDK container by running the following in a separate window:

```bash
docker run -it --net=host nvcr.io/nvidia/tritonserver:23.09-py3-sdk bash
```
This container comes with all of Triton's deployment analyzers pre-installed, meaning
we can simply enter the following to get feedback on our model's inference performance:

```bash
perf_analyzer -m falcon7b --concurrency-range 1:4
```

This command should run quickly and profile the performance of our falcon7b model at
increasing levels of concurrency. As the analyzer runs, it will output useful metrics
such as latency percentiles, latency by stage of inference, and successful request
count. Ultimately, the analyzer will neatly summarize the data in the final output:

```json
Concurrency: 1, throughput: 23.2174 infer/sec, latency 43041 usec
Concurrency: 2, throughput: 23.3284 infer/sec, latency 85590 usec
Concurrency: 3, throughput: 23.94 infer/sec, latency 125085 usec
Concurrency: 4, throughput: 23.773 infer/sec, latency 167879 usec
```

This is a single, simple use case for the performance analyzer. For more information and
a more complete list of performance analyzer parameters and use cases, please see
[this](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2310/user-guide/docs/user_guide/perf_analyzer.html) guide.

### Model Analyzer

To use the model analyzer, please terminate your Triton server by invoking `Ctrl+C` and relaunching
it with the following command:
```bash
docker run --gpus all -it --rm --net=host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository -v /home/{USER}/.cache/huggingface/hub/models--tiiuae--falcon-7b:/root/.cache/huggingface/hub/models--tiiuae--falcon-7b triton_transformer_server tritonserver --model-repository=model_repository --model-control-mode=explicit
```

Similarly, exit your Triton SDK container by invoking `Ctrl+C` and relanch it with the following
command (ensuring the model_repository path is correct):
```bash
docker run -it --net=host -v ${PWD}/model_repository:/opt/tritonserver/model_repository nvcr.io/nvidia/tritonserver:23.09-py3-sdk bash
```

Once Triton launches successfully, enter the following command into your SDK container:
```bash
model-analyzer profile -m /opt/tritonserver/model_repository/ --profile-models falcon7b --run-config-search-mode quick --triton-launch-mode=remote
```
This tool will take longer to execute than the Performance Analyzer example. However, once it
is complete, the model analyzer will provide you a full summary relating to throughput, latency,
and hardware utilization in csv and pdf format.

## Customization

The `model.py` files have been kept minimal in order to maximize generalizability. Should you wish
to modify the behavior of the transformer models, such as increasing the number of generated sequences
to return, be sure to modify the corresponding `config.pbtxt` and `model.py` files and copy them
into the `model_repository`.

The transformers used in this tutorial were all suited for text-generation tasks, however, this
is not a limitation. The principles of this tutorial can be applied to serve models suited for
any other transformer task.

Triton offers a rich variety of available server configuration options not mentioned in this tutorial.
For a more custom deployment, please see our [model configuration guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) to see how the scope of this tutorial can be expanded to fit your needs.
