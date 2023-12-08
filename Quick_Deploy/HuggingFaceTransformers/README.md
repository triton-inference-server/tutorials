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
model on the Triton Inference Server using Triton's [Python backend](https://github.com/triton-inference-server/python_backend).
For the purposes of this example, the following transformer models will be deployed:
- [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b)

These models were selected because of their popularity and consistent response quality.
However, this tutorial is also generalizable for any transformer model provided
sufficient infrastructure.

*NOTE*: The tutorial is intended to be a reference example only. It may not be tuned for
optimal performance.

*NOTE*: Llama 2 models are not specifically mentioned in the steps below, but
can be run if `tiiuae/falcon-7b` is replaced with `meta-llama/Llama-2-7b-hf`,
and `falcon7b` folder is replaced by `llama7b` folder.

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

**Note**: For private models like `Llama2`, you need to [request access to the model](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) and add the [access token](https://huggingface.co/settings/tokens) to the docker command `-e PRIVATE_REPO_TOKEN=<hf_your_huggingface_access_token>`.
```bash
docker run --gpus all -it --rm --net=host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -e PRIVATE_REPO_TOKEN=<hf_your_huggingface_access_token> -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server tritonserver --model-repository=model_repository
```

The server has launched successfully when you see the following outputs in your console:

```
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## Step 4: Query the Server

Now we can query the server using curl, specifying the server address and input details:

```bash
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
```bash
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
Beginning in the 23.10 release, users can now interact with large language models (LLMs) hosted
by Triton in a simplified fashion by using Triton's generate endpoint:

```bash
curl -X POST localhost:8000/v2/models/falcon7b/generate -d '{"text_input":"How can you be"}'
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
launched the Triton server. We can avoid this lengthy download process in subsequent runs
by loading cached models into Triton. By default, the provided `model.py` files will cache
the falcon and persimmon models in their respective directories within the `model_repository`
folder. This is accomplished by setting the `TRANSFORMERS_CACHE` environmental variable.
To set this environmental variable for an abtitrary model, include the following lines in
your `model.py` **before** importing the 'transformers' module, making sure to replace
`{MODEL}` with your target model.

```python
import os
os.environ['TRANSFORMERS_CACHE'] = '/opt/tritonserver/model_repository/{MODEL}/hf_cache'
```

Alternatively, if your system has already cached a hugging face model you wish to deploy in Triton,
you can mount it to the Triton container by adding the following mount option to the `docker run`
command from earlier (making sure to replace `${HOME}` with the path to your associated username's home directory):

```bash
# Option to mount a specific cached model (falcon-7b in this case)
-v ${HOME}/.cache/huggingface/hub/models--tiiuae--falcon-7b:/root/.cache/huggingface/hub/models--tiiuae--falcon-7b

# Option to mount all cached models on the host system
-v ${HOME}/.cache/huggingface:/root/.cache/huggingface
```

## Triton Tool Ecosystem
Deploying models in Triton also comes with the benefit of access to a fully-supported suite
of deployment analyzers to help you better understand and tailor your systems to fit your
needs. Triton currently has two options for deployment analysis:
- [Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2310/user-guide/docs/user_guide/perf_analyzer.html): An inference performance optimizer.
- [Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html) A GPU memory and compute utilization optimizer.

### Performance Analyzer
To use the performance analyzer, please remove the persimmon8b model from `model_repository` and restart
the Triton server using the `docker run` command from above.

Once Triton launches successfully, start a Triton SDK container by running the following in a separate window:

```bash
docker run -it --net=host nvcr.io/nvidia/tritonserver:23.10-py3-sdk bash
```
This container comes with all of Triton's deployment analyzers pre-installed, meaning
we can simply enter the following to get feedback on our model's inference performance:

```bash
perf_analyzer -m falcon7b --collect-metrics
```

This command should run quickly and profile the performance of our falcon7b model.
As the analyzer runs, it will output useful metrics such as latency percentiles,
latency by stage of inference, and successful request count. A subset of the output
data is shown below:

```bash
#Avg request latency
46307 usec (overhead 25 usec + queue 25 usec + compute input 26 usec + compute infer 46161 usec + compute output 68 usec)

#Avg GPU Utilization
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 66.0556%

#Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 21.3841 infer/sec, latency 46783 usec
```

These metrics tell us that we are not fully utilizing our hardware and that our
throughput is low. We can immediately improve these results by batching our requests
instead of computing inferences one at a time. The `model.py` file for the falcon model
is already configured to handle batched requests. Enabling batching in Triton is as simple
as adding the following to falcon's `config.pbtxt` file:

```
dynamic_batching { }
max_batch_size: 8
```
The integer corresponding to the `max_batch_size`, can be any of your choosing, however,
for this example, we select 8. Now let's re-run the perf_analyzer with increasing levels
of concurrency and see how it impacts GPU utilization and throughput by executing:
```bash
perf_analyzer -m falcon7b --collect-metrics --concurrency-range=2:16:2
```
After executing for a few minutes, the performance analyzer should return
results similar to these (depending on hardware):
```bash
# Concurrency = 4
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 74.1111%
Throughput: 31.8264 infer/sec, latency 125174 usec

# Concurrency = 8
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 81.7895%
Throughput: 46.2105 infer/sec, latency 172920 usec

# Concurrency = 16
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 90.5556%
Throughput: 53.6549 infer/sec, latency 299178 usec
```
Using the performance analyzer we were able to quickly profile different model configurations
to obtain better throughput and hardware utilization. In this case, we were able to
identify a configuration that nearly triples our throughput and increases GPU
utilization by ~24% in less than 5 minutes.

This is a single, simple use case for the performance analyzer. For more information and
a more complete list of performance analyzer parameters and use cases, please see
[this](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2310/user-guide/docs/user_guide/perf_analyzer.html)
guide.

For more information regarding dynamic batching in Triton, please see [this](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher)
guide.

### Model Analyzer

In the performance analyzer section, we used intuition to increase our throughput by changing
a subset of variables and measuring the difference in performance. However, we only changed
a few variables across a wide search space.

To sweep this parameter space in a more robust fashion, we can use Triton's model analyzer, which
not only sweeps a large spectrum of configuration parameters, but also generates visual reports
to analyze post-execution.

To use the model analyzer, please terminate your Triton server by invoking `Ctrl+C` and relaunching
it with the following command (ensuring the dynamic_batching parameters from above have been added
to the falcon model's config.pbtxt):
```bash
docker run --gpus all -it --rm --net=host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server
```

Next, to get the most accurate GPU metrics from the model analyzer, we will install and launch it from
our local server container. To accomplish this, first install the model analyzer:
```bash
pip3 install triton-model-analyzer
```

Once the model analyzer installs successfully, enter the following command (modifying the instance
count to something lower for your GPU, if necessary):
```bash
model-analyzer profile -m /opt/tritonserver/model_repository/ --profile-models falcon7b --run-config-search-max-instance-count=3 --run-config-search-min-model-batch-size=8
```
This tool will take longer to execute than the performance analyzer example (~40 minutes).
If this execution time is too long, you can also run the analyzer with the
`--run-config-search-mode quick` option. In our experimentation, enabling the quick search option
yielded fewer results but took half the time. Regardless, once the model analyzer is complete,
it will provide you a full summary relating to throughput, latency, and hardware utilization
in multiple formats. A snippet from the summary report produced by the model analyzer for
our run is ranked by performance and shown below:

| Model Config Name | Max Batch Size | Dynamic Batching | Total Instance Count | p99 Latency (ms) | Throughput (infer/sec) | Max GPU Memory Usage (MB) | Average GPU Utilization (%) |
| :---: | :----: | :---: | :----: | :---: | :----:   | :---: | :---: |
| falcon7b_config_7 | 16 | Enabled | 3:GPU | 1412.581 | 71.944 | 46226 | 100.0 |
| falcon7b_config_8 | 32 | Enabled | 3:GPU | 2836.225 | 63.9652 | 46268 | 100.0 |
| falcon7b_config_4 | 16 | Enabled | 2:GPU | 7601.437 | 63.9454 | 31331 | 100.0 |
| falcon7b_config_default | 8 | Enabled | 1:GPU | 4151.873 | 63.9384 | 16449 | 89.3 |

We can examine the performance of any of these configurations with more granularity by viewing
their detailed reports. This subset of reports focuses on a single configuration's latency
and concurrency metrics as they relate to throughput and hardware utilization. A snippet from
the top performing configuration for our tests is shown below (abridged for brevity):

| Request Concurrency | p99 Latency (ms) | Client Response Wait (ms) | Server Queue (ms) | Server Compute Input (ms) | Server Compute Infer (ms) | Throughput (infer/sec) | Max GPU Memory Usage (MB) | Average GPU Utilization (%) |
| :---: | :----: | :---: | :----: | :---: | :----:   | :---: | :---: | :---: |
| 512	| 8689.491 | 8190.506 | 7397.975 | 0.166 | 778.565 | 63.954 | 46230.667264 | 100.0 |
| | | | | ... | | | | |
| 128 | 2289.118 | 2049.37 | 1277.34 | 0.159 | 770.771 | 61.2953 | 46230.667264 | 100.0 |
| 64 | 1412.581 | 896.924 | 227.108 | 0.157 | 667.757 | 71.944 | 46226.47296 | 100.0 |
| 32 | 781.362 | 546.35 | 86.078 | 0.103 | 459.257 | 57.7877 | 46226.47296 | 100.0 |
| | | | | ... | | | | |
| 1 | 67.12 | 49.707 | 0.049 | 0.024 | 49.121 | 20.0993 | 46207.598592 | 54.9 |

Similarly, this is a single use case for the model analyzer. For more information and a more complete list
of model analyzer parameters and run options, please see [this](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html) guide.

*Please note that both the performance and model analyzer experiments were conducted
on a system with an Intel i9 and NVIDIA A6000 GPU. Your results may vary depending on
you hardware.*

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
