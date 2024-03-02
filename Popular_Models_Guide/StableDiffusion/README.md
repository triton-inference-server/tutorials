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

# Deploying Stable Diffusion Models with Triton and TensorRT

This example demonstrates how to deploy Stable Diffusion models in
Triton by leveraging the [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/9.2/demo/Diffusion)
pipeline and utilities.

Using the TensorRT demo as a base this example contains a reusable
[python based backend](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md), [`/backend/diffusion/model.py`](backend/diffusion/model.py),
suitable for deploying multiple versions and configurations of
Diffusion models.

For more information on Stable Diffusion please visit
[stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5),
[stable-diffusion-xl](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl). For
more information on the TensorRT implementation please see the [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/9.2/demo/Diffusion).

> [!Note]
> This example is given as sample code and should be reviewed before use in production settings.

| [Requirements](#requirements) | [Building Server Image](#building-the-triton-inference-server-image) | [Stable Diffusion v1.5](#building-and-running-stable-diffusion-v-15) | [Stable Diffusion XL](#building-and-running-stable-diffusion-xl) | [Sending an Inference Request](#sending-an-inference-request) | [Model Configuration](docs/model_configuration.md) | [Sample Client](#sample-client) | [Known Issues and Limitations](#known-issues-and-limitations) |

## Requirements

The following instructions require a Linux system with Docker
installed. For CUDA support, make sure your CUDA driver meets the
requirements in the "NVIDIA Driver" section of [Deep Learning Framework
support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Building the Triton Inference Server Image

The example is designed based on the
`nvcr.io/nvidia/tritonserver:24.01-py3` docker image and [TensorRT OSS v9.2.0](https://github.com/NVIDIA/TensorRT/releases/tag/v9.2.0).

A set of convenience scripts are provided to create a docker image
based on the `nvcr.io/nvidia/tritonserver:24.01-py3` image with the
dependencies for the TensorRT Stable Diffusion demo installed.

### Triton Inference Server + TensorRT OSS

#### Clone Repository
```bash
git clone https://github.com/triton-inference-server/tutorials.git --single-branch
cd tutorials/Popular_Models_Guide/StableDiffusion
```

#### Build Tritonserver Diffusion Docker Image
```bash
./build.sh
```

#### Included Models

The `default` build includes model configuration files located in the
`/diffusion-models` folder. Example configurations are provided for
[`stable_diffusion_1_5`](diffusion-models/stable_diffustion_1_5) and
[`stable_diffusion_xl`](diffusion-models/stable_diffustion_xl).

Model artifacts and engine files are not included in the image but are
built into a volume mounted directory as a separate step.

## Building and Running Stable Diffusion v 1.5

### Start Tritonserver Diffusion Container

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
./run.sh
```

### Build Stable Diffusion v 1.5 Engine

```bash
./scripts/build_models.sh --model stable_diffusion_1_5
```

#### Expected Output
```
 diffusion-models
|-- stable_diffusion_1_5
|   |-- 1
|   |   |-- 1.5-engine-batch-size-1
|   |   |-- 1.5-onnx
|   |   |-- 1.5-pytorch_model
|   `-- config.pbtxt

```

### Start a Server Instance

> [!Note]
> We use `EXPLICIT` model control mode for demonstration purposes to
> control which stable diffusion version is loaded.  For production
> deployments please refer to [Secure Deployment Considerations][secure_guide]
> for more information on the risks associated with `EXPLICIT` mode.

[secure_guide]: https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/deploy.md

```bash
tritonserver --model-repository diffusion-models --model-control-mode explicit --load-model stable_diffusion_1_5
```

#### Expected Output
```
<SNIP>
I0229 20:15:52.125050 749 server.cc:676]
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| stable_diffusion_1_5 | 1       | READY  |
+----------------------+---------+--------+

<SNIP>
```

## Building and Running Stable Diffusion XL

### Start Tritonserver Diffusion Container

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
./run.sh
```

### Build Stable Diffusion XL Engine

```bash
./scripts/build_models.sh --model stable_diffusion_xl
```

#### Expected Output
```
 diffusion-models
 |-- stable_diffusion_xl
    |-- 1
    |   |-- xl-1.0-engine-batch-size-1
    |   |-- xl-1.0-onnx
    |   `-- xl-1.0-pytorch_model
    `-- config.pbtxt
```

### Start a Server Instance

> [!Note]
> We use `EXPLICIT` model control mode for demonstration purposes to
> control which stable diffusion version is loaded.  For production
> deployments please refer to [Secure Deployment Considerations][secure_guide]
> for more information on the risks associated with `EXPLICIT` mode.


```bash
tritonserver --model-repository diffusion-models --model-control-mode explicit --load-model stable_diffusion_xl
```

#### Expected Output
```
<SNIP>
I0229 20:22:22.912465 1440 server.cc:676]
+---------------------+---------+--------+
| Model               | Version | Status |
+---------------------+---------+--------+
| stable_diffusion_xl | 1       | READY  |
+---------------------+---------+--------+

<SNIP>
```

## Sending an Inference Request

We've provided a sample [client](client.py) application to make
sending and receiving requests simpler.

### Start Tritonserver Diffusion Container

In a separate terminal from the server start a new container.

The following command starts a container and volume mounts the current
directory as `workspace`.

```bash
./run.sh
```


### Send Prompt to Stable Diffusion 1.5

```bash
python3 client.py --model stable_diffusion_1_5 --prompt "butterfly in new york, 4k, realistic" --save-image
```

#### Example Output

```bash
Client: 0 Throughput: 0.7201335361144658 Avg. Latency: 1.3677194118499756
Throughput: 0.7163933558221957 Total Time: 1.395881175994873
```

If `--save-image` is given then output images will be saved as jpegs.

`
 client_0_generated_image_0.jpg
`

![sample_generated_image](./docs/client_0_generated_image_0_1_5.jpg)


### Send Prompt to Stable Diffusion XL

```bash
python3 client.py --model stable_diffusion_xl --prompt "butterfly in new york, 4k, realistic" --save-image
```

#### Example Output

```bash
Client: 0 Throughput: 0.1825067711674996 Avg. Latency: 5.465569257736206
Throughput: 0.18224859609447058 Total Time: 5.487010717391968
```

If `--save-image` is given then output images will be saved as jpegs.

`
 client_0_generated_image_0.jpg
`

![sample_generated_image](./docs/client_0_generated_image_0_xl.jpg)


## Sample Client

The sample [client](client.py) application enables users to quickly
test the diffusion models under different concurrency scenarios. For a
full list and description of the client application's options use:

```
python3 client.py --help
```

### Sending Concurrent Requests

To increase load and concurrency users can use the `clients` and
`requests` options to control the number of client processes and the
number of requests sent by each client.

#### Example: Ten Clients Sending Ten Requests Each

The following command enables ten clients each sending ten
requests. Each client is an independent process that sends its
requests one after the other in parallel with the other nine clients.

```bash
python3 client.py --model stable_diffusion_xl --requests 10 --clients 10
```

## Known Issues and Limitations

1. When shutting down the server, an invalid memory operation occurs:

   > [!Note]
   > This error is also seen in standalone applications outside of the Triton Inference Server
   > and we believe this is due to an interaction between imported python modules. Further
   > we haven't seen any issues related to this error and believe it can be safely
   > ignored.


   ```
   free(): invalid pointer
   ```


2. The diffusion backend doesn't yet support using an optional refiner
   model unlike the [demo][demo_reference] it's based on. See also
   [demo_txt2img_xl.py][demo_code]


[demo_code]: https://github.com/NVIDIA/TensorRT/blob/release/9.2/demo/Diffusion/demo_txt2img_xl.py


[demo_reference]: https://github.com/NVIDIA/TensorRT/tree/release/9.2/demo/Diffusion#text-to-image-using-sdxl-stable-diffusion-xl

