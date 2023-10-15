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

# Deploy models using Triton

| Navigate to | [Part 2: Improving Resource Utilization](../Part_2-improving_resource_utilization/) | [Documentation: Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md) | [Documentation: Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md) |
| ------------ | --------------- | --------------- | --------------- |

Any deep learning inference serving solution needs to tackle two fundamental challenges:

* Managing multiple models.
* Versioning, loading, and unloading models.

## Before we begin

The conceptual guide aims to educate developers about the challenges faced whilst building inference infrastructure for deploying deep learning pipelines. `Part 1 - Part 5` of this guide build towards solving a simple problem: deploying a performant and scalable pipeline for transcribing text from images. This pipeline includes 5 steps:

1. Pre-process the raw image
1. Detect which parts of the image contain text (Text Detection Model)
1. Crop image to regions with text
1. Find text probabilities (Text Recognition Model)
1. Convert probabilities to actual text

In `Part 1`, we start by deploying both models on Triton with the pre/post processing steps done on the client.

## Deploying multiple models

The key challenge around managing multiple models is to build an infrastructure that can cater to the different requirements of different models. For instance, users may need to deploy a PyTorch model and TensorFlow model on the same server, and they have different loads for both the models, need to run them on different hardware devices, and need to independently manage the serving configurations (model queues, versions, caching, acceleration, and more). The Triton Inference Server caters to all of the above and more.

![multiple models](./img/multiple_models.PNG)

The first step in deploying models using the Triton Inference Server is building a repository that houses the models which will be served and the configuration schema. For the purposes of this demonstration, we will be making use of an [EAST](https://arxiv.org/pdf/1704.03155v2.pdf) model to detect text and a text recognition model. This workflow is largely an adaptation of [OpenCV's Text Detection](https://docs.opencv.org/4.x/db/da4/samples_2dnn_2text_detection_8cpp-example.html) samples.

To begin, let's clone the repository and navigate to this folder.

```bash
cd Conceptual_Guide/Part_1-model_deployment
```

Next, we'll be downloading the necessary models and making sure they are in a format that triton can deploy.

### Model 1: Text Detection

Download and unzip OpenCV's EAST model.

```bash
wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
tar -xvf frozen_east_text_detection.tar.gz
```

Export to ONNX.
>Note: The following step requires you to have the TensorFlow library installed. We recommend executing the following step within the NGC TensorFlow container environment, which you can launch with `docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:<yy.mm>-tf2-py3`

```bash
pip install -U tf2onnx
python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
```

### Model 2: Text Recognition

Download the Text Recognition model weights.

```bash
wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth
```

Export the models as `.onnx` using the file in the model definition file in the `utils` folder. This file is adapted from [Baek et. al. 2019](https://github.com/clovaai/deep-text-recognition-benchmark).

>Note:  The following python script requires you to have the PyTorch library installed. We recommend executing the following step within the NGC PyTorch container environment, which you can launch with `docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:<yy.mm>-py3`

```python
import torch
from utils.model import STRModel

# Create PyTorch Model Object
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# Load model weights from external file
state = torch.load("None-ResNet-None-CTC.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

# Create ONNX file by tracing model
trace_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, trace_input, "str.onnx", verbose=True)
```

### Setting up the model repository

A [model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) is Triton's way of reading your models and any associated metadata with each model (configurations, version files, etc.). These model repositories can live in a local or network attached filesystem, or in a cloud object store like AWS S3, Azure Blob Storage or Google Cloud Storage. For more details on model repository location, refer to [the documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-repository-locations). Servers can use also multiple different model repositories. For simplicity, this explanation only uses a single repository stored in the [local filesystem](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#local-file-system), in the following format:

```bash
# Example repository structure
<model-repository>/
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  ...
```

There are three important components to be discussed from the above structure:

* `model-name`: The identifying name for the model.
* `config.pbtxt`: For each model, users can define a model configuration. This configuration, at minimum, needs to define: the backend, name, shape, and datatype of model inputs and outputs. For most of the popular backends, this configuration file is autogenerated with defaults. The full specification of the configuration file can be found in the [`model_config` protobuf definition](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto).
* `version`: versioning makes multiple versions of the same model available for use depending on the policy selected. [More Information about versioning.](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-versions)

For this example you can set up the model repository structure in the following manner:

```bash
mkdir -p model_repository/text_detection/1
mv detection.onnx model_repository/text_detection/1/model.onnx

mkdir -p model_repository/text_recognition/1
mv str.onnx model_repository/text_recognition/1/model.onnx
```

These commands should give you a repository that looks this:

```bash
# Expected folder layout
model_repository/
├── text_detection
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── text_recognition
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

Note that, for this example, we've already created the `config.pbtxt` files and placed them in the necessary location. In the next section, we'll discuss the contents of these files.

### Model configuration

With the models and the file structure ready, the next things we need to look at are the `config.pbtxt` model configuration files. Let's first look at the model configuration for the `EAST text detection` model that's been provided for you at `/model_repository/text_detection/config.pbtxt`. This shows that `text_detection` is an ONNX model that has one `input` and two `output` tensors.

``` text proto
name: "text_detection"
backend: "onnxruntime"
max_batch_size : 256
input [
  {
    name: "input_images:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 3 ]
  }
]
output [
  {
    name: "feature_fusion/Conv_7/Sigmoid:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }
]
output [
  {
    name: "feature_fusion/concat_3:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 5 ]
  }
]
```

* `name`: "name" is an optional field, the value of which should match the name of the directory of the model.
* `backend`: This field indicates which backend is being used to run the model. Triton supports a wide variety of backends like TensorFlow, PyTorch, Python, ONNX and more. For a complete list of field selection refer to [these comments](https://github.com/triton-inference-server/backend#backends).
* `max_batch_size`: As the name implies, this field defines the maximum batch size that the model can support.
* `input` and `output`: The input and output sections specify the name, shape, datatype, and more, while providing operations like [reshaping](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#reshape) and support for [ragged batches](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/ragged_batching.md#ragged-batching).

In [most cases](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#auto-generated-model-configuration), it's possible to leave out the `input` and `output` sections and let Triton extract that information from the model files directly. Here, we've included them for clarity and because we'll need to know the names of our output tensors in the client application later on.

For details of all supported fields and their values, refer to the [model config protobuf definition file](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto).

### Launching the server

With our repository created and our models configured, we're ready to launch the server. While the Triton Inference Server can be [built from source](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-triton), the use of [pre-built Docker containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) freely available from NGC is highly recommended for this example.

```bash
# Replace the yy.mm in the image name with the release year and month
# of the Triton version needed, eg. 22.08

docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3
```

Once Triton Inference Server has been built or once inside the container, it can be launched with the command:

```bash
tritonserver --model-repository=/models
```

This will spin up the server and model instances will be ready for inference.

```text
I0712 16:37:18.246487 128 server.cc:626]
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| text_detection   | 1       | READY  |
| text_recognition | 1       | READY  |
+------------------+---------+--------+

I0712 16:37:18.267625 128 metrics.cc:650] Collecting metrics for GPU 0: NVIDIA GeForce RTX 3090
I0712 16:37:18.268041 128 tritonserver.cc:2159]
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                                                                                        |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                                                                                       |
| server_version                   | 2.23.0                                                                                                                                                                                       |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data statistics trace |
| model_repository_path[0]         | /models                                                                                                                                                                                      |
| model_control_mode               | MODE_NONE                                                                                                                                                                                    |
| strict_model_config              | 1                                                                                                                                                                                            |
| rate_limit                       | OFF                                                                                                                                                                                          |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                                                                                    |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                                                                                     |
| response_cache_byte_size         | 0                                                                                                                                                                                            |
| min_supported_compute_capability | 6.0                                                                                                                                                                                          |
| strict_readiness                 | 1                                                                                                                                                                                            |
| exit_timeout                     | 30                                                                                                                                                                                           |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

I0712 16:37:18.269464 128 grpc_server.cc:4587] Started GRPCInferenceService at 0.0.0.0:8001
I0712 16:37:18.269956 128 http_server.cc:3303] Started HTTPService at 0.0.0.0:8000
I0712 16:37:18.311686 128 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```

## Building a client application

Now that our Triton server has been launched, we can start sending messages to it. There are three ways to interact with the Triton Inference Server:

* HTTP(S) API
* gRPC API
* Native C API

There are also pre-built [client libraries](https://github.com/triton-inference-server/client#client-library-apis) in [C++](https://github.com/triton-inference-server/client/tree/main/src/c%2B%2B), [Python](https://github.com/triton-inference-server/client/tree/main/src/python), and [Java](https://github.com/triton-inference-server/client/tree/main/src/java) that wrap over the HTTP and gRPC APIs. This example contains a Python client script in `client.py` which uses the `tritonclient` python library to communicate with Triton over the HTTP API.

Let's examine the contents of this file:

* First, we import our HTTP client from the `tritonclient` library, as well as a few other libraries we'll use for processing our images:

  ```python
  import math
  import numpy as np
  import cv2
  import tritonclient.http as httpclient
  ```

* Next, we'll define a few helper functions for taking care of the pre and post processing steps for our pipeline. The details are omitted here for brevity, but you can check the `client.py` file for more details

  ```python
  def detection_preprocessing(image: cv2.Mat) -> np.ndarray:
    ...

  def detection_postprocessing(scores: np.ndarray, geometry: np.ndarray, preprocessed_image: np.ndarray) -> np.ndarray:
    ...

  def recognition_postprocessing(scores: np.ndarray) -> str:
    ...
  ```

* Then, we create a client object, and initialize a connection with the Triton Inference Server.

  ```python
  client = httpclient.InferenceServerClient(url="localhost:8000")
  ```

* Now, we'll create the `InferInput` that we'll be sending to Triton from our data.

  ```python
  raw_image = cv2.imread("./img2.jpg")
  preprocessed_image = detection_preprocessing(raw_image)

  detection_input = httpclient.InferInput("input_images:0", preprocessed_image.shape, datatype="FP32")
  detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)
  ```

* Finally, we're ready to send an inference request to the Triton Inference Server and retrieve the response

  ```python
  detection_response = client.infer(model_name="text_detection", inputs=[detection_input])
  ```

* After that, we'll repeat the process with the text recognition model, performing our next processing step, creating the input object, querying the server and finally performing postprocessing and printing the result.

  ```python
  # Process responses from detection model
  scores = detection_response.as_numpy('feature_fusion/Conv_7/Sigmoid:0')
  geometry = detection_response.as_numpy('feature_fusion/concat_3:0')
  cropped_images = detection_postprocessing(scores, geometry, preprocessed_image)

  # Create input object for recognition model
  recognition_input = httpclient.InferInput("input.1", cropped_images.shape, datatype="FP32")
  recognition_input.set_data_from_numpy(cropped_images, binary_data=True)

  # Query the server
  recognition_response = client.infer(model_name="text_recognition", inputs=[recognition_input])

  # Process response from recognition model
  text = recognition_postprocessing(recognition_response.as_numpy('308'))

  print(text)
  ```

Let's try it out!

```bash
pip install tritonclient[http] opencv-python-headless
python client.py
```

You might have noticed that it's a bit redundant to retrieve the results of the first model only to do some processing and send them right back to Triton. In [Part 5](../Part_5-Model_Ensembles/) of this tutorial we explore how you can move more processing steps to the server and execute multiple models in a single network call.

## Model Versioning

The ability to deploy different versions of a model is essential to building an MLOps pipeline. The need arises from use cases like conducting A/B tests, easy model version rollbacks and more. Triton users can add a folder and the new model in the same repository:

```text
model_repository/
├── text_detection
│   ├── 1
│   │   └── model.onnx
│   ├── 2
│   │   └── model.onnx
│   └── config.pbtxt
└── text_recognition
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

By default Triton serves the "latest" model, but the policy to serve different versions of the model is customizable. For more information, [refer this guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#version-policy).

## Loading & Unloading Models

Triton has model management API that can be used to control the model loading unloading policies. This API is extremely useful in cases where one or more models need to be loaded or unloaded without interrupting inference for other models being served on the same server. Users can select from one of three control modes:

* NONE
* EXPLICIT
* POLL

```bash
tritonserver --model-repository=/models --model-control-mode=poll
```

The policies can also be set via command line arguments whilst launching the server. For more information, refer [this section](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md#model-management) of the documentation.

# What's next?

In this tutorial, we covered the very basics of setting up and querying a Triton Inference Server. This is Part 1 of a 6 part tutorial series that covers the challenges faced in deploying Deep Learning models to production. [Part 2](../Part_2-improving_resource_utilization/) covers `Concurrent Model Execution and Dynamic Batching`. Depending on your workload and experience you might want to jump to [Part 5](../Part_5-Model_Ensembles/) which covers `Building an Ensemble Pipeline with multiple models, pre and post processing steps, and adding business logic`.
