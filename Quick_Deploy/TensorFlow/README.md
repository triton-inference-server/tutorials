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


# Deploying a TensorFlow Model

This README showcases how to deploy a simple ResNet model on Triton Inference Server.

## Step 1: Export the model

Export a TensorFlow model as a saved model.

```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's Tensorflow
# container; eg. 22.04

docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:<xx.xx>-tf2-py3
python export.py
```

## Step 2: Set Up Triton Inference Server

To use Triton, we need to build a model repository. The structure of the repository as follows:
```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.savedmodel
            |
            +-- saved_model.pb
            +-- variables
                |
                +-- variables.data-00000-of-00001
                +-- variables.index
```

A sample model configuration of the model is included with this demo as `config.pbtxt`. If you are new to Triton, it is highly recommended to [review Part 1](../../Conceptual_Guide/Part_1-model_deployment/README.md) of the conceptual guide.
```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models --backend-config=tensorflow,version=2
```

## Step 3: Using a Triton Client to Query the Server

Install dependencies & download an example image to test inference.

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk bash
pip install --upgrade tensorflow
pip install image

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```
Building a client requires three basic points. Firstly, we setup a connection with the Triton Inference Server.
```
triton_client = httpclient.InferenceServerClient(url="localhost:8000")
```
Secondly, we specify the names of the input and output layer(s) of our model.
```
inputs = httpclient.InferInput("input_1", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

output = httpclient.InferRequestedOutput("predictions", binary_data=True, class_count=1000)
```
Lastly, we send an inference request to the Triton Inference Server.
```
# Querying the server
results = triton_client.infer(model_name="resnet50", inputs=[inputs], outputs=[output])
predictions = results.as_numpy('predictions')
print(predictions)
```
The output of the same should look like below:
```
[b'0.301167:90' b'0.169790:14' b'0.161309:92' b'0.093105:94'
 b'0.058743:136' b'0.050185:11' b'0.033802:91' b'0.011760:88'
 b'0.008309:989' b'0.004927:95' b'0.004905:13' b'0.004095:317'
 b'0.004006:96' b'0.003694:12' b'0.003526:42' b'0.003390:313'
 ...
 b'0.000001:751' b'0.000001:685' b'0.000001:408' b'0.000001:116'
 b'0.000001:627' b'0.000001:933' b'0.000000:661' b'0.000000:148']
```
The output format here is `<confidence_score>:<classification_index>`. To learn how to map these to the label names and more, refer to our [documentation](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md). The client code above is available in `client.py`.