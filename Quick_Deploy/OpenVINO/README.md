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

# Deploying ONNX, PyTorch and TensorFlow Models with the OpenVINO Backend

This README demonstrates how to deploy simple ONNX, PyTorch and TensorFlow models on Triton Inference Server using the OpenVINO backend. 


## Deploying an ONNX Model
### 1. Build the model repository and download the ONNX model.
```
mkdir -p model_repository/densenet_onnx/1 
wget -O model_repository/densenet_onnx/1/model.onnx \
https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
```

### 2. Create a new file named `config.pbtxt`
```
name: "densenet_onnx" 
backend: "openvino" 
default_model_filename: "model.onnx" 
```

### 3. Place the `config.pbtxt` file in the model repository, the structure should look as follows:
```
model_repository 
| 
+-- densenet_onnx 
    | 
    +-- config.pbtxt 
    +-- 1 
        | 
        +-- model.onnx
```
 
Note: This directory structure is how the Triton Inference Server can read the configuration and model files and must follow the required layout. Do not place any other folders or files in the model repository other than the needed model files.

### 4. Run the Triton Inference Server
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models 
```

### 5. Download the Triton Client code `client.py` from GitHub to a place you want to run the Triton Client from.
```
wget https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Quick_Deploy/ONNX/client.py 
```

### 6. Run the Triton Client in the same location as the `client.py` file, install dependencies, and query the server
```
docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.04-py3-sdk bash 
```
```
pip install torchvision 
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg" 
python3 client.py 
```

### 7. Output
```
['11.549026:92' '11.232335:14' '7.528014:95' '6.923391:17' '6.576575:88'] 
```

## Deploying a PyTorch Model
### 1. Download and prepare the PyTorch model.
PyTorch models (.pt) will need to be converted to OpenVINO format. Create a `downloadAndConvert.py` file to download the PyTorch model and use the OpenVINO Model Converter to save a `model.xml` and `model.bin`:
```
import torchvision 
import torch 
import openvino as ov 
model = torchvision.models.resnet50(weights='DEFAULT') 
ov_model = ov.convert_model(model) 
ov.save_model(ov_model, 'model.xml')
```

Install the dependencies:
```
pip install openvino 
pip install torchvision
```

Run `downloadAndConvert.py`
```
python3 downloadAndConvert.py 
```

To convert your own PyTorch model, refer to [Converting a PyTorch Model](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html)

### 2. Create a new file named `config.pbtxt`
```
name: "resnet50 " 
backend: "openvino" 
max_batch_size : 0 
input [ 
  { 
    name: "x" 
    data_type: TYPE_FP32 
    dims: [ 3, 224, 224 ] 
    reshape { shape: [ 1, 3, 224, 224 ] } 
  } 
] 
output [ 
  { 
    name: "x.45" 
    data_type: TYPE_FP32 
    dims: [ 1, 1000 ,1, 1] 
    reshape { shape: [ 1, 1000 ] } 
  } 
]
```

3. Place the config.pbtxt file in the model repository as well as the model.xml and model.bin, the folder structure should look as follows:
```
model_repository 
| 
+-- resnet50 
    | 
    +-- config.pbtxt 
    +-- 1 
        | 
        +-- model.xml 
        +-- model.bin 
```

Note: This directory structure is how the Triton Inference Server can read the configuration and model files and must follow the required layout. Do not place any other folders or files in the model repository other than the needed model files.

### 4. Run the Triton Inference Server
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models 
```

### 5. In another terminal, download the Triton Client code `client.py` from GitHub to the place you want to run the Triton Client from.
```
wget https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Quick_Deploy/PyTorch/client.py 
```

In the `client.py` file, you’ll need to update the model input and output names to match those expected by the backend as the model is slightly different from the one in the Triton tutorial. For example, change the original input name used in the PyTorch model (input__0) to the name used by the OpenVINO backend (x).

| Old Value   | New Value |
| :-------------: | :-------------: |
| input__0 |x |
| output__0 |x.45 |

### 6. Run the Triton Client in the same location as the `client.py` file, install dependencies, and query the server.
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.04-py3-sdk bash 
```
```
pip install torchvision 
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg" 
python3 client.py 
```

### 7. Output
```
[b'6.354599:14' b'4.292510:92' b'3.886345:90' b'3.333909:136' b'3.096908:15']
```

Note: OpenVINO also has an integration into TorchServe, which enables serving PyTorch models without conversion to OpenVINO IR format. See code samples here.

## Deploying a TensorFlow Model
### 1. Download and prepare the TensorFlow model.
Export the TensorFlow model in SavedModel format:
```
docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:24.04-tf2-py3
```
```
python3 export.py 
```

The model will need to be converted to OpenVINO format. Create a `convert.py` file to use the OpenVINO Model Converter to save a `model.xml` and `model.bin`:
```
import openvino as ov 
ov_model = ov.convert_model(' path_to_saved_model_dir’) 
ov.save_model(ov_model, 'model.xml')
```

Install the dependencies:
```
pip install openvino 
```

Run `convert.py`
```
python3 convert.py 
```

To convert your TensorFlow model, refer to Converting a TensorFlow Model

### 2. Create a new file named `config.pbtxt`
```pbtxt
name: "resnet50" 
backend: "openvino" 
max_batch_size : 0 
input [ 
  { 
    name: "input_1" 
    data_type: TYPE_FP32 
    dims: [-1, 224, 224, 3 ] 
  } 
] 
output [ 
  { 
    name: "predictions" 
    data_type: TYPE_FP32 
    dims: [-1, 1000] 
  } 
]
```

### 3. Place the `config.pbtxt` file in the model repository as well as the `model.xml` and `model.bin`, the structure should look as follows:
```
model_repository 
| 
+-- resnet50 
    | 
    +-- config.pbtxt 
    +-- 1 
        | 
        +-- model.xml 
        +-- model.bin 
```

Note: This directory structure is how the Triton Inference Server can read the configuration and model files and must follow the required layout. Do not place any other folders or files in the model repository other than the needed model files.

### 4. Run the Triton Inference Server
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models 
```

### 5. In another terminal, download the Triton Client code `client.py` from GitHub to the place you want to run the Triton Client from.
```
wget https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Quick_Deploy/TensorFlow/client.py 
```

### 6. Run the Triton Client in the same location as the `client.py` file, install dependencies, and query the server.
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.04-py3-sdk bash 
```
```
pip install --upgrade tensorflow 
pip install image 
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg" 
python3 client.py 
```

### 7. Output
```
[b'0.301167:90' b'0.169790:14' b'0.161309:92' b'0.093105:94' 

 b'0.058743:136' b'0.050185:11' b'0.033802:91' b'0.011760:88' 

 b'0.008309:989' b'0.004927:95' b'0.004905:13' b'0.004095:317' 

 b'0.004006:96' b'0.003694:12' b'0.003526:42' b'0.003390:313' 

 ... 

 b'0.000001:751' b'0.000001:685' b'0.000001:408' b'0.000001:116' 

 b'0.000001:627' b'0.000001:933' b'0.000000:661' b'0.000000:148']
```
