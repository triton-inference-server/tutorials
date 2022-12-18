# Deploying a PyTorch Model

This README showcases how to deploy a simple ResNet model on Triton Inference Server.

## Step 1: Export the model

Save the PyTorch model.

```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's PyTorch 
# container; eg. 22.04

docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:<xx.xx>-py3
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
        +-- model.pt
```

A sample model configuration of the model is included with this demo as `config.pbtxt`. If you are new to Triton, it is highly recommended to [review Part 1](../../Conceptual_Guide/Part_1-model_deployment/README.md) of the conceptual guide.
```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## Step 3: Using a Triton Client to Query the Server

Install dependencies & download an example image to test inference.

```
pip install torchvision
pip install attrdict
pip install nvidia-pyindex
pip install tritonclient[all]

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```
Building a client requires three basic points. Firstly, we setup a connection with the Triton Inference Server.
```
client = httpclient.InferenceServerClient(url="localhost:8000")
```
Secondly, we specify the names of the input and output layer(s) of our model.
```
inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=1000)
```
Lastly, we send an inference request to the Triton Inference Server.
```
# Querying the server
results = client.infer(model_name="resnet50", inputs=[inputs], outputs=[outputs])
predictions = results.as_numpy('output__0')
print(predictions[:5])
```
The output of the same should look like below:
```
[b'12.468750:90' b'11.523438:92' b'9.664062:14' b'8.429688:136'
 b'8.234375:11']
```
The output format here is `<confidence_score>:<classification_index>`. To learn how to map these to the label names and more, refer to our [documentation](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md). The client code above is available in `client.py`. 
