import tritonserver
import numpy
import requests
from io import BytesIO

from PIL import Image
import starlette.requests
import torch
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from ray import serve

from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import CLIPTextModel, CLIPTokenizer

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, resnet50_model_handle: DeploymentHandle) -> None:
        self.handle = resnet50_model_handle

    @app.get(
        "/classify",
    )
    async def classify(
            self,prompt:str) -> str:
        max_ = await self.handle.classify.remote()
        return f"max: {max_}"


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=5,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 1,
        "min_replicas": 0,
        "initial_replicas": 1,
        "max_replicas": 1,
    },
)
class Resnet50Triton:
    def __init__(self):
        self._triton_server = tritonserver.Server(model_repository="/workspace/resnet50-models").start()

    def classify(self) -> int:
      model = self._triton_server.models()["resnet50"]
      input_ = numpy.random.rand(10, 3, 224, 224).astype(numpy.float32)
  
   #   responses = model.infer(inputs={"INPUT__0": input_})
      responses = model.infer(inputs={"input.1": input_})
      for response in responses:
          #output_ = numpy.from_dlpack(response.outputs["OUTPUT__0"].to_host())
          output_ = numpy.from_dlpack(response.outputs["495"].to_host())
          max_ = numpy.argmax(output_[0][0])
          return int(max_)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=5,
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 1,
        "min_replicas": 0,
        "initial_replicas": 1,
        "max_replicas": 1,
    },
)
class Resnet50:
    def __init__(self):
        self.resnet50 = (
            models.resnet50(weights=ResNet50_Weights.DEFAULT).eval().to("cuda")
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        resp = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        self.categories = resp.content.decode("utf-8").split("\n")
        

    def classify(self) -> str:
        input_tensor = torch.Tensor(numpy.random.rand(10, 3, 224, 224).astype(numpy.float32)).to("cuda")
        
        with torch.no_grad():
            output = self.resnet50(input_tensor)
            sm_output = torch.nn.functional.softmax(output[0][0], dim=0)
        ind = torch.argmax(sm_output)
        return 0

def entrypoint_native(args):
    return APIIngress.bind(Resnet50.bind())
    

def entrypoint_triton(args):
    return APIIngress.bind(Resnet50Triton.bind())
