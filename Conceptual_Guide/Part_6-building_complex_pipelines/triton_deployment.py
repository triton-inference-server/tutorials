import os
import time

import numpy as np
import requests
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve
import json

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(route_prefix="/", ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):

        self._triton_server = tritonserver.Server(
            model_repository=["/workspace/models"]
        )
        self._triton_server.start(blocking=True)

        self._models = self._triton_server.models
        for (name, version), model in self._models.items():
            print(model)
            print(model.batch_properties())
            print(model.transaction_properties())
            print(model.config())

    @app.get("/generate")
    def generate(self, prompt: str) -> str:
        text_obj = np.array([prompt], dtype="object").reshape((-1, 1))

        model = self._triton_server.models["pipeline"]

        responses = model.infer(inputs={"prompt":text_obj})

        generated_image = list(responses)[0].outputs["generated_image"]

        im = Image.fromarray(np.squeeze(generated_image.astype(np.uint8)))
        im.save("generated_image2.jpg")
        return ""


if __name__ == "__main__":
    # 2: Deploy the deployment.
    serve.run(TritonDeployment.bind())

    print(
        requests.get(
            "http://localhost:8000/generate", params={"prompt": "Theodore"}
        ).json()
    )


else:
    triton_app = TritonDeployment.bind()
