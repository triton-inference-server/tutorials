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

import os
from pprint import pprint
from typing import Optional

import numpy
import requests
import torch
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

S3_BUCKET_URL = None

if "S3_BUCKET_URL" in os.environ:
    S3_BUCKET_URL = os.environ["S3_BUCKET_URL"]


def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class BaseDeployment:
    def __init__(self):
        self._image_size = 512
        self._model_id = "runwayml/stable-diffusion-v1-5"
        from diffusers import StableDiffusionPipeline

        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self._model_id, revision="fp16", torch_dtype=torch.float16
        )
        self._pipeline = self._pipeline.to("cuda")

    @app.get("/generate")
    def generate(self, prompt: str, filename: Optional[str] = None) -> None:
        with torch.autocast("cuda"):
            image_ = self._pipeline(
                prompt,
                height=self._image_size,
                width=self._image_size,
                num_inference_steps=50,
            ).images[0]
            if filename:
                image_.save(filename)


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        self._triton_server = tritonserver

        if S3_BUCKET_URL is not None:
            model_repository = S3_BUCKET_URL
        else:
            model_repository = [
                "/workspace/identity-models",
                "/workspace/diffusion-models",
            ]

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())
        self._stable_diffusion = None
        self._test_model = None

        if not self._triton_server.model("stable_diffusion_1_5").ready():
            try:
                self._stable_diffusion = self._triton_server.load(
                    "stable_diffusion_1_5"
                )

                if not self._stable_diffusion.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print("Error can't load stable diffusion model!")
                print(
                    f"Please ensure dependencies are met and you have set the environment variable HF_TOKEN {error}"
                )
                return
        _print_heading("Models")
        pprint(self._triton_server.models())

    @app.get("/identity")
    def test(self, string_input: str) -> str:
        if not self._triton_server.model("identity").ready():
            self._test_model = self._triton_server.load("identity")

        output = []
        for response in self._test_model.infer(
            inputs={"string_input": [[string_input]]}
        ):
            output.append(response.outputs["string_output"].to_string_array()[0][0])

        return "".join(output)

    @app.get("/generate")
    def generate(self, prompt: str, filename: Optional[str] = None) -> None:
        for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
            generated_image = (
                numpy.from_dlpack(response.outputs["generated_image"])
                .squeeze()
                .astype(numpy.uint8)
            )

            image_ = Image.fromarray(generated_image)
            if filename:
                image_.save(filename)


def tritonserver_deployment(_args):
    return TritonDeployment.bind()


def base_deployment(_args):
    return BaseDeployment.bind()


if __name__ == "__main__":
    # 2: Deploy the deployment.
    serve.run(TritonDeployment.bind(), route_prefix="/")

    # 3: Query the deployment and print the result.
    print(
        requests.get(
            "http://localhost:8000/identity", params={"name": "Theodore"}
        ).json()
    )

    # 3: Query the deployment and print the result.
    print(
        requests.get(
            "http://localhost:8000/generate",
            params={"prompt": "pigeon in new york, realistic, 4k, photograph"},
        )
    )
