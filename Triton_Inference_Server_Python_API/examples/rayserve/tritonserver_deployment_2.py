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
import tritonserver
from fastapi import FastAPI
from fastapi.responses import Response
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

S3_BUCKET_URL = None

if "S3_BUCKET_URL" in os.environ:
    S3_BUCKET_URL = os.environ["S3_BUCKET_URL"]


def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(
        self, prompt: str, img_size: int = 512, filename: Optional[str] = None
    ) -> None:
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)
        #        file_stream = BytesIO()
        if filename:
            image.save(filename)


#        return Response(content=file_stream.getvalue(), media_type="image/png")


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
)
class StableDiffusionV1_4:
    def __init__(self):
        self._triton_server = tritonserver

        if S3_BUCKET_URL is not None:
            model_repository = S3_BUCKET_URL
        else:
            model_repository = [
                "/workspace/identity-models",
                "/workspace/diffuser-models",
            ]

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

        if not self._triton_server.model("stable_diffusion").ready():
            try:
                self._triton_server.load("text_encoder")
                self._triton_server.load("vae")

                self._stable_diffusion = self._triton_server.load("stable_diffusion")
                if not self._stable_diffusion.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print("Error can't load stable diffusion model!")
                print(
                    f"Please ensure dependencies are met and you have set the environment variable HF_TOKEN {error}"
                )
                return

        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())
        _print_heading("Models")
        pprint(self._triton_server.models())

    def generate(
        self, prompt: str, img_size: int = 512, filename: Optional[str] = None
    ) -> None:
        for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
            generated_image = (
                numpy.from_dlpack(response.outputs["generated_image"])
                .squeeze()
                .astype(numpy.uint8)
            )
            image_ = Image.fromarray(generated_image)
            if filename:
                image_.save(filename)


def triton_app(_args):
    return TritonDeployment.bind()


def entrypoint(_args):
    return TritonDeployment.bind()


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

entrypoint = APIIngress.bind(StableDiffusionV1_4.bind())
