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
import uuid

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
    def __init__(self, llm_model_handle:DeploymentHandle) -> None:
        #self.diffusion_handle = diffusion_model_handle
        self.llm_handle = llm_model_handle
    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate_image(
        self, prompt: str, img_size: int = 512, filename: Optional[str] = None
    ) -> None:
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.diffusion_handle.generate.remote(prompt, img_size=img_size)

        if filename:
            image.save(filename)

    @app.get(
        "/generate")
    async def generate(
            self, prompt: str, max_tokens:int=100 ) -> str:
        assert len(prompt), "prompt parameter cannot be empty"
        
        text = await self.llm_handle.generate.remote(prompt, max_tokens)

        return text


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=10,
    autoscaling_config={"min_replicas": 1,
                        "target_num_ongoing_requests_per_replica":2,
                        "max_replicas": 8,
                        "upscale_delay_s":0,
                        "downscale_delay_s":20},
)
class LLama_2_7b:
    def __init__(self):
        self._triton_server = tritonserver

        model_repository = [
            "/workspace/trt-llm-models",
        ]

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
            backend_configuration={"python":{
                "shm-region-prefix-name":str(uuid.uuid4())}}
        )
        self._triton_server.start(wait_until_ready=True)

        if not self._triton_server.model("llama-2-7b").ready():
            try:
                self._llm_model = self._triton_server.load("llama-2-7b")
                self._triton_server.load("preprocessing")
                self._triton_server.load("postprocessing")
                self._triton_server.load("tensorrt_llm")
                if not self._llm_model.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print("Error can't llm diffusion model!")
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
            self, prompt, max_tokens=100) -> str:
        
        for response in self._llm_model.infer(inputs={
                "text_input": [[prompt]],
                "max_tokens":
                numpy.array([[max_tokens]]).astype(numpy.int32)}):
            try:
                text = response.outputs["text_output"].to_string_array()
            except:
                text = response.outputs["text_output"].to_bytes_array()
            print(text)
            return str(text)
    

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class StableDiffusionV1_4:
    def __init__(self):
        self._triton_server = tritonserver

        if S3_BUCKET_URL is not None:
            model_repository = S3_BUCKET_URL
        else:
            model_repository = [
                "/workspace/diffuser-models",
            ]

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
            backend_configuration={"python":{"shm-region-prefix-name":str(uuid.uuid4())}}
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
        self, prompt: str, img_size: int = 512) -> None:
        for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
            generated_image = (
                numpy.from_dlpack(response.outputs["generated_image"])
                .squeeze()
                .astype(numpy.uint8)
            )
            image = Image.fromarray(generated_image)

            return image
        

if __name__ == "__main__":
    # 2: Deploy the deployment.
    serve.run(APIIngress.bind(StableDiffusionV1_4.bind(), LLama_2_7b.bind()), route_prefix="/")

    # 3: Query the deployment and print the result.
    print(
        requests.get(
            "http://localhost:8000/imagine",
            params={"prompt": "pigeon in new york, realistic, 4k, photograph"},
        )
    )

        # 3: Query the deployment and print the result.
    print(
        requests.get(
            "http://localhost:8000/generate",
            params={"prompt": "pigeon in new york"},
        )
    )


entrypoint = APIIngress.bind(LLama_2_7b.bind())
