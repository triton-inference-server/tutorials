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
    autoscaling_config={"min_replicas": 1, "max_replicas": 1},
)
class StableDiffusionV1_4:
    def __init__(self):
        from diffusers import (
            EulerDiscreteScheduler,
            LMSDiscreteScheduler,
            StableDiffusionPipeline,
        )

        model_id = "CompVis/stable-diffusion-v1-4"

        # scheduler = EulerDiscreteScheduler.from_pretrained(
        #   model_id, subfolder="scheduler"
        # )

        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image


entrypoint = APIIngress.bind(StableDiffusionV1_4.bind())
