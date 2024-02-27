# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import os
import sys

from cuda import cudart

file_location = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(file_location, "Diffusion"))

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from Diffusion.stable_diffusion_pipeline import StableDiffusionPipeline
from Diffusion.utilities import (
    PIPELINE_TYPE,
    TRT_LOGGER,
    add_arguments,
    process_pipeline_args,
)
from tqdm.auto import tqdm

onnx_opset = 18
opt_batch_size = 4
opt_image_height = 512
opt_image_width = 512
seed = 10


class TritonPythonModel:
    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "generated_image"
            )["data_type"]
        )
        self.pipeline = StableDiffusionPipeline(
            pipeline_type=PIPELINE_TYPE.TXT2IMG,
            max_batch_size=opt_batch_size,
            use_cuda_graph=True,
        )

        model_directory = os.path.join(args["model_repository"], args["model_version"])
        engine_dir = os.path.join(
            model_directory, f"engine-batch-size-{opt_batch_size}"
        )
        framework_model_dir = os.path.join(model_directory, "pytorch_model")
        onnx_dir = os.path.join(model_directory, "onnx")
        self.pipeline.loadEngines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            onnx_opset=onnx_opset,
            opt_batch_size=opt_batch_size,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            static_batch=True,
        )
        _, shared_device_memory = cudart.cudaMalloc(
            self.pipeline.calculateMaxDeviceMemory()
        )
        self.pipeline.activateEngines(shared_device_memory)
        self.pipeline.loadResources(
            opt_image_height, opt_image_width, opt_batch_size, seed=seed
        )
        self.image_height = opt_image_height
        self.image_width = opt_image_width
        self.batch_size = opt_batch_size

    def finalize(self):
        self.pipeline.teardown()

    def execute(self, requests):
        responses = []
        prompts = []
        negative_prompts = []
        prompts_per_request = []
        image_results = []
        self.pipeline.seed = 5
        for request in requests:
            prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, "prompt"
            ).as_numpy()

            for prompt in prompt_tensor:
                prompts.append(prompt[0].decode())

            negative_prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, "negative_prompt"
            )

            if not negative_prompt_tensor:
                negative_prompts.extend([""] * len(prompt_tensor))
            else:
                negative_prompt_tensor = negative_prompt_tensor.as_numpy()
                for negative_prompt in negative_prompt_tensor:
                    negative_prompts.append(negative_prompt[0].decode())
            prompts_per_request.append(len(prompt_tensor))
        num_requests = len(requests)
        num_prompts = len(prompts)
        remainder = self.batch_size - (num_prompts % self.batch_size)
        if remainder < self.batch_size:
            prompts.extend([""] * remainder)
            negative_prompts.extend([""] * remainder)
        num_prompts = len(prompts)

        for batch in range(0, num_prompts, self.batch_size):
            (images, walltime_ms) = self.pipeline.infer(
                prompts[batch : batch + self.batch_size],
                negative_prompts[batch : batch + self.batch_size],
                self.image_height,
                self.image_width,
                save_image=False,
            )
            images = (
                ((images + 1) * 255 / 2)
                .clamp(0, 255)
                .detach()
                .permute(0, 2, 3, 1)
                .round()
                .type(torch.uint8)
                .cpu()
                .numpy()
            )
            image_results.extend(images)

        result_index = 0
        for num_prompts_in_request in prompts_per_request:
            generated_images = []
            for image_result in image_results[result_index:num_prompts_in_request]:
                generated_images.append(image_result)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(generated_images, dtype=self.output_dtype),
                    )
                ]
            )
            responses.append(inference_response)

        return responses
