# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import shutil
import sys

import numpy as np
import torch
from cuda import cudart

file_location = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(file_location, "Diffusion"))

import triton_python_backend_utils as pb_utils
from Diffusion.stable_diffusion_pipeline import StableDiffusionPipeline
from Diffusion.utilities import PIPELINE_TYPE


class TritonPythonModel:
    _KNOWN_VERSIONS = {"1.5": PIPELINE_TYPE.TXT2IMG, "xl-1.0": PIPELINE_TYPE.XL_BASE}

    def _set_defaults(self):
        self._batch_size = 1
        self._onnx_opset = 18
        self._image_height = 512
        self._image_width = 512
        self._seed = None
        self._version = "1.5"
        self._scheduler = None
        self._steps = 30
        self._force_engine_build = False

    def _set_from_parameter(self, parameter, parameters, class_):
        value = parameters.get(parameter, None)
        if value is not None:
            value = value["string_value"]
            if value:
                setattr(self, "_" + parameter, class_(value))

    def _set_from_config(self, model_config):
        model_config = json.loads(model_config)
        self._batch_size = int(model_config.get("max_batch_size", 1))
        if self._batch_size < 1:
            self._batch_size = 1

        config_parameters = model_config.get("parameters", {})

        if config_parameters:
            parameter_type_map = {
                "onnx_opset": int,
                "image_height": int,
                "image_width": int,
                "steps": int,
                "seed": int,
                "scheduler": str,
                "guidance_scale": float,
                "version": str,
                "force_engine_build": bool,
            }

            for parameter, parameter_type in parameter_type_map.items():
                self._set_from_parameter(parameter, config_parameters, parameter_type)

    def initialize(self, args):
        self._set_defaults()
        self._set_from_config(args["model_config"])

        if self._version not in TritonPythonModel._KNOWN_VERSIONS:
            raise Exception(
                f"Invalid Stable Diffusion Version: {self._version}, choices: {list(TritonPythonModel._KNOWN_VERSIONS.keys())}"
            )

        self._model_instance_device_id = int(args["model_instance_device_id"])

        self._pipeline = StableDiffusionPipeline(
            pipeline_type=TritonPythonModel._KNOWN_VERSIONS[self._version],
            max_batch_size=self._batch_size,
            use_cuda_graph=True,
            version=self._version,
            denoising_steps=self._steps,
        )

        model_directory = os.path.join(args["model_repository"], args["model_version"])
        engine_dir = os.path.join(
            model_directory, f"{self._version}-engine-batch-size-{self._batch_size}"
        )
        framework_model_dir = os.path.join(
            model_directory, f"{self._version}-pytorch_model"
        )
        onnx_dir = os.path.join(model_directory, f"{self._version}-onnx")

        if self._force_engine_build:
            shutil.rmtree(engine_dir, ignore_errors=True)
            shutil.rmtree(framework_model_dir, ignore_errors=True)
            shutil.rmtree(onnx_dir, ignore_errors=True)

        if self._model_instance_device_id != 0:
            raise Exception("Only device id 0 is currently supported")

        self._pipeline.loadEngines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            onnx_opset=self._onnx_opset,
            opt_batch_size=self._batch_size,
            opt_image_height=self._image_height,
            opt_image_width=self._image_width,
            static_batch=True,
        )
        _, shared_device_memory = cudart.cudaMalloc(
            self._pipeline.calculateMaxDeviceMemory()
        )
        self._pipeline.activateEngines(shared_device_memory)
        self._pipeline.loadResources(
            self._image_height, self._image_width, self._batch_size, seed=self._seed
        )
        self._logger = pb_utils.Logger

    def finalize(self):
        self._pipeline.teardown()

    def execute(self, requests):
        responses = []
        prompts = []
        negative_prompts = []
        prompts_per_request = []
        image_results = []
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
        remainder = self._batch_size - (num_prompts % self._batch_size)
        self._logger.log_info(f"Client Requests in Batch:{num_requests}")
        self._logger.log_info(f"Prompts in Batch:{num_prompts}")
        if remainder < self._batch_size:
            prompts.extend([""] * remainder)
            negative_prompts.extend([""] * remainder)
        num_prompts = len(prompts)
        for batch in range(0, num_prompts, self._batch_size):
            (images, walltime_ms) = self._pipeline.infer(
                prompts[batch : batch + self._batch_size],
                negative_prompts[batch : batch + self._batch_size],
                self._image_height,
                self._image_width,
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
            for image_result in image_results[
                result_index : result_index + num_prompts_in_request
            ]:
                generated_images.append(image_result)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(generated_images, dtype=np.uint8),
                    )
                ]
            )
            responses.append(inference_response)
            result_index += num_prompts_in_request

        return responses
