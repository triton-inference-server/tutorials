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
import argparse
from cuda import cudart

sys.path.append(os.path.join(os.getcwd(), "backend/diffusion"))
sys.path.append(os.path.join(os.getcwd(), "backend/diffusion/Diffusion"))

from Diffusion.stable_diffusion_pipeline import StableDiffusionPipeline
from Diffusion.utilities import PIPELINE_TYPE

model_configs = {
    "stable_diffusion_xl": {
        "max_batch_size": 1,
        "parameters": {
            "image_width": {"string_value": "1024"},
            "guidance_scale": {"string_value": "5.0"},
            "onnx_opset": {"string_value": "18"},
            "seed": {"string_value": ""},
            "steps": {"string_value": "30"},
            "force_engine_build": {"string_value": ""},
            "scheduler": {"string_value": ""},
            "version": {"string_value": "xl-1.0"},
            "image_height": {"string_value": "1024"},
        },
    },
    "stable_diffusion_1_5": {
        "max_batch_size": 1,
        "parameters": {
            "steps": {"string_value": "50"},
            "guidance_scale": {"string_value": "7.5"},
            "scheduler": {"string_value": ""},
            "force_engine_build": {"string_value": ""},
            "image_height": {"string_value": "512"},
            "seed": {"string_value": ""},
            "onnx_opset": {"string_value": "18"},
            "image_width": {"string_value": "512"},
            "version": {"string_value": "1.5"},
        },
    },
}


class BuildModel:
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
        self._set_from_config(model_configs[args.model_name])

        if self._version not in BuildModel._KNOWN_VERSIONS:
            raise Exception(
                f"Invalid Stable Diffusion Version: {self._version}, choices: {list(BuildModel._KNOWN_VERSIONS.keys())}"
            )

        self._model_instance_device_id = int(args.model_instance_device_id)

        self._pipeline = StableDiffusionPipeline(
            pipeline_type=BuildModel._KNOWN_VERSIONS[self._version],
            max_batch_size=self._batch_size,
            use_cuda_graph=True,
            version=self._version,
            denoising_steps=self._steps,
        )

        model_directory = os.path.join(
            args.model_repository, args.model_name, str(args.model_version)
        )
        os.makedirs(model_directory, exist_ok=True)

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

    def finalize(self):
        self._pipeline.teardown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "stable_diffusion_1_5",
            "stable_diffusion_xl",
        ],
        required=True,
        help="The model name to use. 'stable_diffusion_1_5' or 'stable_diffusion_xl'. Default is 'stable_diffusion_xl'.",
    )
    parser.add_argument(
        "--model_repository",
        type=str,
        default="diffusion-models",
        help="The model repository to use. Default is 'diffusion-models'.",
    )
    parser.add_argument(
        "--model_version",
        type=int,
        default=1,
        help="The model version to use. Default is 1.",
    )
    parser.add_argument(
        "--model_instance_device_id",
        type=int,
        default=0,
        help="The device ID of the model instance to use. Default is 0.",
    )

    args = parser.parse_args()
    build_model = BuildModel()
    try:
        build_model.initialize(args)
    finally:
        build_model.finalize
