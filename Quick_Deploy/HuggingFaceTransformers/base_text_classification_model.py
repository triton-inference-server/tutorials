# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
import numpy as np
import triton_python_backend_utils as pb_utils

import torch
import transformers


class TritonPythonModel:
    @classmethod
    def auto_complete_config(cls, cfg):
        inputs = [{"name": "prompt", "data_type": "TYPE_STRING", "dims": [1]}]
        outputs = [
            {"name": "label", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "score", "data_type": "TYPE_FP32", "dims": [-1]},
        ]

        [cfg.add_input(i) for i in inputs]
        [cfg.add_output(o) for o in outputs]
        cfg.set_max_batch_size(0)
        return cfg

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
        default_hf_model = "facebook/opt-125m"
        # Check for user-specified model name in model config parameters
        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", default_hf_model
        )

        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")
        self.pipeline = transformers.pipeline(
            "text-classification",
            model=hf_model,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            self.logger.log_info(f"Request parameters: {request.parameters()}")
            # Assume input named "prompt" for now, specified in autocomplete above
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt = prompt_tensor.as_numpy()[0].decode("utf-8")

            self.logger.log_info(f"Conducting sentiment analysis of prompt: {prompt}")
            response = self.generate(prompt)
            responses.append(response)
        return responses

    def generate(self, prompt):
        analysis = self.pipeline(prompt)
        labels = []
        scores = []
        for index in range(0, len(analysis)):
            labels.append(analysis[index]["label"])
            scores.append(analysis[index]["score"])

        out_tensor_0 = pb_utils.Tensor("label", np.array(labels, dtype=np.object_))
        out_tensor_1 = pb_utils.Tensor("score", np.array(scores, dtype=np.float32))
        response = pb_utils.InferenceResponse(
            output_tensors=[out_tensor_0, out_tensor_1]
        )
        return response

    def finalize(self):
        print("Cleaning up...")
