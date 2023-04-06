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
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "model_1_input_string")
            inp2 = pb_utils.get_input_tensor_by_name(request, "model_1_input_UINT8_array")
            inp3 = pb_utils.get_input_tensor_by_name(request, "model_1_input_INT8_array")
            inp4 = pb_utils.get_input_tensor_by_name(request, "model_1_input_FP32_image")
            inp5 = pb_utils.get_input_tensor_by_name(request, "model_1_input_bool")

            print("Model 1 received", flush=True)
            print(inp.as_numpy(), flush=True)
            print(inp2.as_numpy(), flush=True)
            print(inp3.as_numpy(), flush=True)
            print(inp4.as_numpy(), flush=True)
            print(inp5.as_numpy(), flush=True)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "model_1_output_string",
                    inp.as_numpy(),
                ),
                pb_utils.Tensor(
                    "model_1_output_UINT8_array",
                    inp2.as_numpy(),
                ),
                pb_utils.Tensor(
                    "model_1_output_INT8_array",
                    inp3.as_numpy(),
                ),
                pb_utils.Tensor(
                    "model_1_output_FP32_image",
                    inp4.as_numpy(),
                ),
                pb_utils.Tensor(
                    "model_1_output_bool",
                    inp5.as_numpy(),
                )
            ])
            responses.append(inference_response)
        return responses
