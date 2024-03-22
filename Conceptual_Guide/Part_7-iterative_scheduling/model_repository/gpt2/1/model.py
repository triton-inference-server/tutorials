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
import triton_python_backend_utils as pb_utils
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        # Load the GPT-2 model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.state = {}

    @staticmethod
    def auto_complete_config(config):
        inputs = [
            {
                "name": "input",
                "data_type": "TYPE_STRING",
                "dims": [1],
            }
        ]
        outputs = [{"name": "output", "data_type": "TYPE_STRING", "dims": [1]}]

        for input in inputs:
            config.add_input(input)
        for output in outputs:
            config.add_output(output)

        return config

    def execute(self, requests):
        for request in requests:
            input_tensor = str(pb_utils.get_input_tensor_by_name(
                request, "input"
            ).as_numpy()[0], encoding="utf-8")
            correlation_id = pb_utils.get_input_tensor_by_name(
                request, "correlation_id"
            ).as_numpy()[0]
            start = pb_utils.get_input_tensor_by_name(
                request, "start").as_numpy()[0]
            if start:
                self.state[correlation_id] = [
                    self.tokenizer.encode(input_tensor, return_tensors="pt")
                ]

            response_sender = request.get_response_sender()
            state = self.state[correlation_id]
            outputs = self.model.generate(torch.cat(state, dim=1), max_new_tokens=1)
            generated_token = outputs[0][-1].reshape(1, 1)
            self.state[correlation_id].append(generated_token)
            if generated_token.item() == self.tokenizer.eos_token_id:
                del self.state[correlation_id]
                flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            else:
                request.set_release_flags(pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE)
                flags = 0

            output_decoded = self.tokenizer.decode(generated_token.item())
            response = pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("output", np.array([output_decoded], dtype=np.object_))])
            response_sender.send(response, flags=flags)
