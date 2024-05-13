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
import json
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import triton_python_backend_utils as pb_utils


class State:
    def __init__(self):
        self.prompt_tokens_len = 0
        self.max_tokens = 0
        self.streamed_tokens = 0


class TritonPythonModel:
    def initialize(self, args):
        self.state = {}
        self.model_path = str(Path(args["model_repository"]) / args["model_version"])
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def create_batch(self, requests, states):
        """
        Create a batch of input data for the model.

        Args:
            requests (list): A list of Triton requests.
            states (list): A list of State objects representing the state of each request.

        Returns:
            og.GeneratorParams: A generator parameters object for the model.
        """
        generator_params = og.GeneratorParams(self.model)

        input_ids = []
        for request in requests:
            input_tensor = str(
                pb_utils.get_input_tensor_by_name(request, "text_input")
                .as_numpy()
                .item(),
                encoding="utf-8",
            )
            state = State()
            tokens = self.tokenizer.encode(input_tensor)
            state.prompt_tokens_len = len(tokens)

            # Store the parameters
            parameters = json.loads(request.parameters())
            state.max_tokens = parameters["max_tokens"]

            states.append(state)
            input_ids.append(tokens)

        # Find the max sequence length
        max_len = max([len(x) for x in input_ids])
        input_ids = [
            [generator_params.pad_token_id] * (max_len - len(x)) + x for x in input_ids
        ]
        generator_params.input_ids = np.asarray(input_ids)

        return generator_params

    def send_responses(self, requests, outputs, states, generator_params):
        """
        Sends responses to the requests based on the model outputs and updates the state of each request.

        Args:
            requests (list): A list of Triton requests.
            outputs (list): A list of generated tokens from the model.
            states (list): A list of State objects representing the state of each request.
            generator_params (og.GeneratorParams): A generator parameters object for the model.

        Returns:
            tuple: A tuple containing two lists:
                   - remaining_requests: Requests that need further processing.
                   - remainings_states: States corresponding to the remaining requests.
        """
        remaining_requests = []
        remainings_states = []
        for i, request in enumerate(requests):
            response_sender = request.get_response_sender()
            generated_token = outputs[i]
            max_tokens = states[i].max_tokens

            if (generated_token == generator_params.eos_token_id) or states[
                i
            ].streamed_tokens >= max_tokens - 1:
                flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                request.set_release_flags(pb_utils.TRITONSERVER_REQUEST_RELEASE_ALL)
            else:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                flags = 0
                remaining_requests.append(request)
                remainings_states.append(states[i])
                states[i].streamed_tokens += 1

            output_decoded = self.tokenizer.decode(generated_token)
            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "text_output", np.array([output_decoded], dtype=np.object_)
                    )
                ]
            )
            response_sender.send(response, flags=flags)
        return remaining_requests, remainings_states

    def execute(self, requests):
        states = []
        generator_params = self.create_batch(requests, states)

        # Create a generator object
        generator = og.Generator(self.model, generator_params)

        # Generate tokens until all requests are processed
        while requests:
            generator.compute_logits()
            generator.generate_next_token()
            outputs = generator.get_next_tokens()
            requests, states = self.send_responses(
                requests, outputs, states, generator_params
            )
