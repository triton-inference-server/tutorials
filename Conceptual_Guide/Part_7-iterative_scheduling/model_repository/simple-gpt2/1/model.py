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

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class State:
    def __init__(self):
        self.prompt_tokens_len = 0
        self.tokens = []
        self.max_tokens = 0
        self.ignore_eos = False


class TritonPythonModel:
    def initialize(self, args):
        self.state = {}
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"

        # Load the GPT-2 model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def auto_complete_config(config):
        inputs = [
            {
                "name": "text_input",
                "data_type": "TYPE_STRING",
                "dims": [1],
            }
        ]
        outputs = [{"name": "text_output", "data_type": "TYPE_STRING", "dims": [1]}]

        for input in inputs:
            config.add_input(input)
        for output in outputs:
            config.add_output(output)

        transaction_policy = {"decoupled": True}
        config.set_dynamic_batching()
        config.set_max_batch_size(8)
        config.set_model_transaction_policy(transaction_policy)

        return config

    def init_state(self, states, requests):
        """
        Initializes the state for each request.

        Args:
            states (list): A list to store the state for each request.
            requests (list): A list of requests.

        Returns:
            None
        """
        for request in requests:
            input_tensor = str(
                pb_utils.get_input_tensor_by_name(request, "text_input")
                .as_numpy()
                .item(),
                encoding="utf-8",
            )
            state = State()

            parameters = json.loads(request.parameters())
            state.ignore_eos = parameters["ignore_eos"]
            state.max_tokens = parameters["max_tokens"]
            state.tokens = self.tokenizer(
                input_tensor, return_tensors="pt", padding=True
            )["input_ids"][0].to(self.device)
            state.prompt_tokens_len = len(state.tokens)

            states.append(state)

    def create_batch(self, states):
        # Find the max sequence length
        max_len = max([len(x.tokens) for x in states])

        # Pad the input tensors.
        input_ids_torch = torch.cat(
            [
                torch.cat(
                    [
                        torch.tensor(
                            [self.tokenizer.eos_token_id] * (max_len - len(x.tokens)),
                            device=self.device,
                        )
                    ]
                    + [x.tokens]
                ).unsqueeze(0)
                for x in states
            ]
        )
        attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.tensor([0] * (max_len - x.tokens.numel())),
                        torch.tensor([1] * x.tokens.numel()),
                    ]
                ).unsqueeze(0)
                for x in states
            ]
        )
        return input_ids_torch.long(), attention_mask.long().to(self.device)

    def send_responses(self, states, requests, outputs):
        """
        Sends responses to the requests based on the model outputs.

        Args:
            states (list): A list of states for each request.
            requests (list): A list of requests.
            outputs (torch.Tensor): A tensor containing the model outputs.

        Returns:
            list: A list of requests that have not been completed.
            list: A list of states for each request that have not been completed.

        """
        updated_request_list = []
        updated_states = []

        for i, request in enumerate(requests):
            response_sender = request.get_response_sender()
            # Convert scalar to a one dimensional tensor
            generated_token = outputs[i][-1].reshape(1)

            max_tokens = states[i].max_tokens + states[i].prompt_tokens_len
            states[i].tokens = torch.cat([states[i].tokens, generated_token])

            if (
                generated_token.item() == self.tokenizer.eos_token_id
                and not states[i].ignore_eos
            ) or len(states[i].tokens) >= max_tokens:
                flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            else:
                flags = 0
                updated_states.append(states[i])
                updated_request_list.append(request)

            output_decoded = self.tokenizer.decode(generated_token.cpu().item())
            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "text_output", np.array([output_decoded], dtype=np.object_)
                    )
                ]
            )
            response_sender.send(response, flags=flags)
        return updated_request_list, updated_states

    def execute(self, requests):
        pb_utils.Logger.log_verbose(f"Processing {len(requests)} request(s).")

        states = []
        self.init_state(states, requests)
        while requests:
            input_ids, attention_mask = self.create_batch(states)

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention_mask,
            )
            requests, states = self.send_responses(states, requests, outputs)
