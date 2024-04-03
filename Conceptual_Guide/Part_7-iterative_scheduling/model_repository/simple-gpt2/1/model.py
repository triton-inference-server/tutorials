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
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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

        self.max_tokens = 64

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

    def init_state(self, requests):
        for i, request in enumerate(requests):
            input_tensor = str(
                pb_utils.get_input_tensor_by_name(request, "text_input")
                .as_numpy()
                .item(),
                encoding="utf-8",
            )
            if i not in self.state:
                self.state[i] = self.tokenizer(
                    input_tensor, return_tensors="pt", padding=True
                )["input_ids"][0].to(self.device)

    def create_batch(self, requests):
        """
        Gathers input tensors from the requests and returns processed input tensors.

        Args:
            requests (list): A list of requests containing input tensors.

        Returns:
            input_ids (torch.Tensor): A tensor containing the processed input IDs.
            attention_mask (torch.Tensor): A tensor containing the attention mask.
            mapping (list): A list of indices that map the input tensors to the requests.
        """

        input_ids = []
        mapping = []
        for index, state in self.state.items():
            if state != []:
                mapping.append(index)
                input_ids.append(state)

        if not mapping:
            return None, None, mapping

        # Find the max sequence length
        max_len = max([len(x) for x in input_ids])

        # Pad the input tensors.
        input_ids_torch = torch.cat(
            [
                torch.cat(
                    [
                        torch.tensor(
                            [self.tokenizer.eos_token_id] * (max_len - len(x)),
                            device=self.device,
                        )
                    ]
                    + [x]
                ).unsqueeze(0)
                for x in input_ids
            ]
        )
        attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.tensor([0] * (max_len - x.numel())),
                        torch.tensor([1] * x.numel()),
                    ]
                ).unsqueeze(0)
                for x in input_ids
            ]
        )
        return input_ids_torch.long(), attention_mask.long().to(self.device), mapping

    def send_responses(self, requests, outputs, mapping):
        """
        Scatter method for processing requests and sending responses.

        Args:
            requests (list): List of Triton InferenceRequest objects.
            outputs (list): List of output tensors generated by the model.
            mapping (list): List of indices that map the input tensors to the requests.

        Returns:
            None
        """
        for i in mapping:
            index = mapping[i]
            request = requests[index]

            response_sender = request.get_response_sender()
            # Convert scalar to a one dimensional tensor
            generated_token = outputs[index][-1].reshape(1)

            self.state[index] = torch.cat([self.state[index], generated_token])
            if (
                generated_token.item() == self.tokenizer.eos_token_id
                or len(self.state[index]) >= self.max_tokens
            ):
                flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                self.state[index] = []
            else:
                flags = 0

            output_decoded = self.tokenizer.decode(generated_token.cpu().item())
            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "text_output", np.array([output_decoded], dtype=np.object_)
                    )
                ]
            )
            response_sender.send(response, flags=flags)

    def execute(self, requests):
        pb_utils.Logger.log_verbose(f"Processing {len(requests)} request(s).")

        self.init_state(requests)
        mapping = [i for i in range(len(requests))]
        while mapping:
            input_ids, attention_mask, mapping = self.create_batch(requests)

            # Break if there are no other inflight requests.
            if not mapping:
                break

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention_mask,
            )
            self.send_responses(requests, outputs, mapping)

        # Clean up the state
        self.state = {}
