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

import base64
import os
from io import BytesIO

import numpy as np
import requests as rq
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        HF_LOCATION = os.getenv("HF_LOCATION", pb_utils.get_model_dir())
        self.image_processor = AutoProcessor.from_pretrained(HF_LOCATION)
        self.logger = pb_utils.Logger
        self.tokenizer = AutoTokenizer.from_pretrained(HF_LOCATION)
        self.vocab_size = 32064
        self.max_input_len = 2048

    def _tokenize(self, prompt, num_visual_tokens):
        chunks = prompt.split("<image>")
        assert len(chunks) == 2, "Only support exactly one image per prompt"

        return (
            self.tokenizer.encode(chunks[0])
            + list(range(self.vocab_size, self.vocab_size + num_visual_tokens))
            + self.tokenizer.encode(chunks[1])[self.tokenizer.add_bos_token :]
        )

    def _parse_input(self, request, input_name, default=None):
        input = pb_utils.get_input_tensor_by_name(request, input_name)
        if input is not None:
            return input.as_numpy()[0]

        return default

    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        for request in requests:
            # Get INPUT0
            image = (
                pb_utils.get_input_tensor_by_name(request, "image")
                .as_numpy()
                .flatten()
                .tolist()
            )
            if isinstance(image[0], bytes):
                image = image[0].decode("utf-8")
            pil_image = Image.open(rq.get(image, stream=True).raw).convert("RGB")
            # Get INPUT1
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

            image = self.image_processor(
                text=prompt, images=pil_image, return_tensors="np"
            )["pixel_values"].astype(np.float16)
            # Create inference request object
            infer_request = pb_utils.InferenceRequest(
                model_name="vision_encoder",
                requested_output_names=["features"],
                inputs=[pb_utils.Tensor("image", image)],
            )

            # Perform synchronous blocking inference request
            vision_response = infer_request.exec()
            response_sender = request.get_response_sender()

            image_features = pb_utils.get_output_tensor_by_name(
                vision_response, "features"
            )
            image_features = torch.from_dlpack(image_features.as_numpy())
            # parse input parameters
            max_tokens = self._parse_input(request, "max_tokens", default=50)
            temperature = self._parse_input(request, "temperature", default=0.5)
            top_k = self._parse_input(request, "top_k", default=1)
            frequency_penalty = self._parse_input(
                request, "frequency_penalty", default=0.7
            )
            seed = self._parse_input(request, "seed", default=10)

            input_ids = self._tokenize(prompt, len(image_features[0]))
            input_ids = np.array(input_ids, dtype=np.int32)
            input_len = input_ids.shape[0]
            if input_len > self.max_input_len:
                error = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"Input length ({input_len:d}) exceeds limit ({self.max_input_len:d})"
                    )
                )
                response_sender.send(
                    error, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                return

            # build embedding table
            embedding_args = {
                "prompt_vocab_size": np.array(
                    [[image_features[0].shape[0]]], dtype=np.uint32
                ),
                "prompt_embedding_table": np.expand_dims(image_features[0], 0).astype(
                    np.float16
                ),
            }

            llm_request_inputs = {
                "input_ids": np.expand_dims(input_ids, 0),
                "input_lengths": np.array([[input_len]], dtype=np.int32),
                "request_output_len": np.array([[max_tokens]], dtype=np.int32),
                "temperature": np.array([[temperature]], dtype=np.float32),
                "runtime_top_k": np.array([[top_k]], dtype=np.int32),
                "frequency_penalty": np.array([[frequency_penalty]], dtype=np.float32),
                "end_id": np.array([[self.tokenizer.eos_token_id]], dtype=np.int32),
                "random_seed": np.array([[seed]], dtype=np.uint64),
                "streaming": np.array([[1]], dtype=np.bool_),
                **embedding_args,
            }
            llm_request = pb_utils.InferenceRequest(
                model_name="tensorrt_llm",
                requested_output_names=["output_ids", "sequence_length"],
                inputs=[pb_utils.Tensor(k, v) for k, v in llm_request_inputs.items()],
            )
            output_ids, output_len = [], 0
            for response in llm_request.exec(decoupled=True):
                if response.has_error():
                    raise pb_utils.TritonModelException(response.error().message())

                stream_output_ids = (
                    pb_utils.get_output_tensor_by_name(response, "output_ids")
                    .as_numpy()
                    .flatten()
                    .tolist()
                )

                finish_reason = ""
                if len(stream_output_ids) == 0 or (
                    len(stream_output_ids) != 0
                    and stream_output_ids[-1] == self.tokenizer.eos_token_id
                ):
                    finish_reason = "stop"
                output_ids += stream_output_ids
                if len(output_ids) >= max_tokens:
                    finish_reason = "length"
                    output_ids = output_ids[:max_tokens]
                last_response = finish_reason != ""
                output_len = len(output_ids)

                if last_response:
                    output_text = self.tokenizer.decode(output_ids).strip()
                    response = pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "text", np.array([output_text], np.object_)
                            ),
                            pb_utils.Tensor(
                                "finish_reason", np.array([finish_reason], np.object_)
                            ),
                            pb_utils.Tensor(
                                "completion_tokens", np.array([output_len], np.int32)
                            ),
                            pb_utils.Tensor(
                                "prompt_tokens", np.array([input_len], np.int32)
                            ),
                            pb_utils.Tensor(
                                "total_tokens",
                                np.array([output_len + input_len], np.int32),
                            ),
                        ]
                    )
                    response_sender.send(
                        response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )

        return None
