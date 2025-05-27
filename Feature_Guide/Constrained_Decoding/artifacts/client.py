#!/usr/bin/python
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

import argparse
import sys

import client_utils
import numpy as np
import tritonclient.grpc as grpcclient
from pydantic import BaseModel


class AnswerFormat(BaseModel):
    title: str
    year: int
    director: str
    producer: str
    plot: str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="Inference server URL."
    )

    parser.add_argument("-p", "--prompt", type=str, required=True, help="Input prompt.")

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="ensemble",
        choices=["ensemble", "tensorrt_llm_bls"],
        help="Name of the Triton model to send request to",
    )

    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        required=False,
        default=None,
        help="The repetition penalty value",
    )

    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
    )

    parser.add_argument(
        "--frequency-penalty",
        type=float,
        required=False,
        default=None,
        help="The frequency penalty value",
    )

    parser.add_argument(
        "-o",
        "--output-len",
        type=int,
        default=100,
        required=False,
        help="Specify output length",
    )

    parser.add_argument(
        "--request-id",
        type=str,
        default="",
        required=False,
        help="The request_id for the stop request",
    )

    parser.add_argument("--stop-words", nargs="+", default=[], help="The stop words")

    parser.add_argument("--bad-words", nargs="+", default=[], help="The bad words")

    parser.add_argument(
        "--embedding-bias-words", nargs="+", default=[], help="The biased words"
    )

    parser.add_argument(
        "--embedding-bias-weights",
        nargs="+",
        default=[],
        help="The biased words weights",
    )

    parser.add_argument(
        "--overwrite-output-text",
        action="store_true",
        required=False,
        default=False,
        help="In streaming mode, overwrite previously received output text instead of appending to it",
    )

    parser.add_argument(
        "--return-context-logits",
        action="store_true",
        required=False,
        default=False,
        help="Return context logits, the engine must be built with gather_context_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--return-generation-logits",
        action="store_true",
        required=False,
        default=False,
        help="Return generation logits, the engine must be built with gather_ generation_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--end-id", type=int, required=False, help="The token id for end token."
    )

    parser.add_argument(
        "--pad-id", type=int, required=False, help="The token id for pad token."
    )

    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        required=False,
        default=False,
        help="Enhance text input with system prompt.",
    )

    parser.add_argument(
        "--use-schema",
        action="store_true",
        required=False,
        default=False,
        help="Use client-defined JSON schema.",
    )

    parser.add_argument(
        "--logits-post-processor-name",
        type=str,
        required=False,
        default=None,
        help="Logits Post-Processor to use for output generation.",
    )

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8001"

    embedding_bias_words = (
        FLAGS.embedding_bias_words if FLAGS.embedding_bias_words else None
    )
    embedding_bias_weights = (
        FLAGS.embedding_bias_weights if FLAGS.embedding_bias_weights else None
    )

    try:
        client = grpcclient.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    return_context_logits_data = None
    if FLAGS.return_context_logits:
        return_context_logits_data = np.array(
            [[FLAGS.return_context_logits]], dtype=bool
        )

    return_generation_logits_data = None
    if FLAGS.return_generation_logits:
        return_generation_logits_data = np.array(
            [[FLAGS.return_generation_logits]], dtype=bool
        )

    prompt = FLAGS.prompt

    if FLAGS.use_system_prompt:
        prompt = (
            "<|im_start|>system\n You are a helpful assistant that answers in JSON."
        )

        if FLAGS.use_schema:
            prompt += "Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>".format(
                schema=AnswerFormat.model_json_schema()
            )

        prompt += "<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n".format(
            user_prompt=FLAGS.prompt
        )

    output_text = client_utils.run_inference(
        client,
        prompt,
        FLAGS.output_len,
        FLAGS.request_id,
        FLAGS.repetition_penalty,
        FLAGS.presence_penalty,
        FLAGS.frequency_penalty,
        FLAGS.temperature,
        FLAGS.stop_words,
        FLAGS.bad_words,
        embedding_bias_words,
        embedding_bias_weights,
        FLAGS.model_name,
        FLAGS.streaming,
        FLAGS.beam_width,
        FLAGS.overwrite_output_text,
        return_context_logits_data,
        return_generation_logits_data,
        FLAGS.end_id,
        FLAGS.pad_id,
        FLAGS.verbose,
        logits_post_processor_name=FLAGS.logits_post_processor_name,
    )

    print(output_text)
