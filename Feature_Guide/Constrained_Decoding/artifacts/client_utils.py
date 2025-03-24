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

import queue
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def run_inference(
    triton_client,
    prompt,
    output_len,
    request_id,
    repetition_penalty,
    presence_penalty,
    frequency_penalty,
    temperature,
    stop_words,
    bad_words,
    embedding_bias_words,
    embedding_bias_weights,
    model_name,
    streaming,
    beam_width,
    overwrite_output_text,
    return_context_logits_data,
    return_generation_logits_data,
    end_id,
    pad_id,
    verbose,
    num_draft_tokens=0,
    use_draft_logits=None,
    logits_post_processor_name=None,
):
    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.int32) * output_len
    streaming_data = np.array([[streaming]], dtype=bool)
    beam_width_data = np.array([[beam_width]], dtype=np.int32)
    temperature_data = np.array([[temperature]], dtype=np.float32)

    inputs = [
        prepare_tensor("text_input", input0_data),
        prepare_tensor("max_tokens", output0_len),
        prepare_tensor("stream", streaming_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
    ]

    if num_draft_tokens > 0:
        inputs.append(
            prepare_tensor(
                "num_draft_tokens", np.array([[num_draft_tokens]], dtype=np.int32)
            )
        )
    if use_draft_logits is not None:
        inputs.append(
            prepare_tensor(
                "use_draft_logits", np.array([[use_draft_logits]], dtype=bool)
            )
        )

    if bad_words:
        bad_words_list = np.array([bad_words], dtype=object)
        inputs += [prepare_tensor("bad_words", bad_words_list)]

    if stop_words:
        stop_words_list = np.array([stop_words], dtype=object)
        inputs += [prepare_tensor("stop_words", stop_words_list)]

    if repetition_penalty is not None:
        repetition_penalty = [[repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty, dtype=np.float32)
        inputs += [prepare_tensor("repetition_penalty", repetition_penalty_data)]

    if presence_penalty is not None:
        presence_penalty = [[presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)
        inputs += [prepare_tensor("presence_penalty", presence_penalty_data)]

    if frequency_penalty is not None:
        frequency_penalty = [[frequency_penalty]]
        frequency_penalty_data = np.array(frequency_penalty, dtype=np.float32)
        inputs += [prepare_tensor("frequency_penalty", frequency_penalty_data)]

    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits", return_context_logits_data),
        ]

    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits", return_generation_logits_data),
        ]

    if (embedding_bias_words is not None and embedding_bias_weights is None) or (
        embedding_bias_words is None and embedding_bias_weights is not None
    ):
        assert 0, "Both embedding bias words and weights must be specified"

    if embedding_bias_words is not None and embedding_bias_weights is not None:
        assert len(embedding_bias_words) == len(
            embedding_bias_weights
        ), "Embedding bias weights and words must have same length"
        embedding_bias_words_data = np.array([embedding_bias_words], dtype=object)
        embedding_bias_weights_data = np.array(
            [embedding_bias_weights], dtype=np.float32
        )
        inputs.append(prepare_tensor("embedding_bias_words", embedding_bias_words_data))
        inputs.append(
            prepare_tensor("embedding_bias_weights", embedding_bias_weights_data)
        )
    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.int32)
        inputs += [prepare_tensor("end_id", end_id_data)]

    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
        inputs += [prepare_tensor("pad_id", pad_id_data)]

    if logits_post_processor_name is not None:
        logits_post_processor_name_data = np.array(
            [[logits_post_processor_name]], dtype=object
        )
        inputs += [
            prepare_tensor(
                "logits_post_processor_name", logits_post_processor_name_data
            )
        ]

    user_data = UserData()
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))
    # Send request
    triton_client.async_stream_infer(model_name, inputs, request_id=request_id)

    # Wait for server to close the stream
    triton_client.stop_stream()

    # Parse the responses
    output_text = ""
    while True:
        try:
            result = user_data._completed_requests.get(block=False)
        except Exception:
            break

        if type(result) == InferenceServerException:
            print("Received an error from server:")
            print(result)
        else:
            output = result.as_numpy("text_output")
            if streaming and beam_width == 1:
                new_output = output[0].decode("utf-8")
                if overwrite_output_text:
                    output_text = new_output
                else:
                    output_text += new_output
            else:
                output_text = output[0].decode("utf-8")
                if verbose:
                    print(output, flush=True)

            if return_context_logits_data is not None:
                context_logits = result.as_numpy("context_logits")
                if verbose:
                    print(f"context_logits.shape: {context_logits.shape}")
                    print(f"context_logits: {context_logits}")
            if return_generation_logits_data is not None:
                generation_logits = result.as_numpy("generation_logits")
                if verbose:
                    print(f"generation_logits.shape: {generation_logits.shape}")
                    print(f"generation_logits: {generation_logits}")

    if streaming and beam_width == 1:
        if verbose:
            print(output_text)

    return output_text
