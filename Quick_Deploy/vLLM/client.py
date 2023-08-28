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

import argparse
import asyncio
import json
import queue
import sys
import uuid

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def get_request_data(prompt, temperature, top_p):
    request_json = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "echo": True,
    }
    return np.array([json.dumps(request_json).encode("utf-8")], dtype=np.object_)


async def async_stream_yield(
    prompt_tokens, temperature, top_p, sequence_id, model_name
):
    count = 1
    for token in prompt_tokens:
        # Create the tensor for INPUT
        request_data = get_request_data(token, temperature, top_p)
        inputs = []
        inputs.append(
            grpcclient.InferInput(
                "serialized_request_json", request_data.shape, "BYTES"
            )
        )
        # Initialize the data
        inputs[0].set_data_from_numpy(request_data)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("serialized_response_json"))
        # Issue the asynchronous sequence inference.
        yield {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": "{}_{}".format(sequence_id, count),
            "sequence_id": sequence_id,
            "sequence_start": (count == 1),
            "sequence_end": (count == len(prompt_tokens)),
        }
        count = count + 1


async def main(FLAGS):
    model_name = "vllm"
    temperature = 0.8
    top_p = 0.95
    prompt_list = [
        ["Hello, my name is"],
        ["The president of the United States is"],
        ["The capital of France is"],
        ["The future of AI is"],
    ]

    async with grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose
    ) as triton_client:
        # Request iterator that yields the next request
        async def async_request_iterator():
            prompt_id = FLAGS.offset
            for prompt in prompt_list:
                prompt_id = prompt_id + 1
                async for request in async_stream_yield(
                    prompt, temperature, top_p, prompt_id, model_name
                ):
                    yield request

        try:
            # Start streaming
            response_iterator = triton_client.stream_infer(
                inputs_iterator=async_request_iterator(),
                stream_timeout=FLAGS.stream_timeout,
            )
            # Read response from the stream
            user_data = UserData()
            async for response in response_iterator:
                result, error = response
                if error:
                    user_data._completed_requests.put(error)
                else:
                    user_data._completed_requests.put(result)
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    # Retrieve results
    results = []
    results_dict = {}
    recv_count = 0
    while recv_count < len(prompt_list):
        data_item = user_data._completed_requests.get()
        if type(data_item) == InferenceServerException:
            print(data_item)
            sys.exit(1)
        else:
            response = data_item.as_numpy("serialized_response_json")
            results.append(json.loads(response[0]))
            if FLAGS.verbose:
                print(f"[VERBOSE RESPONSE]: {results[-1]}")
            prompt = results[-1]["prompt"]
            if prompt not in results_dict:
                results_dict[prompt] = []
            for completion in results[-1]["completions"]:
                results_dict[prompt].append(completion["text"])
        if results[-1]["finished"]:
            recv_count = recv_count + 1

    for prompt in results_dict.keys():
        print("===========")
        print(f"prompt => {prompt!r}")
        print("===========")
        print(f"response => {' '.join(results_dict[prompt])!r}")
        print("=========== \n")

    print("PASS: vLLM example")


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
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL and it gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "-o",
        "--offset",
        type=int,
        required=False,
        default=0,
        help="Add offset to sequence ID used",
    )
    FLAGS = parser.parse_args()
    asyncio.run(main(FLAGS))
