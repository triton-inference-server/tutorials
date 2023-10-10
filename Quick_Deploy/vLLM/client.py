import argparse
import asyncio
import queue
import sys
from os import system
import json

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *

class LLMClient:
    def __init__(self, flags: argparse.Namespace):
        self._client = grpcclient.InferenceServerClient(url=flags.url, verbose=flags.verbose)
        self._flags = flags
        self._loop = asyncio.get_event_loop()
        self._results_dict = {}

    async def async_request_iterator(self, prompts, sampling_parameters, model_name):
        try:
            for iter in range(self._flags.iterations):
                for i, prompt in enumerate(prompts):
                    prompt_id = self._flags.offset + (len(prompts) * iter) + i
                    self._results_dict[str(prompt_id)] = []
                    yield self.create_request(prompt, self._flags.streaming_mode, prompt_id, sampling_parameters, model_name)
        except Exception as error:
            print(f"Caught an error in the request iterator: {error}")

    async def stream_infer(self, prompts, sampling_parameters, model_name):
        try:
            response_iterator = self._client.stream_infer(
                inputs_iterator=self.async_request_iterator(prompts, sampling_parameters, model_name),
                stream_timeout=self._flags.stream_timeout,
            )
            async for response in response_iterator:
                yield response
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    async def process_stream(self, prompts, sampling_parameters, model_name):
        # Clear results in between process_stream calls
        self.results_dict = []
        
        async for response in self.stream_infer(prompts, sampling_parameters, model_name):
            result, error = response
            if error:
                print(f"Encountered error while processing: {error}")
            else:
                output = result.as_numpy("TEXT")
                for i in output:
                    self._results_dict[result.get_response().id].append(i)
    async def run(self):
        model_name = "vllm"
        sampling_parameters = {"temperature": "0.1", "top_p": "0.95"}
        stream = self._flags.streaming_mode
        with open(self._flags.input_prompts, "r") as file:
            print(f"Loading inputs from `{self._flags.input_prompts}`...")
            prompts = file.readlines()

        await self.process_stream(prompts, sampling_parameters, model_name)

        with open(self._flags.results_file, "w") as file:
            for id in self._results_dict.keys():
                for result in self._results_dict[id]:
                    file.write(result.decode("utf-8"))
                    file.write("\n")
                file.write("\n=========\n\n")
            print(f"Storing results into `{self._flags.results_file}`...")

        if self._flags.verbose:
            print(f"\nContents of `{self._flags.results_file}` ===>")
            system(f"cat {self._flags.results_file}")

        print("PASS: vLLM example")

    def run_async(self):
        self._loop.run_until_complete(self.run())

    def create_request(self, prompt, stream, request_id, sampling_parameters, model_name, send_parameters_as_tensor=True):
        inputs = []
        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        try:
            inputs.append(grpcclient.InferInput("PROMPT", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as error:
            print(f"Encountered an error during request creation: {error}")

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("STREAM", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        # Request parameters are not yet supported via BLS. Provide an
        # optional mechanism to send serialized parameters as an input
        # tensor until support is added
        
        if send_parameters_as_tensor:
            sampling_parameters_data = np.array(
                [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
            )
            inputs.append(grpcclient.InferInput("SAMPLING_PARAMETERS", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_parameters_data)

        # Add requested outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("TEXT"))

        # Issue the asynchronous sequence inference.
        return {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
            "parameters": sampling_parameters
        }

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
        help="Inference server URL and its gRPC port. Default is localhost:8001.",
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
        "--offset",
        type=int,
        required=False,
        default=0,
        help="Add offset to request IDs used",
    )
    parser.add_argument(
        "--input-prompts",
        type=str,
        required=False,
        default="prompts.txt",
        help="Text file with input prompts",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=False,
        default="results.txt",
        help="The file with output results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=False,
        default=1,
        help="Number of iterations through the prompts file",
    )
    parser.add_argument(
        "-s",
        "--streaming-mode",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode",
    )
    FLAGS = parser.parse_args()
    
    client = LLMClient(FLAGS)
    client.run_async()
