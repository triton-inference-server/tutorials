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

import asyncio
import json
import os
import threading
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import triton_python_backend_utils as pb_utils
from pydantic import BaseModel, Field
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput as VLLMOutput

_VLLM_ENGINE_ARGS_FILENAME = "vllm_engine_args.json"


class VLLMAsyncEngineConfig(BaseModel):
    # required model
    model: str

    # arguments from vLLM engine
    max_num_batched_tokens: int = 2560
    max_num_seqs: int = 256
    disable_log_requests: bool = True


class CompletionRequest(BaseModel):
    # required prompt
    prompt: str

    # generation parameters
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None

    # sampling parameters
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = -1
    top_p: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False

    # output options
    echo: Optional[bool] = False
    stream: Optional[bool] = False


class Completion(BaseModel):
    index: int
    text: str
    gen_token_count: Optional[int] = None
    cumulative_logprob: Optional[float] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    request_id: Optional[str] = None
    finished: bool = False

    prompt: Optional[str] = None
    prompt_token_count: Optional[int] = None

    completions: List[Completion] = []

    # inflight request stats
    current_inflight_count: Optional[int] = None
    average_inflight_count: Optional[float] = None


class OutputOptions:
    def __init__(self, *, echo: bool = False, stream: bool = False):
        self._echo = echo
        self._stream = stream

    @property
    def echo(self) -> bool:
        """
        If true, the prompt and prompt token ids will be echoed back to the client in the response.
        """
        return self._echo

    @property
    def stream(self) -> bool:
        """
        If true, the response will be streamed back to the client as it is generated.
        """
        return self._stream


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger

        # load triton model config from args
        self.model_config = json.loads(args["model_config"])

        # assert are in decoupled mode
        self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )
        assert (
            self.using_decoupled
        ), "vLLM Triton backend must be configured to use decoupled model transaction policy"

        engine_args_filepath = os.path.join(
            args["model_repository"], _VLLM_ENGINE_ARGS_FILENAME
        )
        assert os.path.isfile(
            engine_args_filepath
        ), f"'{_VLLM_ENGINE_ARGS_FILENAME}' containing vllm engine args must be provided in '{args['model_repository']}'"
        with open(engine_args_filepath) as file:
            vllm_engine_config = VLLMAsyncEngineConfig(**json.load(file))

        # create the vllm engine
        engine_args = AsyncEngineArgs(
            vllm_engine_config.model,
            disable_log_requests=vllm_engine_config.disable_log_requests,
            max_num_batched_tokens=vllm_engine_config.max_num_batched_tokens,
            max_num_seqs=vllm_engine_config.max_num_seqs,
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)

        self._loop = asyncio.get_event_loop()
        self._loop_thread = threading.Thread(
            target=self.engine_loop, args=(self._loop,)
        )
        self._shutdown_event = asyncio.Event()
        self._loop_thread.start()

        # local counters to track inflight requests
        self._inflight_counter = 0

    def create_task(self, coro):
        """
        Creates a task on the engine's event loop which is running on a separate thread.
        """
        assert (
            self._shutdown_event.is_set() is False
        ), "Cannot create tasks after shutdown has been requested"

        async def _wrapped_coro(coro):
            """
            Wraps the given coroutine in a new coroutine and decrements the in-flight counter after awaiting the original coroutine.
            """
            await coro
            self._inflight_counter -= 1

        # locally increment the in-flight counter
        self._inflight_counter += 1
        return asyncio.run_coroutine_threadsafe(_wrapped_coro(coro), self._loop)

    def engine_loop(self, loop):
        """
        Runs the engine's event loop on a separate thread.
        """
        asyncio.set_event_loop(loop)
        self._loop.run_until_complete(self.await_shutdown())

    async def await_shutdown(self):
        """
        Primary coroutine running on the engine event loop. This coroutine is responsible for
        keeping the engine alive until a shutdown is requested.
        """
        # first await the shutdown signal
        while self._shutdown_event.is_set() is False:
            await asyncio.sleep(5)

        # then wait for all in-flight tasks to complete
        while self._inflight_counter > 0:
            self.logger.log_info(
                "Awaiting remaining {} inflight requests".format(self._inflight_counter)
            )
            await asyncio.sleep(5)

        self.logger.log_info("Shutdown complete")

    def make_sample_params(self, request: CompletionRequest) -> SamplingParams:
        """
        Creates a vLLM SamplingParams object from the given arguments.
        """
        return SamplingParams(
            n=1,
            best_of=1,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens,
        )

    def make_output_options(self, req: CompletionRequest) -> OutputOptions:
        """
        Creates an OutputOptions object from the given arguments.
        """
        return OutputOptions(
            echo=req.echo,
            stream=req.stream,
        )

    def preprocess(self, request) -> Tuple[str, SamplingParams, OutputOptions]:
        """
        Converts a Triton request into a VLLM input by extracting the prompt and sampling parameters.
        """
        # deserialize the VLLM request from the Triton InferRequest
        try:
            serialized_json = pb_utils.get_input_tensor_by_name(
                request, "serialized_request_json"
            ).as_numpy()[0]
            request = CompletionRequest(**json.loads(serialized_json))

            return (
                request.prompt,
                self.make_sample_params(request),
                self.make_output_options(request),
            )
        except Exception as e:
            self.logger.log_info(f"Error parsing request: {e}")
            raise e

    def postprocess(
        self, output: VLLMOutput, output_options: OutputOptions, inflight_avg: float
    ):
        """
        Converts a VLLM output into a Triton response.
        """
        # create a Triton Response from the vLLM output
        response = CompletionResponse()
        response.request_id = output.request_id
        response.finished = output.finished

        # number of tokens in the prompt and total number of tokens generated across all completions
        response.prompt_token_count = len(output.prompt_token_ids)

        # optionally echo the prompt back to the client
        if output_options.echo:
            response.prompt = output.prompt

        for completion in output.outputs:
            completion_output = Completion(
                index=completion.index,
                text=completion.text,
                finish_reason=completion.finish_reason,
                cumulative_logprob=completion.cumulative_logprob,
                gen_token_count=len(completion.token_ids),
            )
            response.completions.append(completion_output)

        response.current_inflight_count = self._inflight_counter
        response.average_inflight_count = inflight_avg

        # serialize the NemoResponse into a tensor packed into Triton InferenceResponse
        serialized_response = response.json().encode("utf-8")
        triton_output_tensor = pb_utils.Tensor(
            "serialized_response_json", np.array([serialized_response], dtype=np.object)
        )

        return pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])

    async def generate_stream(self, request):
        """
        Generates a stream of parital outputs from a single request.
        """
        try:
            request_id = request.request_id()
            prompt, sampling_params, output_options = self.preprocess(request)
            response_sender = request.get_response_sender()
            forward_passes = 0
            inflight_total = 0

            async for output in self.llm.generate(prompt, sampling_params, request_id):
                forward_passes += 1
                inflight_total += self._inflight_counter
                inflight_avg = inflight_total / forward_passes
                if output_options.stream or output.finished:
                    response_sender.send(
                        self.postprocess(output, output_options, inflight_avg)
                    )

            # TODO: Improve the API for sending FINAL flag; perhaps .complete() or .finalize()
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        except Exception as e:
            self.logger.log_info(f"Error generating stream: {e}")
            raise e

    def execute(self, requests):
        """
        Triton core issues requests to the backend via this method.

        When this method retruns, new requested can be issued to the backend.
        We are pushing all the requests on vllm and let it handle the full traffic.
        """
        for request in requests:
            self.create_task(self.generate_stream(request))
        return None

    def finalize(self):
        """
        Triton virtual method; called when the model is unloaded.
        """
        self.logger.log_info("Issuing finalize to vllm backend")
        self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join()
            self._loop_thread = None
