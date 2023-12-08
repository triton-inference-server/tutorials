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
import queue
import time
import unittest

import numpy
import pytest
import tritonserver
try:
    import cupy
except Exception:
    cupy=None

class TrtionServerAPITest(unittest.TestCase):
    def test_not_started(self):
        server = tritonserver.Server()
        with self.assertRaises(tritonserver.InvalidArgumentError):
            server.is_ready()

    @pytest.mark.skipif(cupy is None, reason="Skipping gpu memory, cpupy not installed")
    def test_gpu_memory(self):

        import cupy

        server = tritonserver.Server(
            model_repository="/workspace/models",
            exit_timeout=5
        )

        server.start(blocking=True)

        test = server.get_model("test")
        fp16_input = cupy.array([[5], [6], [7], [8]], dtype=numpy.float16)
        responses = test.infer(inputs={"fp16_input": fp16_input}, request_id="1")

        for response in responses:
            print(response)

        responses = server.models["test"].infer(
            inputs={"fp16_input": fp16_input}, request_id="1"
        )

        for response in responses:
            print(response)
        try:
            pass
#            server.stop()
        except Exception as error:
            print(error)
            

    def test_inference(self):
        server = tritonserver.Server(
            model_repository="/workspace/models",
            exit_timeout=5
            #           log_verbose=True,
            #            log_error=True,
        )
        server.start()
        while not server.is_ready():
            pass

        response_queue = queue.SimpleQueue()

        test = server.get_model("test")
        test_2 = server.get_model("test_2")

        inputs = {
            "text_input": numpy.array(["hello"], dtype=numpy.object_),
            "fp16_input": numpy.array([["1"]], dtype=numpy.float16),
        }

        responses_1 = test.infer(
            inputs=inputs, request_id="1", response_queue=response_queue
        )
        responses_2 = test.infer(
            inputs=inputs, request_id="2", response_queue=response_queue
        )

        responses_3 = test_2.infer(inputs=inputs)

        for response in responses_3:
            print(response)

        count = 0
        while count < 2:
            response = response_queue.get()
            count += 1
            print(response, count)
            print(response.outputs["text_output"])
            print(bytes(response.outputs["text_output"][0]))
            print(type(response.outputs["text_output"][0]))
            print(response.outputs["fp16_output"])
            print(type(response.outputs["fp16_output"][0]))

        #     for response in test.infer(inputs=inputs):
        #        print(response.outputs["text_output"])
        #       print(response.outputs["fp16_output"])

        print(test.statistics())
        print(test_2.statistics())

        #        print(server.metrics())


        try:
            pass
#            server.stop()
        except Exception as error:
            print(error)



class AsyncInferenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_async_inference(self):
        server = tritonserver.Server(
            model_repository=["/workspace/models"],
            exit_timeout=30
            #                                         log_verbose=True,
            #                                        log_error=True)
        )
        server.start()
        while not server.is_ready():
            pass

        test = server.models["test"]

        inputs = {
            "text_input": numpy.array(["hello"], dtype=numpy.object_),
            "fp16_input": numpy.array([["1"]], dtype=numpy.float16),
        }

        response_queue = asyncio.Queue()
        responses = test.async_infer(
            inputs=inputs, response_queue=response_queue, request_id="1"
        )
        responses_2 = test.async_infer(
            inputs=inputs, response_queue=response_queue, request_id="2"
        )
        responses_3 = test.async_infer(
            inputs=inputs, response_queue=response_queue, request_id="3"
        )

        print("here cancelling!", flush=True)
        responses.cancel()
        print("here cancelling!", flush=True)

        async for response in responses:
            print("async")
            print(response.outputs["text_output"])
            print(response.outputs["fp16_output"])
            print(response.request_id)

        count = 0
        while count < 3:
            response = await response_queue.get()
            print(response, count)
            count += 1

        print("calling stop!")
        try:
            pass
#            server.stop()
        except Exception as error:
            print(error)
        print("stopped!", flush=True)
