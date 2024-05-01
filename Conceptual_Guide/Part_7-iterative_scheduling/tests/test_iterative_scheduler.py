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

import time
import unittest

from tritonserver import InferenceRequest, Model, Server


class IterativeSchedulerTest(unittest.TestCase):
    def setUp(self):
        self._server = Server(
            model_repository="../model_repository",
            log_info=True,
            log_error=True,
            strict_model_config=False,
        )
        self._server.start(wait_until_ready=True)

    def _infer_and_verify_request(self, max_tokens, prompt, ignore_eos, model_name):
        parameters = {"max_tokens": max_tokens, "ignore_eos": ignore_eos}
        model = Model(self._server, name=model_name)
        self.assertTrue(model.ready())
        request = InferenceRequest(
            model, parameters=parameters, inputs={"text_input": [[prompt]]}
        )
        responses = []
        response_iterator = model.infer(request)
        for response in response_iterator:
            responses.append(response)

        self.assertEqual(max_tokens, len(responses))

    def test_max_tokens(self):
        self._infer_and_verify_request(
            32,
            "Triton Inference Server is",
            ignore_eos=True,
            model_name="iterative-gpt2",
        )
        self._infer_and_verify_request(
            32, "Triton Inference Server is", ignore_eos=True, model_name="simple-gpt2"
        )

    def tearDown(self):
        # Add a small delay to allow graceful shutdown
        # This is related to a known issue with server
        # stop.
        time.sleep(5)
        self._server.stop()


if __name__ == "__main__":
    unittest.main()
