import time
import unittest

import pytest
from tritonserver import InferenceRequest, Model, Server


class IterativeSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._server = Server(
            model_repository="../model_repository",
            log_info=True,
            log_error=True,
            strict_model_config=False,
        )
        cls._server.start(wait_until_ready=True)

    def _infer_and_verify_response(self, max_tokens, prompt, ignore_eos, model_name):
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
        self._infer_and_verify_response(
            32,
            "Triton Inference Server is",
            ignore_eos=True,
            model_name="iterative-gpt2",
        )
        self._infer_and_verify_response(
            32, "Triton Inference Server is", ignore_eos=True, model_name="simple-gpt2"
        )

    @classmethod
    def tearDownClass(cls):
        time.sleep(5)
        cls._server.stop()
