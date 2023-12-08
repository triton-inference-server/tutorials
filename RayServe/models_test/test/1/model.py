import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self._model_config = json.loads(args["model_config"])
        self._decoupled = self._model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )

    def execute_decoupled(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            output_tensors = []
            for input_tensor in request.inputs():
                input_value = input_tensor.as_numpy()
                output_tensor = pb_utils.Tensor(
                    input_tensor.name().replace("input", "output"), input_value
                )
                output_tensors.append(output_tensor)
            sender.send(pb_utils.InferenceResponse(output_tensors=output_tensors))
            sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None

    def execute(self, requests):
        if self._decoupled:
            return self.execute_decoupled(requests)
        responses = []
        for request in requests:
            output_tensors = []
            for input_tensor in request.inputs():
                input_value = input_tensor.as_numpy()
                output_tensor = pb_utils.Tensor(
                    input_tensor.name().replace("input", "output"), input_value
                )
                output_tensors.append(output_tensor)

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses
