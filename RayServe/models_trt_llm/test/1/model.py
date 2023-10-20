import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
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
