
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self,args):
        pass
    def execute(self,requests):
        responses = []
        for request in requests:
            text_input = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0]
            if isinstance(text_input,bytes):
                text_input = text_input.decode("utf-8")
            output_tensor = pb_utils.Tensor("text_output",
                                            np.asarray([text_input],dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses
                                                        
                
                
