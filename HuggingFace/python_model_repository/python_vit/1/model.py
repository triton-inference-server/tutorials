import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import ViTFeatureExtractor, ViTModel

class TritonPythonModel:
    def initialize(self, args):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k').to("cuda")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to("cuda")

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "image")
            input_image = np.squeeze(inp.as_numpy()).transpose((2,0,1))
            inputs = self.feature_extractor(images=input_image, return_tensors="pt")

            outputs = self.model(**inputs)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "label",
                    outputs.last_hidden_state.numpy()
                )
            ])
            responses.append(inference_response)
        return responses
