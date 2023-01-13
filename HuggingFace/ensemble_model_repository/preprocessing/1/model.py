import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import ViTFeatureExtractor


class TritonPythonModel:

    def initialize(self, args):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "image")
            input_image = np.squeeze(inp.as_numpy()).transpose((2,0,1))

            inputs = self.feature_extractor(images=input_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].numpy()

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "pixel_values",
                    pixel_values,
                )
            ])
            responses.append(inference_response)
        return responses