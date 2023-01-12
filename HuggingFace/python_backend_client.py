import numpy as np
import time
from tritonclient.utils import *
from PIL import Image
import tritonclient.http as httpclient
import requests

def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    url = 'http://images.cocodataset.org/val2017/000000161642.jpg'
    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("image", image.shape, datatype="FP32")
    ]
    input_tensors[0].set_data_from_numpy(image)

    # Set outputs
    output_label = [
        httpclient.InferRequestedOutput("last_hidden_state")
    ]

    # Query
    query_response = client.infer(model_name="python_vit",
                                  inputs=input_tensors,
                                  outputs=output_label)

    # Output
    last_hidden_state = query_response.as_numpy("last_hidden_state")
    print(last_hidden_state.shape)

if __name__ == "__main__":
    main()
