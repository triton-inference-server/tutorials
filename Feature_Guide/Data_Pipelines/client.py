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

import numpy as np
import time
from tritonclient.utils import np_to_triton_dtype
from PIL import Image
import tritonclient.http as httpclient
import requests

def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    prompts = ["This is a string"]
    text_obj = np.array([prompts], dtype="object")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(np.float32)
    uint8_array = np.expand_dims(np.array([1,2,3], dtype = np.uint8), axis = 0)
    int8_array = np.expand_dims(np.array([-1,2,-3], dtype = np.int8), axis = 0)
    image = np.expand_dims(image, axis=0)
    boolean = np.expand_dims(np.array([True]), axis = 0)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("ensemble_input_string", text_obj.shape,np_to_triton_dtype(text_obj.dtype)),
        httpclient.InferInput("ensemble_input_UINT8_array", uint8_array.shape, datatype="UINT8"),
        httpclient.InferInput("ensemble_input_INT8_array", int8_array.shape, datatype="INT8"),
        httpclient.InferInput("ensemble_input_FP32_image", image.shape, datatype="FP32"),
        httpclient.InferInput("ensemble_input_bool", boolean.shape, datatype="BOOL")
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(uint8_array)
    input_tensors[2].set_data_from_numpy(int8_array)
    input_tensors[3].set_data_from_numpy(image)
    input_tensors[4].set_data_from_numpy(boolean)

    # Set outputs
    output = [
        httpclient.InferRequestedOutput("ensemble_output_string"),
        httpclient.InferRequestedOutput("ensemble_output_UINT8_array"),
        httpclient.InferRequestedOutput("ensemble_output_INT8_array"),
        httpclient.InferRequestedOutput("ensemble_output_FP32_image"),
        httpclient.InferRequestedOutput("ensemble_output_bool")
    ]

    # Query
    query_response = client.infer(model_name="ensemble_model",
                                  inputs=input_tensors,
                                  outputs=output)

    print(query_response.as_numpy("ensemble_output_string"))
    print(query_response.as_numpy("ensemble_output_UINT8_array"))
    print(query_response.as_numpy("ensemble_output_INT8_array"))
    print(query_response.as_numpy("ensemble_output_FP32_image"))
    print(query_response.as_numpy("ensemble_output_bool"))


if __name__ == "__main__":
    main()
