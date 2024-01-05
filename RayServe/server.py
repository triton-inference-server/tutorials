# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import cupy as cupy
from PIL import Image
import tritonserver
import torch


class TorchAllocator(tritonserver.MemoryAllocator):
    def allocate(self,
                 size,
                 memory_type,
                 memory_type_id,
                 tensor_name):

        device = "cpu"

        if memory_type == tritonserver.MemoryType.GPU:
            device = "cuda"
        
        tensor = torch.zeros(size,dtype=torch.uint8,device=device)
        print("torch allocator!")
        return tritonserver.MemoryBuffer.from_dlpack(tensor)

def example_1():
    server = tritonserver.Server(
        model_repository="/workspace/models_stable_diffusion",
        exit_timeout=5,
        log_error=True,
    )
    server.start(wait_until_ready=True)

    model = server.models()["pipeline"]

    responses = model.infer(
        inputs={
            "prompt": np.array(
                ["boy with balloon, gritty, urban, charcoal sketch"], dtype="object"
            ).reshape(-1, 1)
        }
    )

    for response in responses:
        output_tensor = response.outputs["generated_image"]
        ndarray = output_tensor.to_ndarray(np)
        generated_image = ndarray.squeeze().astype(np.uint8)
        im = Image.fromarray(generated_image)
        im.save("generated_image_1.jpg")

def example_2():
    server = tritonserver.Server(
        model_repository="/workspace/models_stable_diffusion",
        exit_timeout=5,
        log_error=True,
    )
    server.start(wait_until_ready=True)

    model = server.models()["pipeline"]

    responses = model.infer(
        inputs={
            "prompt": np.array(
                ["boy with balloon, gritty, urban, charcoal sketch"], dtype="object"
            ).reshape(-1, 1)
        },
        output_memory_type="CPU",
        output_array_module=np
    )

    for response in responses:
        output_tensor = response.outputs["generated_image"]
        generated_image = output_tensor.squeeze().astype(np.uint8)
        im = Image.fromarray(generated_image)
        im.save("generated_image_2.jpg")


def example_3():
    server = tritonserver.Server(
        model_repository="/workspace/models_stable_diffusion",
        exit_timeout=5,
        log_error=True,
    )
    server.start(wait_until_ready=True)

    model = server.models()["pipeline"]

    responses = model.infer(
        inputs={
            "prompt": np.array(
                ["boy with balloon, gritty, urban, charcoal sketch"], dtype="object"
            ).reshape(-1, 1)
        },
        output_memory_type="GPU",
        output_array_module=cupy
    )

    for response in responses:
        output_tensor = response.outputs["generated_image"]
        generated_image = output_tensor.squeeze().astype(np.uint8)
        im = Image.fromarray(generated_image.get())
        im.save("generated_image_3.jpg")


def example_4():
    server = tritonserver.Server(
        model_repository="/workspace/models_stable_diffusion",
        exit_timeout=5,
        log_error=True,
    )
    server.start(wait_until_ready=True)

    model = server.models()["pipeline"]

    responses = model.infer(
        inputs={
            "prompt": np.array(
                ["boy with balloon, gritty, urban, charcoal sketch"], dtype="object"
            ).reshape(-1, 1)
        },
        output_memory_type="GPU",
        output_array_module=torch,
        output_memory_allocator = TorchAllocator()
    )

    for response in responses:
        output_tensor = response.outputs["generated_image"]
        generated_image = output_tensor.squeeze().type(torch.uint8)
        im = Image.fromarray(generated_image.to("cpu").numpy())
        im.save("generated_image_4.jpg")


def example_5():
    server = tritonserver.Server(
        model_repository="/workspace/models_stable_diffusion",
        exit_timeout=5,
        log_error=True,
    )
    server.start(wait_until_ready=True)

    tritonserver.default_memory_allocators[tritonserver.MemoryType.GPU] = TorchAllocator()
    
    model = server.models()["pipeline"]

    responses = model.infer(
        inputs={
            "prompt": np.array(
                ["boy with balloon, gritty, urban, charcoal sketch"], dtype="object"
            ).reshape(-1, 1)
        },
        output_memory_type="GPU",
        output_array_module=torch,
    )

    for response in responses:
        output_tensor = response.outputs["generated_image"]
        generated_image = output_tensor.squeeze().type(torch.uint8)
        im = Image.fromarray(generated_image.to("cpu").numpy())
        im.save("generated_image_5.jpg")

        
def main():

    example_1()

    example_2()

    example_3()

    example_4()

    example_5()
    


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print("Time taken:", end - start)
