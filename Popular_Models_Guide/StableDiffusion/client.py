# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import subprocess
import time
from multiprocessing import Process

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tqdm import tqdm
from tritonclient.utils import *


def client(model, request_count, prompt, batch_size, save_image, index):
    client = httpclient.InferenceServerClient(url="localhost:8000")
    latencies = []
    start = time.time()
    for i in tqdm(range(request_count), position=index):
        prompts = [prompt] * batch_size

        text_obj = np.array(prompts, dtype="object").reshape((-1, 1))

        input_text = httpclient.InferInput(
            "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        )
        input_text.set_data_from_numpy(text_obj)

        output_img = httpclient.InferRequestedOutput("generated_image")
        request_start = time.time()
        query_response = client.infer(
            model_name=model, inputs=[input_text], outputs=[output_img]
        )
        latencies.append(time.time() - request_start)
        image = query_response.as_numpy("generated_image")
        if save_image:
            im = Image.fromarray(np.squeeze(image.astype(np.uint8)))
            im.save(f"client_{index}_generated_image_{i}.jpg")
    print(
        f"Client: {index} Throughput: {request_count/(time.time()-start)} Avg. Latency: {np.mean(latencies)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example client demonstrating sending prompts to generative AI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Number of concurrent clients. Each client sends --requests number of requests.",
    )
    parser.add_argument(
        "--requests", type=int, default=1, help="Number of requests to send."
    )
    parser.add_argument(
        "--static-batch-size",
        type=int,
        default=1,
        help="Number of prompts to send in a single request",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="skeleton sitting by the side of a river looking soulful, concert poster, 4k, artistic",
        help="Prompt. All requests and batches will use the same prompt",
    )
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="If provided, generated images will be saved as jpeg files",
    )
    parser.add_argument(
        "--launch-nvidia-smi",
        action="store_true",
        help="Launch nvidia smi in daemon mode and log data to nvidia_smi_output.txt",
    )
    parser.add_argument(
        "--model", type=str, default="stable_diffusion_xl", help="model name"
    )
    args = parser.parse_args()
    if args.launch_nvidia_smi:
        nvidia_smi_proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-f", "nvidia_smi_output.txt"]
        )
        time.sleep(5)
    procs = []
    start_time = time.time()
    for i in range(args.clients):
        procs.append(
            Process(
                target=client,
                args=(
                    args.model,
                    args.requests,
                    args.prompt,
                    args.static_batch_size,
                    args.save_image,
                    i,
                ),
            )
        )
        procs[-1].start()

    for proc in procs:
        proc.join()
    end_time = time.time()
    if args.launch_nvidia_smi:
        time.sleep(5)
        nvidia_smi_proc.kill()
    print(
        f"Throughput: {(args.requests*args.clients)/(end_time-start_time)} Total Time: {end_time-start_time}"
    )
