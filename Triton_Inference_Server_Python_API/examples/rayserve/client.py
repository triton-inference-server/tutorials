# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import subprocess
import time
from multiprocessing import Process

import numpy as np
import requests
from tqdm import tqdm

file_location = os.path.dirname(os.path.realpath(__file__))


def client(endpoint, request_count, prompt, save_image, index):
    latencies = []
    start = time.time()
    for i in tqdm(range(request_count)):
        if save_image:
            filename = os.path.join(
                file_location, f"client_{index}_generated_image{i}.jpg"
            )
            filename_input = "%20".join(f"&filename={filename}".split(" "))
        else:
            filename_input = ""
        prompt_input = "%20".join(prompt.split(" "))
        request_start = time.time()
        requests.get(
            f"http://127.0.0.1:8000/{endpoint}?prompt={prompt_input}{filename_input}",
            timeout=300,
        )
        latencies.append(time.time() - request_start)
    print(
        f"Client: {index} Throughput: {request_count/(time.time()-start)} Avg. Latency: {np.mean(latencies)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument(
        "--prompt",
        type=str,
        default="skeleton sitting by the side of a river looking soulful, concert poster, 4k, artistic",
    )
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--launch-nvidia-smi", action="store_true")
    parser.add_argument("--endpoint", type=str, default="generate")
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
                    args.endpoint,
                    args.requests,
                    args.prompt,
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
