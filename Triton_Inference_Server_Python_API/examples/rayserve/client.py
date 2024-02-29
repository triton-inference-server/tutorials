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
        input = "%20".join(prompt.split(" "))
        request_start = time.time()
        resp = requests.get(
            f"http://127.0.0.1:8000/{endpoint}?prompt={input}{filename_input}"
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
