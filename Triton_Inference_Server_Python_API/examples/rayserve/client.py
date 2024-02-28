import argparse
import subprocess
import time
from multiprocessing import Process

import requests
from tqdm import tqdm


def client(request_count, endpoint, index):
    prompts = [
        "who is garfield?",
        "what is ai?",
        "who are you?",
        "what did you have for lunch?",
        "what is rock and roll?",
    ]
    #    endpoint = "imagine"
    #    endpoint = "generate"
    # endpoint = "classify"

    print(f"starting client: {index}")
    start = time.time()
    for i in tqdm(range(request_count)):
        prompt = prompts[i % len(prompts)]
        input = "%20".join(prompt.split(" "))
        resp = requests.get(f"http://127.0.0.1:8000/{endpoint}?prompt={input}")
    end = time.time()
    print(f"client: {index} : requests per second: {request_count/(end-start)}")


# with open("output.png", 'wb') as f:
#   f.write(resp.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--endpoint", type=str, default="generate")
    parser.add_argument("--delay", type=int, default=0)
    args = parser.parse_args()
    proc = subprocess.Popen(["nvidia-smi", "dmon", "-f", "output.txt"])
    time.sleep(5)
    procs = []
    start_time = time.time()
    for i in range(args.concurrent):
        procs.append(Process(target=client, args=(args.requests, args.endpoint, i)))
        procs[-1].start()
        time.sleep(args.delay)

    for proc in procs:
        proc.join()
    end_time = time.time()
    time.sleep(5)
    proc.kill()
    print(
        f"overall throughput: {(args.requests*args.concurrent)/(end_time-start_time)}"
    )
