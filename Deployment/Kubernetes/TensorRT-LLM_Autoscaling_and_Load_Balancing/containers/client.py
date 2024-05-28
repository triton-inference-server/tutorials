# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import os
import re
import subprocess
import sys
import time

ERROR_CODE_FATAL = 255
EXIT_CODE_SUCCESS = 0

DEBUG_KEY = "TRTLLM_DEBUG"
MAX_TOKENS_KEY = "TRTLLM_MAX_TOKENS"
MODEL_NAME_KEY = "TRTLLM_MODEL_NAME"
TRITON_URL_KEY = "TRTLLM_TRITON_URL"

MAX_TOKENS_DEFAULT = 512

is_debug = False

debug_value = os.getenv(DEBUG_KEY)

if debug_value is not None:
    is_debug = (
        debug_value == "1"
        or debug_value == "true"
        or debug_value == "yes"
        or debug_value == "debug"
    )

model_name = os.getenv(MODEL_NAME_KEY)

if model_name is None:
    raise Exception(f"Required environment variable '{MODEL_NAME_KEY}' not provided.")

if is_debug:
    print(f'model_name: "{model_name}".', file=sys.stdout, flush=True)

triton_url = os.getenv(TRITON_URL_KEY)

if triton_url is None:
    raise Exception(f"Required environment variable '{TRITON_URL_KEY}' not provided.")

triton_url = f"{triton_url}:8000/v2/models/{model_name}/generate"

if is_debug:
    print(f'triton_url: "{triton_url}".', file=sys.stdout, flush=True)

max_tokens = MAX_TOKENS_DEFAULT

max_token_value = os.getenv(MAX_TOKENS_KEY)
if max_token_value is not None:
    try:
        max_tokens = int(max_token_value)

    except:
        print(
            f"error: Environment variable {MAX_TOKENS_KEY}={max_token_value} is not valid and will be ignored.",
            file=sys.stderr,
            flush=True,
        )
        print(" ", file=sys.stderr, flush=True)

if is_debug:
    print(f"max_tokens: {max_tokens}.", file=sys.stdout, flush=True)

prompts = [
    "What is the market capitalization of NVIDIA?",
    "What is the largest company in the world?",
    "Who was the first president of France?",
    "How tall was Napoleon?",
    "Which colors are in the German flag?",
    "Does China have a national animal?",
    "What time is it in London?",
    "Write me a rap song using references to SpongeBob SquarePants.",
    "Give me 10 date-night ideas for my partner and me, but include ideas that we can do in the house, outdoors, and within a 10-mile radius.",
    "Write a short story about a unicorn and a postbox using only emojis.",
    "Write me advice on career planning, including how I can make steps towards financial goals and getting a promotion.",
    "Suggest 10 web extensions students can use to increase productivity.",
    "Write a strategy for how I can stay motivated at work and maintain focus.",
    "Create a bulleted list of organic supplements that boost metabolism.",
    "Which airlines have the best customer experience for long-haul flights?",
]

index = 0
error_count = 0

# Do this forever, or at least until a SIGABRT, SIGINT, or SIGKILL terminates the process.
while True:
    question = prompts[index]

    if is_debug:
        print(f'question: "{question}".')

    # Create a JSON encoded inference payload.
    payload = json.dumps({"text_input": question, "max_tokens": max_tokens})

    if is_debug:
        print(f'payload: "{payload}".')

    # Build up the subprocess args.
    args = ["curl", "-X", "POST", "-s", triton_url, "-d", payload]

    if is_debug:
        print(f"args: {args}")

    # Concat a human friendly command line and then log it.
    command = ""
    for arg in args:
        command += arg
        command += " "

    print(f"> {command}", file=sys.stdout, flush=True)

    index += 1
    index %= len(prompts)

    # Run the subprocess and catch any failures.
    try:
        time_start = time.time()

        sp_ran = subprocess.run(args, capture_output=True, check=True)

        if sp_ran.returncode != 0:
            print(sp_ran.stderr, file=sys.stderr, flush=True)
            print(" ", file=sys.stderr, flush=True)

            raise Exception(f'Inference command failed: "{exception}".')

        time_end = time.time()

        print(
            f"  completed in {(time_end - time_start)} seconds.",
            file=sys.stdout,
            flush=True,
        )
        print(" ", file=sys.stdout, flush=True)

        output = sp_ran.stdout

        if is_debug:
            print(f'output: "{output}".', file=sys.stdout, flush=True)

        result = json.loads(output)

        if is_debug:
            print(f'result: "{result}".', file=sys.stdout, flush=True)

        text_output = result["text_output"]

        if is_debug:
            print(f'text_output: "{text_output}".', file=sys.stdout, flush=True)

        answers = re.split("(\s{2,}|\n)", text_output)

        print("Prompt:", file=sys.stdout, flush=True)
        print(f"  {question}", file=sys.stdout, flush=True)
        print(" ", file=sys.stdout, flush=True)

        print("Response:", file=sys.stdout, flush=True)

        for answer in answers:
            print(f"  {answer.strip()}", file=sys.stdout, flush=True)

        print(" ", file=sys.stdout, flush=True)

    except Exception as exception:
        error_count += 1

        print(" ", file=sys.stderr, flush=True)
        print(f"error: {exception}", file=sys.stderr, flush=True)
        print(
            f"       Inference command has failed {error_count} time(s).",
            file=sys.stderr,
            flush=True,
        )

        if error_count > 30:
            print(f"fatal: Quitting after 30 failures.", file=sys.stderr, flush=True)
            exit(ERROR_CODE_FATAL)

    # 250ms delay between inference requests.
    time.sleep(0.250)
    print(" ", file=sys.stdout, flush=True)
