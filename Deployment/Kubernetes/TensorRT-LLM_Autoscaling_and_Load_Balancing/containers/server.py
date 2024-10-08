# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import subprocess
import sys
import time

# These values are expected to match the mount points in the Helm Chart.
# Any changes here must also be made there, and vice versa.
CACHE_DIRECTORY = "/var/run/cache"
HUGGING_FACE_TOKEN_PATH = "/var/run/secrets/hugging_face/password"
MODEL_DIRECTORY = "/var/run/models/tensorrtllm_backend/triton_model_repo"

ERROR_EXIT_DELAY = 15
ERROR_CODE_FATAL = 255
ERROR_CODE_USAGE = 253
EXIT_SUCCESS = 0

# Environment variable keys.
CLI_VERBOSE_KEY = "TRITON_CLI_VERBOSE"
ENGINE_PATH_KEY = "ENGINE_DEST_PATH"
HUGGING_FACE_KEY = "HF_HOME"

HUGGING_FACE_CLI = "huggingface-cli"


# ---


def clean_directory(directory_path: str):
    if os.path.exists(path):
        if os.path.isfile(path):
            if is_verbose:
                write_output(f"> rm {path}")

            os.remove(path)
        else:
            if is_verbose:
                write_output(f"> rm -rf {path}")

            shutil.rmtree(path)


# ---


def die(exception: Exception = None):
    if exception is not None:
        write_error(f"fatal: {exception}")

    write_error(f"       Waiting {ERROR_EXIT_DELAY} second before exiting.")
    # Delay the process' termination to provide a small window for administrators to capture the logs before it exits and restarts.
    time.sleep(ERROR_EXIT_DELAY)

    exit(ERROR_CODE_USAGE)


# ---


def hugging_face_authenticate(args):
    # Validate that `HF_HOME` environment variable was set correctly.
    if HUGGING_FACE_HOME is None or len(HUGGING_FACE_HOME) == 0:
        raise Exception(f"Required environment variable '{HUGGING_FACE_KEY}' not set.")

    # When a Hugging Face secret has been mounted, we'll use that to authenticate with Hugging Face.
    if os.path.exists(HUGGING_FACE_TOKEN_PATH):
        with open(HUGGING_FACE_TOKEN_PATH) as token_file:
            write_output(
                f"Hugging Face token file '{HUGGING_FACE_TOKEN_PATH}' detected, attempting to authenticate w/ Hugging Face."
            )
            write_output(" ")

            hugging_face_token = token_file.read()

            # Use Hugging Face's CLI to complete the authentication.
            result = run_command(
                [HUGGING_FACE_CLI, "login", "--token"], [hugging_face_token]
            )

            if result != 0:
                raise Exception(f"Hugging Face authentication failed. ({result})")

            write_output("Hugging Face authentication successful.")
            write_output(" ")


# ---


def run_command(cmd_args: [], extra_args: [] = None):
    command = " ".join(cmd_args)

    if extra_args is not None and len(extra_args) > 0:
        command += "****"
        cmd_args += extra_args

    write_output(f"> {command}")
    write_output(" ")

    # Run triton_cli to build the TRT-LLM engine + plan.
    return subprocess.call(cmd_args, stderr=sys.stderr, stdout=sys.stdout)


# ---


def write_output(message: str):
    print(message, file=sys.stdout, flush=True)


# ---


def write_error(message: str):
    print(message, file=sys.stderr, flush=True)


# ---
# Below this line are the primary functions.
# ---


def execute_triton(args):
    world_size = args.tp * args.pp

    if world_size <= 0:
        raise Exception(
            "usage: Options --pp and --pp must both be equal to or greater than 1."
        )

    # Single GPU setups can start a tritonserver process directly.
    if world_size == 1:
        cmd_args = [
            "tritonserver",
            "--allow-cpu-metrics=false",
            "--allow-gpu-metrics=false",
            "--allow-metrics=true",
            "--metrics-interval-ms=1000",
            f"--model-repository={MODEL_DIRECTORY}",
            "--model-load-thread-count=2",
            "--strict-readiness=true",
        ]

        if args.verbose > 0:
            cmd_args += ["--log-verbose=1"]

        if args.iso8601 > 0:
            cmd_args += ["--log-format=ISO8601"]

    # Multi-GPU setups require a specialized command line which based on `mpirun`.
    else:
        cmd_args = ["mpirun", "--allow-run-as-root"]

        for i in range(world_size):
            if i != 0:
                cmd_args += [":"]

            cmd_args += [
                "-n",
                "1",
                "tritonserver",
                f"--id=rank{i}",
                f"--http-port={(8000 + i * 10)}",
                f"--grpc-port={(8001 + i * 10)}",
                "--model-load-thread-count=2",
                f"--model-repository={MODEL_DIRECTORY}",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=rank{i}_",
            ]

            if i == 0:
                cmd_args += [
                    "--allow-cpu-metrics=false",
                    "--allow-gpu-metrics=false",
                    "--allow-metrics=true",
                    "--metrics-interval-ms=1000",
                ]

                if args.verbose > 0:
                    cmd_args += ["--log-verbose=1"]

                if args.iso8601 > 0:
                    cmd_args += ["--log-format=ISO8601"]

            else:
                cmd_args += [
                    "--allow-http=false",
                    "--allow-grpc=false",
                    "--allow-metrics=false",
                    "--log-info=false",
                    "--log-warning=false",
                    "--model-control-mode=explicit",
                    "--load-model=tensorrt_llm",
                ]

    result = run_command(cmd_args)
    exit(result)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["exec", "init"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--dt",
        type=str,
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="Tensor type.",
    )
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism.")
    parser.add_argument("--iso8601", action="count", default=0)
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument(
        "--engine", type=str, default="trtllm", choices=["trtllm", "vllm"]
    )

    return parser.parse_args()


# ---


try:
    ENGINE_DIRECTORY = os.getenv(ENGINE_PATH_KEY)
    HUGGING_FACE_HOME = os.getenv(HUGGING_FACE_KEY)

    is_verbose = os.getenv(CLI_VERBOSE_KEY) is not None
    # Parse options provided.
    args = parse_arguments()

    if args.mode == "init":
        print("Hello, World!")
        exit(EXIT_SUCCESS)

    elif args.mode == "exec":
        # Update the is_verbose flag with values passed in by options.
        is_verbose = is_verbose or args.verbose > 0
        execute_triton(args)
    else:
        write_error(f"usage: server.py <mode> [<options>].")
        write_error(f'       Invalid mode ("{args.mode}") provided.')
        write_error(f'       Supported values are "init" or "exec".')
        die(None)

except Exception as exception:
    die(exception)
