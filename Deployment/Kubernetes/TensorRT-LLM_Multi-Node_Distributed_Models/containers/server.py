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
import signal
import subprocess
import sys
import time

# These values are expected to match the mount points in the Helm Chart.
# Any changes here must also be made there, and vice versa.
HUGGING_FACE_TOKEN_PATH = "/var/run/secrets/hugging_face/password"

ERROR_EXIT_DELAY = 15
ERROR_CODE_FATAL = 255
ERROR_CODE_USAGE = 253
EXIT_SUCCESS = 0

# Environment variable keys.
CLI_VERBOSE_KEY = "TRITON_CLI_VERBOSE"
ENGINE_PATH_KEY = "ENGINE_DEST_PATH"
HUGGING_FACE_KEY = "HF_HOME"
MODEL_PATH_KEY = "MODEL_DEST_PATH"

HUGGING_FACE_CLI = "huggingface-cli"
DELAY_BETWEEN_QUERIES = 2


# ---


def create_directory(directory_path: str):
    if directory_path is None or len(directory_path) == 0:
        return

    segments = directory_path.split("/")
    path = ""

    for segment in segments:
        if segment is None or len(segment) == 0:
            continue

        path = f"{path}/{segment}"

        if is_verbose:
            write_output(f"> mkdir {path}")

        if not os.path.exists(path):
            os.mkdir(path)


# ---


def die(exit_code: int):
    if exit_code is None:
        exit_code = ERROR_CODE_FATAL

    write_error(f"       Waiting {ERROR_EXIT_DELAY} second before exiting.")
    # Delay the process' termination to provide a small window for administrators to capture the logs before it exits and restarts.
    time.sleep(ERROR_EXIT_DELAY)

    exit(exit_code)


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
                [HUGGING_FACE_CLI, "login", "--token", hugging_face_token], [3]
            )

            if result != 0:
                raise Exception(f"Hugging Face authentication failed. ({result})")

            write_output("Hugging Face authentication successful.")
            write_output(" ")


# ---


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["convert", "leader", "worker"])
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
        "--deployment", type=str, help="Name of the Kubernetes deployment."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Namespace of the Kubernetes deployment.",
    )
    parser.add_argument("--multinode", action="count", default=0)
    parser.add_argument(
        "--noconvert",
        action="count",
        default=0,
        help="Prevents leader waiting for model conversion before inference serving begins.",
    )

    return parser.parse_args()


# ---


def remove_path(path: str):
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


def run_command(cmd_args: [str], omit_args: [int] = None):
    command = ""

    for i, arg in enumerate(cmd_args):
        command += " "
        if omit_args is not None and i in omit_args:
            command += "*****"
        else:
            command += arg

    write_output(f">{command}")
    write_output(" ")

    # Run triton_cli to build the TRT-LLM engine + plan.
    return subprocess.call(cmd_args, stderr=sys.stderr, stdout=sys.stdout)


# ---


def signal_handler(sig, frame):
    write_output(f"Signal {sig} detected, quitting.")
    exit(EXIT_SUCCESS)


# ---


def wait_for_convert(args):
    if args.noconvert != 0:
        write_output("Leader skip waiting for model-conversion job.")
        return

    write_output("Begin waiting for model-conversion job.")

    cmd_args = [
        "kubectl",
        "get",
        f"job/{args.deployment}",
        "-n",
        f"{args.namespace}",
        "-o",
        'jsonpath={.status.active}{"|"}{.status.failed}{"|"}{.status.succeeded}',
    ]
    command = " ".join(cmd_args)

    active = 1
    failed = 0
    succeeded = 0

    while active > 0 and succeeded == 0:
        time.sleep(DELAY_BETWEEN_QUERIES)

        if is_verbose:
            write_output(f"> {command}")

        output = subprocess.check_output(cmd_args).decode("utf-8")
        if output is None or len(output) == 0:
            continue

        if is_verbose:
            write_output(output)

        output = output.strip(" ")
        if len(output) > 0:
            parts = output.split("|")

            if len(parts) > 2 and len(parts[2]) > 0:
                succeeded = int(parts[2])
            else:
                succeeded = 0

            if len(parts) > 1 and len(parts[1]) > 0:
                failed = int(parts[1])
            else:
                failed = 0

            if len(parts) > 0 and len(parts[0]) > 0:
                active = int(parts[0])
            else:
                active = 0

        if active > 0:
            write_output("Waiting for model-conversion job.")
        elif succeeded > 0:
            write_output("Model-conversion job succeeded.")
        elif failed > 0:
            write_error("Model-conversion job failed.")
            raise RuntimeError("Model-conversion job failed.")

    write_output(" ")


# ---


def wait_for_workers(world_size: int):
    if world_size is None or world_size <= 0:
        raise RuntimeError("Argument `world_size` must be greater than zero.")

    write_output("Begin waiting for worker pods.")

    cmd_args = [
        "kubectl",
        "get",
        "pods",
        "-n",
        f"{args.namespace}",
        "-l",
        f"app={args.deployment}",
        "-o",
        "jsonpath='{.items[*].metadata.name}'",
    ]
    command = " ".join(cmd_args)

    workers = []

    while len(workers) < world_size:
        time.sleep(DELAY_BETWEEN_QUERIES)

        if is_verbose:
            write_output(f"> {command}")

        output = subprocess.check_output(cmd_args).decode("utf-8")

        if is_verbose:
            write_output(output)

        output = output.strip("'")

        workers = output.split(" ")

        if len(workers) < world_size:
            write_output(
                f"Waiting for worker pods, {len(workers)} of {world_size} ready."
            )
        else:
            write_output(f"{len(workers)} of {world_size} workers ready.")

    write_output(" ")

    if workers is not None and len(workers) > 1:
        workers.sort()

    return workers


# ---


def write_output(message: str):
    print(message, file=sys.stdout, flush=True)


# ---


def write_error(message: str):
    print(message, file=sys.stderr, flush=True)


# ---
# Below this line are the primary functions.
# ---


def do_convert(args):
    write_output("Initializing Model")

    if args.model is None or len(args.model) == 0:
        write_error("fatal: Model name must be provided.")
        die(ERROR_CODE_FATAL)

    create_directory(ENGINE_DIRECTORY)
    create_directory(MODEL_DIRECTORY)

    hugging_face_authenticate(args)

    engine_path = ENGINE_DIRECTORY
    engine_lock_file = os.path.join(engine_path, "lock")
    engine_ready_file = os.path.join(engine_path, "ready")
    model_path = MODEL_DIRECTORY
    model_lock_file = os.path.join(model_path, "lock")
    model_ready_file = os.path.join(model_path, "ready")

    # When the model and plan already exist, we can exit early, happily.
    if os.path.exists(engine_ready_file) and os.path.exists(model_ready_file):
        everything_exists = True

        if os.path.exists(engine_lock_file):
            write_output("Incomplete engine directory detected, removing.")
            everything_exists = False
            remove_path(engine_path)

        if os.path.exists(model_lock_file):
            write_output("Incomplete model directory detected, removing.")
            everything_exists = False
            remove_path(engine_path)

        if everything_exists:
            write_output(
                f"TensorRT engine and plan detected for {args.model}. No work to do, exiting."
            )
            exit(EXIT_SUCCESS)

    write_output(f"Begin generation of TensorRT engine and plan for {args.model}.")
    write_output(" ")

    create_directory(engine_path)

    # Create a lock file for the engine directory.
    if is_verbose:
        write_output(f"> echo '{args.model}' > {engine_lock_file}")

    with open(engine_lock_file, "w") as f:
        f.write(args.model)

    create_directory(model_path)

    # Create a lock file for the engine model.
    if is_verbose:
        write_output(f"> echo '{args.model}' > {model_lock_file}")

    with open(model_lock_file, "w") as f:
        f.write(args.model)

    try:
        # Build up a set of args for the subprocess call.
        cmd_args = [
            "triton",
            "import",
            "--model",
            args.model,
            "--model-repository",
            MODEL_DIRECTORY,
        ]

        cmd_args += ["--backend", "tensorrtllm"]

        if args.dt is not None and args.dt in ["bfloat", "float16", "float32"]:
            cmd_args += ["--data-type", args.dt]

        if args.pp > 1:
            cmd_args += ["--pipeline-parallelism", f"{args.pp}"]

        if args.tp > 1:
            cmd_args += ["--tensor-parallelism", f"{args.tp}"]

        if args.tp * args.pp > 1 and args.multinode > 0:
            cmd_args += ["--disable-custom-all-reduce"]

        # When verbose, insert the verbose flag.
        # It is important to note that the flag must immediately follow `triton` and cannot be in another ordering position.
        # This limitation will likely be removed a future release of triton_cli.
        if is_verbose:
            cmd_args.insert(1, "--verbose")

        result = run_command(cmd_args)

        if result == 0:
            # Create the ready file.
            if is_verbose:
                write_output(f"> echo '{args.model}' > {engine_ready_file}")

            with open(engine_ready_file, "w") as f:
                f.write(args.model)

            # Create the ready file.
            if is_verbose:
                write_output(f"> echo '{args.model}' > {model_ready_file}")

            with open(model_ready_file, "w") as f:
                f.write(args.model)

            # Remove the lock files.
            if is_verbose:
                write_output(f"> rm {engine_lock_file}")

            os.remove(engine_lock_file)

            if is_verbose:
                write_output(f"> rm {model_lock_file}")

            os.remove(model_lock_file)
        else:
            # Clean the model and engine directories when the command fails.
            remove_path(engine_path)
            remove_path(model_path)

        exit(result)

    except Exception as exception:
        remove_path(engine_path)
        remove_path(model_path)
        raise exception


# ---


def do_leader(args):
    world_size = args.tp * args.pp

    if world_size <= 0:
        raise Exception(
            "usage: Options --pp and --pp must both be equal to or greater than 1."
        )

    write_output(f"Executing Leader (world size: {world_size})")

    wait_for_convert(args)

    workers = wait_for_workers(world_size)

    if len(workers) != world_size:
        write_error(f"fatal: {len(workers)} found, expected {world_size}.")
        die(ERROR_EXIT_DELAY)

    cmd_args = [
        "mpirun",
        "--allow-run-as-root",
    ]

    if is_verbose > 0:
        cmd_args += ["--debug-devel"]

    cmd_args += [
        "--report-bindings",
        "-mca",
        "plm_rsh_agent",
        "kubessh",
        "-np",
        f"{world_size}",
        "--host",
        ",".join(workers),
    ]

    # Add per node command lines separated by ':'.
    for i in range(world_size):
        if i != 0:
            cmd_args += [":"]

        cmd_args += [
            "-n",
            "1",
            "tritonserver",
            "--allow-cpu-metrics=false",
            "--allow-gpu-metrics=false",
            "--disable-auto-complete-config",
            f"--id=rank{i}",
            "--model-load-thread-count=2",
            f"--model-repository={MODEL_DIRECTORY}",
        ]

        # Rank0 node needs to support metrics collection and web services.
        if i == 0:
            cmd_args += [
                "--allow-metrics=true",
                "--metrics-interval-ms=1000",
            ]

            if is_verbose > 0:
                cmd_args += ["--log-verbose=1"]

            if args.iso8601 > 0:
                cmd_args += ["--log-format=ISO8601"]

        # Rank(N) nodes can disable metrics, web services, and logging.
        else:
            cmd_args += [
                "--allow-http=false",
                "--allow-grpc=false",
                "--allow-metrics=false",
                "--model-control-mode=explicit",
                "--load-model=tensorrt_llm",
                "--log-info=false",
                "--log-warning=false",
            ]

    result = run_command(cmd_args)

    if result != 0:
        die(result)

    exit(result)


# ---


def do_worker(args):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    write_output("Worker paused awaiting SIGINT or SIGTERM.")
    signal.pause()


# ---


write_output("Reporting system information.")
run_command(["whoami"])
run_command(["cgget", "-n", "--values-only", "--variable memory.limit_in_bytes", "/"])
run_command(["nvidia-smi"])

ENGINE_DIRECTORY = os.getenv(ENGINE_PATH_KEY)
HUGGING_FACE_HOME = os.getenv(HUGGING_FACE_KEY)
MODEL_DIRECTORY = os.getenv(MODEL_PATH_KEY)

is_verbose = os.getenv(CLI_VERBOSE_KEY) is not None

# Validate that `ENGINE_PATH_KEY` isn't empty.
if ENGINE_DIRECTORY is None or len(ENGINE_DIRECTORY) == 0:
    raise Exception(f"Required environment variable '{ENGINE_PATH_KEY}' not set.")

# Validate that `MODEL_PATH_KEY` isn't empty.
if MODEL_DIRECTORY is None or len(MODEL_DIRECTORY) == 0:
    raise Exception(f"Required environment variable '{MODEL_PATH_KEY}' not set.")

# Parse options provided.
args = parse_arguments()

# Update the is_verbose flag with values passed in by options.
is_verbose = is_verbose or args.verbose > 0

if is_verbose:
    write_output(f"{ENGINE_PATH_KEY}='{ENGINE_DIRECTORY}'")
    write_output(f"{HUGGING_FACE_KEY}='{HUGGING_FACE_HOME}'")
    write_output(f"{MODEL_PATH_KEY}='{MODEL_DIRECTORY}'")

if args.mode == "convert":
    do_convert(args)

elif args.mode == "leader":
    do_leader(args)

elif args.mode == "worker":
    do_worker(args)

else:
    write_error(f"usage: server.py <mode> [<options>].")
    write_error(f'       Invalid mode ("{args.mode}") provided.')
    write_error(f'       Supported values are "init" or "exec".')
    die(ERROR_CODE_USAGE)
