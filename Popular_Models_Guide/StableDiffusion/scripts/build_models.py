# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import time

import tritonserver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument(
        "--model-repository", type=str, default="/workspace/diffusion-models"
    )
    parser.add_argument("--timeout", type=int, default=60 * 20)

    args = parser.parse_args()

    server = tritonserver.Server(
        model_repository=args.model_repository,
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    )

    server.start(wait_until_ready=True)
    models = server.models()

    if args.model == "all":
        models = models.keys()
    else:
        args.model = (args.model, -1)
        if not args.model in models:
            print(f"Model: {args.model} not known")
            sys.exit(1)
        models = [args.model]

    for model in models:
        if model[1] != -1:
            continue
        print(f"Loading Model: {model}")
        model = server.load(model[0])
        start = time.time()
        while not model.ready() and ((time.time() - start) <= args.timeout):
            time.sleep(10)

        if model.ready():
            print(f"Model: {model} Loaded")
        else:
            print(f"Error loading: {model}")
            sys.exit(1)

        server.unload(model, wait_until_unloaded=True)

    server.stop()
