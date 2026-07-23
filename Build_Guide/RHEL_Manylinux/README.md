<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Building Triton Inference Server for RHEL / manylinux

> [!WARNING]
> **This is a community example — not officially supported.** Triton's
> `build.py --target-platform=rhel` path is experimental (RHEL is not an officially
> supported target), and the base image reconstructed in this tutorial is equivalent,
> not identical, to the one NVIDIA uses internally to produce the released `manylinux`
> artifacts.

NVIDIA publishes prebuilt `manylinux` (RHEL 8‑compatible) Triton Inference Server
artifacts. Reproducing them ourselves with `build.py --target-platform=rhel` requires a
CUDA/cuDNN/TensorRT base image that is not published, so this tutorial reconstructs an
equivalent one from public sources — a public NVIDIA CUDA image on Rocky Linux 8 plus
TensorRT from the public CUDA repo — and walks through building and running a
RHEL/manylinux Triton server end‑to‑end.

By the end of this tutorial, we will produce the following:

1. **The manylinux artifacts.** `build/install/` holds the `tritonserver` / `tritonfrontend`
   wheels tagged `…-cp312-cp312-manylinux_2_XX_x86_64.whl` plus the backend trees we build (e.g. `onnxruntime`, `pytorch`, `python`) — all from 100% public inputs.
2. **A working, provably-manylinux container.** The built image serves real inference, and both
   the wheels and the runtime are verified to conform to `manylinux` (glibc 2.28 / EL8),
   so it runs on RHEL 8 and derivatives (Rocky, AlmaLinux, …). Running on RHEL 9 works too, but
   the server binary needs OpenSSL 1.1 (`libssl.so.1.1`) there (see
   [Known differences](#known-differences-from-the-released-artifacts)).

The commands below target **Triton 2.69.0 / NGC 26.05** (CUDA 13.2.1, TensorRT
10.16.1.11, Python 3.12). To target a different release, change the versions in
Steps 1 and 3 to match that release.

## Prerequisites

- A **Linux x86-64 host with a working Docker daemon.** This tutorial builds with `build.py`
  the same way a standard Triton build does; Triton's supported build platform is
  [Ubuntu 22.04, x86-64](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md).
  Since the build runs in containers, other x86-64 Linux hosts with Docker should work too.
  A **GPU is not required to build** since
  CUDA libraries come from the base image; you only need a GPU to run GPU inference.
- **Disk space and build time depend on which backends you build.** Building the `onnxruntime`
  backend from source alone takes ~2 hours and tens of GB; a minimal build is much quicker.
- Network access to the public **NVIDIA** package repos, **PyPI**, and **GitHub**
  (`build.py` clones the backend sources).
- [`Dockerfile.base.rhel`](Dockerfile.base.rhel) — **required**; the base image every
  `--target-platform=rhel` build needs.
- [`Dockerfile.pytorch.rhel`](Dockerfile.pytorch.rhel) — **optional**; only if you
  build the `pytorch` backend (Step 2).
- [`Dockerfile.pytorch-runtime.rhel`](Dockerfile.pytorch-runtime.rhel) — **optional**;
  completes the built image so the `pytorch` backend can serve (Step 4).

> [!NOTE]
> **Choosing backends.** Skip any backend by leaving its `--backend=` flag out of Step 3;
> dropping `onnxruntime` also avoids the ~2h source build. The PyTorch image (Step 2) is only
> needed for the `pytorch` backend.
>
> TensorRT is the exception: a GPU `onnxruntime` build auto-enables ONNX Runtime's
> TensorRT provider (when building for `rhel` on x86-64), so it's pulled in even without
> the `tensorrt` backend. For a TensorRT-free build, either skip both `onnxruntime` and
> `tensorrt`, or keep `onnxruntime` and disable the provider with
> `--override-backend-cmake-arg onnxruntime:TRITON_ENABLE_ONNXRUNTIME_TENSORRT=OFF`; then
> drop the TensorRT step from `Dockerfile.base.rhel`.

## Step 1: Build the public base image

On the `rhel` path, `build.py` only installs DCGM (NVIDIA's Data Center GPU Manager) and
expects the CUDA/cuDNN/TensorRT stack and a set of OS `-devel` packages to already be
present in the base image. [`Dockerfile.base.rhel`](Dockerfile.base.rhel) reconstructs
that from public sources: it starts from NVIDIA's official
`nvidia/cuda:*-cudnn-devel-rockylinux8` image (CUDA + cuDNN, on Rocky Linux 8 = RHEL 8 /
glibc 2.28), enables **EPEL + PowerTools**, installs **TensorRT** from the public
`cuda-rhel8` repo, and adds the compiler toolchain, Python headers, and wheel tooling that
`build.py` assumes. (PyTorch's extra CUDA runtime libraries — cuSPARSELt, NCCL, nvshmem — are
*not* added here; they ship inside the torch wheel and get wired up in the Step 4 completion
image, so the base stays generic).

To build this image, run the following command, pinning the CUDA image and TensorRT to your release versions:

```bash
docker build -f Dockerfile.base.rhel \
  --build-arg BASE_IMAGE=nvidia/cuda:13.2.1-cudnn-devel-rockylinux8 \
  --build-arg TENSORRT_VERSION=10.16.1.11-1.cuda13.2 \
  -t triton-manylinux-base:example .
```

## Step 2 (optional): Build the PyTorch backend image

Only needed if you build the `pytorch` backend.
[`Dockerfile.pytorch.rhel`](Dockerfile.pytorch.rhel) installs a public `torch` wheel into a
`manylinux_2_28` image so the PyTorch backend can extract a libtorch that runs on EL8's
glibc 2.28. Note that the default Ubuntu-based `libtorch` is built against a newer glibc and won't
load there.

```bash
docker build -f Dockerfile.pytorch.rhel \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
  -t triton-manylinux-pytorch:example .
```

Note that `build.py` consumes `--image=pytorch` with an unconditional `docker pull` (even under
`--no-container-pull`), so the image must be reachable from a registry. Push it to a temporary local one:

```bash
docker run -d -p 5000:5000 --name registry registry:2
docker tag  triton-manylinux-pytorch:example localhost:5000/triton-manylinux-pytorch:example
docker push localhost:5000/triton-manylinux-pytorch:example
```

Then add the PyTorch flags to the Step 3 command (shown there), and complete the serving image
in Step 4. `localhost:5000` is just a stand-in for NVIDIA's internal registry.

## Step 3: Build the server

Clone the server repo at the matching release branch and run `build.py` with your base
image:

```bash
git clone https://github.com/triton-inference-server/server.git
cd server && git checkout r26.05  # choose any release version

./build.py -v --target-platform=rhel --no-container-pull \
  --version=2.69.0 --container-version=26.05 --upstream-container-version=26.05 \
  --image=base,triton-manylinux-base:example \
  --image=pytorch,localhost:5000/triton-manylinux-pytorch:example \
  --extra-core-cmake-arg=PYBIND11_FINDPYTHON=ON \
  --enable-gpu --enable-logging --enable-stats --enable-metrics \
  --enable-gpu-metrics --enable-cpu-metrics --enable-tracing \
  --endpoint=http --endpoint=grpc \
  --backend=onnxruntime:r26.05 --backend=pytorch:r26.05 --backend=python:r26.05 \
  --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_NVSHMEM=ON \
  --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
  --repoagent=checksum:r26.05
```

Key flags:

- `--no-container-pull` — assuming your base image is local, this stops Docker from
  trying to pull it. It does **not** stop the pytorch backend's own pull of `--image=pytorch`
  — that's why Step 2 pushes to a local registry.
- If you're running headless (CI, `ssh` without a TTY, etc.), add `--no-container-interactive` —
  build.py launches the compile with `docker run -it` by default, which aborts with
  `the input device is not a TTY` when no terminal is attached.
- `--extra-core-cmake-arg=PYBIND11_FINDPYTHON=ON` — makes pybind11 use the Python 3.12 you
  installed instead of Rocky 8's system `python3` (3.6), which it otherwise picks up and, lacking
  dev headers, fails against with `fatal error: Python.h`.
- `--image=pytorch,localhost:5000/…` — the prebuilt PyTorch image from Step 2. The other backends
  (onnxruntime, tensorrt, python) compile from source during the build; PyTorch instead reuses a
  **prebuilt** libtorch (too heavy to build in-tree), which build.py `docker pull`s and extracts from
  this image — the only backend that needs an `--image`.
- `TRITON_PYTORCH_ENABLE_TORCHVISION=OFF` — this example builds without torchvision;
  wiring torchvision up from public sources is an untested path in this tutorial.
- `TRITON_PYTORCH_NVSHMEM=ON` — leave nvshmem on so build.py copies `libtorch_nvshmem.so`
  (libtorch links it); Step 4 supplies the one runtime library it in turn needs.

The `python` backend is built here because it makes build.py provision the pyenv Python + numpy
the pytorch backend serves against (Step 4). It's optional — drop `--backend=python` and you must
re-add those to the completion image yourself. `tensorrt` is optional too (ONNX Runtime already
pulls in its TensorRT provider); add `--backend=tensorrt:r26.05` for the standalone backend.

Triton's `common`, `core`, `backend`, and `third_party` repos don't need explicit
`--repo-tag` flags — build.py defaults them to the branch matching `--container-version`
(`r26.05` here).

ONNX Runtime is compiled from source here (~2 hours — it builds CUDA kernels for several
GPU architectures). To speed it up, build for only your GPU's architecture — see the
[`onnxruntime_backend`](https://github.com/triton-inference-server/onnxruntime_backend) build options.

The build produces the install tree under `build/install/` and a local `tritonserver`
Docker image.

## Step 4: Verify

**1. Manylinux artifact generation**

> [!NOTE]
> The `rhel` build currently **ships wheels tagged `linux_x86_64`, not `manylinux`.** It *does*
> run `auditwheel repair` and produce correct `manylinux_2_27` wheels (under the build container's
> `.../python/generic/` dirs), but the packaging step installs the *un-repaired* copies instead. As a workaround, pull the repaired wheels out of the build container:

```bash
docker start tritonserver_builder >/dev/null
docker exec tritonserver_builder sh -c 'find /tmp/tritonbuild -name "*manylinux*.whl"' \
  | while read -r w; do docker cp "tritonserver_builder:$w" build/install/python/; done
docker stop tritonserver_builder >/dev/null
rm -f build/install/python/*-linux_x86_64.whl
```

Now `ls` the artifacts and prove the tag is real (not just a filename) with `auditwheel`, which
checks the wheel's external symbols actually fit within the target glibc (auditwheel picks the
true minimum — `2_27` here, which is *more* portable than 2_28):

```bash
ls build/install/backends                 # onnxruntime  pytorch  python
find build/install -name '*.whl'          # ...-cp312-cp312-manylinux_2_27_x86_64.whl

# auditwheel lives in the base image — run it from there, nothing installed on your host:
docker run --rm -v "$PWD/build:/b:ro" triton-manylinux-base:example \
  bash -c 'auditwheel show /b/install/python/tritonserver-*.whl'
#  ... is consistent with the following platform tag: "manylinux_2_27_x86_64"
#  ... external versioned symbols in system libraries: libc.so.6 (GLIBC_2.2.5 ... 2.27)
```

**2. Serve real workloads from Manylinux container**

Run the server and real inference. The
server binary links against EL8's `libssl.so.1.1`, so run it **inside the built image** (Rocky 8)
rather than on a non‑EL8 host. First create a model repository with two
`OUTPUT0 = INPUT0 + INPUT1` models — one Python, one ONNX.

```bash
# python backend model
mkdir -p models/add_py/1
cat > models/add_py/config.pbtxt <<'EOF'
name: "add_py"
backend: "python"
max_batch_size: 0
input [
  { name: "INPUT0", data_type: TYPE_FP32, dims: [4] },
  { name: "INPUT1", data_type: TYPE_FP32, dims: [4] }
]
output [ { name: "OUTPUT0", data_type: TYPE_FP32, dims: [4] } ]
instance_group [ { kind: KIND_CPU } ]
EOF
cat > models/add_py/1/model.py <<'EOF'
import numpy as np
import triton_python_backend_utils as pb_utils
class TritonPythonModel:
    def execute(self, requests):
        out = []
        for r in requests:
            a = pb_utils.get_input_tensor_by_name(r, "INPUT0").as_numpy()
            b = pb_utils.get_input_tensor_by_name(r, "INPUT1").as_numpy()
            t = pb_utils.Tensor("OUTPUT0", (a + b).astype(np.float32))
            out.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return out
EOF

# onnx model (validates the onnxruntime backend)
mkdir -p models/add_onnx/1
cat > models/add_onnx/config.pbtxt <<'EOF'
name: "add_onnx"
backend: "onnxruntime"
max_batch_size: 0
input [
  { name: "INPUT0", data_type: TYPE_FP32, dims: [4] },
  { name: "INPUT1", data_type: TYPE_FP32, dims: [4] }
]
output [ { name: "OUTPUT0", data_type: TYPE_FP32, dims: [4] } ]
instance_group [ { kind: KIND_CPU } ]
EOF
# generate the .onnx in a temporary container (no host install)
cat > /tmp/gen_onnx.py <<'EOF'
import onnx
from onnx import helper, TensorProto
g = helper.make_graph(
    [helper.make_node("Add", ["INPUT0", "INPUT1"], ["OUTPUT0"])], "add",
    [helper.make_tensor_value_info("INPUT0", TensorProto.FLOAT, [4]),
     helper.make_tensor_value_info("INPUT1", TensorProto.FLOAT, [4])],
    [helper.make_tensor_value_info("OUTPUT0", TensorProto.FLOAT, [4])])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]),
          "/models/add_onnx/1/model.onnx")
EOF
docker run --rm -v "$PWD/models:/models" -v /tmp/gen_onnx.py:/gen.py:ro python:3.12-slim \
  bash -c "pip install --quiet onnx && python /gen.py"
```

Start the server (CPU‑only, so no GPU is required for this check). If port `8000` is already in
use, map a free host port instead, e.g. `-p 8080:8000`, then use `localhost:8080` below:

```bash
docker run --rm -p8000:8000 -p8001:8001 -v "$PWD/models:/models" \
  tritonserver:latest tritonserver --model-repository=/models
# wait for: "successfully loaded 'add_py'" / "'add_onnx'" and "Started HTTPService"
```

In a second terminal, send inference:

```bash
curl -s localhost:8000/v2/health/ready -o /dev/null -w "ready: %{http_code}\n"

for m in add_py add_onnx; do echo "== $m =="; curl -s localhost:8000/v2/models/$m/infer \
  -H 'Content-Type: application/json' -d '{
  "inputs":[
    {"name":"INPUT0","shape":[4],"datatype":"FP32","data":[1,2,3,4]},
    {"name":"INPUT1","shape":[4],"datatype":"FP32","data":[10,20,30,40]}]}'; echo; done
```

Expected: `ready: 200`, and both models return `OUTPUT0 = [11, 22, 33, 44]`. A correct
`add_onnx` result means the `onnxruntime` backend serves correctly on the reconstructed public base. Stop the server with `Ctrl‑C`.

### PyTorch backend

The `pytorch` backend is verified **separately, in its own model repo** (`models_torch/`, GPU). First complete the serving image: [`Dockerfile.pytorch-runtime.rhel`](Dockerfile.pytorch-runtime.rhel) adds `torch` into the pyenv Python the `python` backend already provisioned:

```bash
docker build -f Dockerfile.pytorch-runtime.rhel -t tritonserver-pytorch:example .
```

Add a TorchScript `OUTPUT__0 = INPUT__0 + INPUT__1` model (the PyTorch backend uses the
`INPUT__N` / `OUTPUT__N` naming convention):

```bash
mkdir -p models_torch/add_torch/1
cat > models_torch/add_torch/config.pbtxt <<'EOF'
name: "add_torch"
backend: "pytorch"
max_batch_size: 0
input [
  { name: "INPUT__0", data_type: TYPE_FP32, dims: [4] },
  { name: "INPUT__1", data_type: TYPE_FP32, dims: [4] }
]
output [ { name: "OUTPUT__0", data_type: TYPE_FP32, dims: [4] } ]
instance_group [ { kind: KIND_GPU } ]
EOF
# script the model in the Step 2 image (it already has torch)
cat > /tmp/gen_pt.py <<'EOF'
import torch
class Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b
torch.jit.script(Add()).save("/models/add_torch/1/model.pt")
EOF
docker run --rm -v "$PWD/models_torch:/models" -v /tmp/gen_pt.py:/gen.py:ro \
  triton-manylinux-pytorch:example python /gen.py
```

Serve with the completed image (`KIND_GPU`, so a GPU is required) and infer. (Again, if `8000` is
taken, map a free host port — `-p 8080:8000`, then `localhost:8080`.)

```bash
docker run --rm --gpus all -p8000:8000 -v "$PWD/models_torch:/models" \
  tritonserver-pytorch:example tritonserver --model-repository=/models
# wait for: "successfully loaded 'add_torch'", then in a second terminal:

curl -s localhost:8000/v2/models/add_torch/infer -H 'Content-Type: application/json' -d '{
  "inputs":[
    {"name":"INPUT__0","shape":[4],"datatype":"FP32","data":[1,2,3,4]},
    {"name":"INPUT__1","shape":[4],"datatype":"FP32","data":[10,20,30,40]}]}'; echo
```

Expected: `OUTPUT__0 = [11, 22, 33, 44]` — the PyTorch backend, built entirely from public
sources, serving correct inference.

## Known differences from the released artifacts

This build is *equivalent*, not identical, to the official `manylinux` release:

- **Pin versions for parity.** `BASE_IMAGE` and `TENSORRT_VERSION` (Step 1) and
  `--version` / `--container-version` (Step 3) must all match the target release.
  Cross‑check against the release's artifact name
  (`…-cu132-cp312-manylinux_2_28-x86_64.zip`) and the framework support matrix.
- **cuDNN** comes from the public CUDA base image and may be a slightly newer patch than the
  release used; if you need an exact match, install a specific cuDNN RPM from the `cuda-rhel8`
  repo in `Dockerfile.base.rhel`.
- **Running the binary** requires EL8's `libssl.so.1.1`. Run it inside the built image /
  on an EL8 host, or bundle OpenSSL 1.1 alongside the executable.

For background on `build.py` and its options, see the server repo's
[build documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md).
