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

NVIDIA publishes prebuilt `manylinux_2_34` (RHEL 9‑compatible) Triton Inference Server
artifacts. Reproducing them ourselves with `build.py --target-platform=rhel` requires a
manylinux CUDA/cuDNN/TensorRT base image that is not published, so this tutorial reconstructs
an equivalent one from public sources — pypa's `manylinux_2_34` image (AlmaLinux 9) plus the
CUDA toolkit, cuDNN and TensorRT from NVIDIA's public `cuda-rhel9` repo — and walks through
building and running a
RHEL/manylinux Triton server end‑to‑end.

By the end of this tutorial, we will produce the following:

1. **The manylinux artifacts.** `build/install/` holds the `tritonserver` / `tritonfrontend`
   wheels tagged `…-cp312-cp312-manylinux_2_34_x86_64.whl` plus the backend trees we build (e.g. `onnxruntime`, `pytorch`, `python`) — all from 100% public inputs.
2. **A working, provably-manylinux container.** The built image serves real inference, and both
   the wheels and the runtime are verified to conform to `manylinux_2_34` (glibc 2.34 / EL9),
   so it runs on RHEL 9 and derivatives (Rocky, AlmaLinux, …).

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
> remove `tensorrt-devel` from `Dockerfile.base.rhel`.

### Choose a Triton version

These variables build `main`; all steps below use them. To build a release, set `TRITON_REF` to
its branch and take the other values from the table.

```bash
export TRITON_REF=main                  # or a release branch from the table
export CUDA_VERSION=13.4.1
export CUDNN_VERSION=9.25.1.1
export TENSORRT_VERSION=11.2.1.2-1.cuda13.3
export TORCH_VERSION=2.14.0             # used by Steps 2 and 4
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
export PYTORCH_BACKEND_CPYTHON_DIR=cpython-3.12.13
```

Known-good versions per release:

| Release | `TRITON_REF` | `CUDA_VERSION` | `CUDNN_VERSION` | `TENSORRT_VERSION` | `TORCH_VERSION` / `TORCH_INDEX_URL` | `PYTORCH_BACKEND_CPYTHON_DIR` |
|---|---|---|---|---|---|---|
| 26.08 | `r26.08` | `13.4.1` | `9.25.1.1` | `11.2.1.2-1.cuda13.3` | `2.14.0` / `cu132` | `cpython-3.12.13` |

> [!IMPORTANT]
> Releases **before 26.08** used a different, pyenv-based `rhel` build path — use the tutorial
> revision that shipped with that release rather than this one.

## Step 1: Build the public base image

On the `rhel` path, `build.py` installs its own build tooling and DCGM (NVIDIA's Data Center
GPU Manager) but expects the base image to already be a **manylinux** image with the
CUDA/cuDNN/TensorRT stack on it: since 26.08 it relies on pypa's `/opt/_internal` layout — the
CPython builds, their static `libpython` archive, and the
`pipx` "shared" venv it puts first on `PATH` and uses as `python3`/`pip3` for its tooling, the
Triton wheels and the python backend. [`Dockerfile.base.rhel`](Dockerfile.base.rhel) starts from
pypa's official `quay.io/pypa/manylinux_2_34_x86_64` image (AlmaLinux 9 = RHEL 9 / glibc 2.34),
adds gcc-toolset-13 (the compiler NVIDIA's 26.08 artifacts are built with; EL9's RapidJSON 1.1.0
doesn't compile under GCC 14), enables **EPEL + CRB** for the `-devel` packages `build.py` installs,
adds the **CUDA toolkit, cuDNN and TensorRT** from the public `cuda-rhel9` repo with the same
environment the `nvidia/cuda` images export, and recreates that shared venv from the image's
**CPython 3.12** — the interpreter Triton's RHEL build targets. (PyTorch's extra runtime
libraries — NCCL, cuSPARSELt — are *not* added here; the Step 4 completion image installs them,
so the base stays generic.)

Build it with the values from your release. TensorRT trails CUDA by a minor, which is why 26.08
pairs CUDA 13.4.1 with a TensorRT built against `cuda13.3`. The pypa manylinux image tag is
pinned in the Dockerfile; pass `--build-arg BASE_IMAGE=...` to change it:

```bash
docker build -f Dockerfile.base.rhel \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg CUDNN_VERSION="$CUDNN_VERSION" \
  --build-arg TENSORRT_VERSION="$TENSORRT_VERSION" \
  -t triton-manylinux-base:example .
```

## Step 2 (optional): Build the PyTorch backend image

Only needed if you build the `pytorch` backend.
[`Dockerfile.pytorch.rhel`](Dockerfile.pytorch.rhel) installs a public `torch` wheel into the
same `manylinux_2_34` image so the PyTorch backend can extract a libtorch that runs on EL9's
glibc 2.34 (the default Ubuntu-based `libtorch` is built against a newer glibc and won't load
there). The backend copies from a **hardcoded** `/opt/_internal/cpython-3.12.x` path
(`cpython-3.12.13` on r26.08); the Dockerfile symlinks that name to whatever 3.12 patch the
image actually ships.

```bash
docker build -f Dockerfile.pytorch.rhel \
  --build-arg TORCH_VERSION="$TORCH_VERSION" \
  --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  --build-arg PYTORCH_BACKEND_CPYTHON_DIR="$PYTORCH_BACKEND_CPYTHON_DIR" \
  -t triton-manylinux-pytorch:example .
```

The pytorch backend's CMake, which build.py runs inside the builder container, does an
unconditional `docker pull` of `--image=pytorch`; `--no-container-pull` only affects build.py's
own base image build. So the image must be reachable from a registry. Push it to a temporary
local one:

```bash
docker run -d -p 5000:5000 --name registry registry:2
docker tag  triton-manylinux-pytorch:example localhost:5000/triton-manylinux-pytorch:example
docker push localhost:5000/triton-manylinux-pytorch:example
```

`localhost:5000` is just a stand-in for NVIDIA's internal registry. Step 3's command already
includes the PyTorch flags; the built image can't serve PyTorch models until Step 4 completes it.

## Step 3: Build the server

Clone the server repo at the matching release branch and run `build.py` with your base
image:

```bash
git clone https://github.com/triton-inference-server/server.git
cd server && git checkout "$TRITON_REF"

./build.py -v --target-platform=rhel --no-container-pull \
  --image=base,triton-manylinux-base:example \
  --image=pytorch,localhost:5000/triton-manylinux-pytorch:example \
  --extra-core-cmake-arg=PYBIND11_FINDPYTHON=ON \
  --extra-core-cmake-arg=Python_LIBRARY=/opt/_internal/nolibpython/libpython3.12.a \
  --enable-gpu --enable-logging --enable-stats --enable-metrics \
  --enable-gpu-metrics --enable-cpu-metrics --enable-tracing \
  --endpoint=http --endpoint=grpc \
  --backend=onnxruntime --backend=pytorch --backend=python \
  --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
  --repoagent=checksum
```

Key flags:

- No `--version` / `--container-version` / `--upstream-container-version`, and no `:tag` on the
  `--backend` / `--repoagent` flags — build.py reads all of them from the checkout
  (`DEFAULT_TRITON_VERSION_MAP` and the branch), so this command is the same for every
  `TRITON_REF`.
- `--no-container-pull` — assuming your base image is local, this stops Docker from
  trying to pull it. It does **not** stop the pytorch backend's own pull of `--image=pytorch`
  — that's why Step 2 pushes to a local registry.
- If you're running headless (CI, `ssh` without a TTY, etc.), add `--no-container-interactive` —
  build.py launches the compile with `docker run -it` by default, which aborts with
  `the input device is not a TTY` when no terminal is attached.
- `--extra-core-cmake-arg=PYBIND11_FINDPYTHON=ON` — makes pybind11 use CMake's modern
  `FindPython`, so the Python 3.12 venv on `PATH` is what the `tritonserver` wheel is built
  against rather than whatever legacy discovery finds first.
- `--extra-core-cmake-arg=Python_LIBRARY=/opt/_internal/nolibpython/libpython3.12.a` — the core
  links libpython into its Python bindings module even though an extension module gets those
  symbols from the interpreter, and pypa's only libpython is a static archive that can't go into
  a shared library. This points CMake at an empty archive from the Step 1 image so nothing is
  linked, matching NVIDIA's released module.
- `--image=pytorch,localhost:5000/…` — the prebuilt PyTorch image from Step 2. The other backends
  (onnxruntime, tensorrt, python) compile from source during the build; PyTorch instead reuses a
  **prebuilt** libtorch (too heavy to build in-tree), which the build pulls and extracts from
  this image — the only backend that needs an `--image`.
- `TRITON_PYTORCH_ENABLE_TORCHVISION=OFF` — this example builds without torchvision;
  wiring torchvision up from public sources is an untested path in this tutorial.

The `python` backend is optional — drop `--backend=python` if you don't need it. `tensorrt` is
optional too (ONNX Runtime already pulls in its TensorRT provider); add `--backend=tensorrt`
for the standalone backend.

Triton's `common`, `core`, `backend`, and `third_party` repos don't need explicit
`--repo-tag` flags either — build.py defaults them, like the backend and repoagent tags
above, to the checked-out release branch (or `main` for a dev checkout).

ONNX Runtime is compiled from source here (~2 hours — it builds CUDA kernels for several
GPU architectures).

The build produces the install tree under `build/install/` and a local `tritonserver`
Docker image.

## Step 4: Verify

**1. Manylinux artifact generation**

Now `ls` the artifacts to verify they were generated correctly:

```bash
ls build/install/backends                 # onnxruntime  pytorch  python
find build/install -name '*.whl'          # ...-cp312-cp312-manylinux_2_34_x86_64.whl
```

**2. Serve real workloads from Manylinux container**

Run the server and real inference **inside the built image** (AlmaLinux 9); the binary needs
glibc 2.34 and won't start on an EL8 host. First create a model repository with two
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
# ir_version 7 pairs with opset 13; onnx's default (its newest IR) can exceed what ORT accepts.
onnx.save(helper.make_model(g, ir_version=7, opset_imports=[helper.make_opsetid("", 13)]),
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
# ready: 200

for m in add_py add_onnx; do
  echo "== $m =="
  curl -s localhost:8000/v2/models/$m/infer -H 'Content-Type: application/json' -d '{
    "inputs": [
      {"name": "INPUT0", "shape": [4], "datatype": "FP32", "data": [1, 2, 3, 4]},
      {"name": "INPUT1", "shape": [4], "datatype": "FP32", "data": [10, 20, 30, 40]}
    ]
  }'
  echo
done
```

Expected output:

```
== add_py ==
{"model_name":"add_py","model_version":"1","outputs":[{"name":"OUTPUT0","datatype":"FP32","shape":[4],"data":[11.0,22.0,33.0,44.0]}]}
== add_onnx ==
{"model_name":"add_onnx","model_version":"1","outputs":[{"name":"OUTPUT0","datatype":"FP32","shape":[4],"data":[11.0,22.0,33.0,44.0]}]}
```

A correct `add_onnx` result means the `onnxruntime` backend serves on the reconstructed public
base. Stop the server with `Ctrl‑C`.

### PyTorch backend

The `pytorch` backend is verified **separately, in its own model repo** (`models_torch/`, GPU). First complete the serving image: [`Dockerfile.pytorch-runtime.rhel`](Dockerfile.pytorch-runtime.rhel) installs `torch` into the image's Python 3.12 plus the NCCL and cuSPARSELt libraries libtorch links:

```bash
cd <tutorials-repo>/Build_Guide/RHEL_Manylinux   # Step 3 left you in server/
docker build -f Dockerfile.pytorch-runtime.rhel \
  --build-arg TORCH_VERSION="$TORCH_VERSION" --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
  -t tritonserver-pytorch:example .
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
  "inputs": [
    {"name": "INPUT__0", "shape": [4], "datatype": "FP32", "data": [1, 2, 3, 4]},
    {"name": "INPUT__1", "shape": [4], "datatype": "FP32", "data": [10, 20, 30, 40]}
  ]
}'
echo
```

Expected output:

```
{"model_name":"add_torch","model_version":"1","outputs":[{"name":"OUTPUT__0","datatype":"FP32","shape":[4],"data":[11.0,22.0,33.0,44.0]}]}
```

The PyTorch backend, built entirely from public sources, is serving correct inference. Stop the
server with `Ctrl‑C`.

## Known differences from the released artifacts

This build is *equivalent*, not identical, to the official `manylinux` release:

- **Pin versions for parity.** The Dockerfile build-args must match the `TRITON_REF` you build —
  use the [Choose a Triton version](#choose-a-triton-version) table. Cross-check
  against the release's artifact name (`…-cp312-manylinux_2_34-x86_64.zip`).
- **Library patch levels** can differ slightly from the release: the public `cuda-rhel9` repo
  does not carry the release's exact cuDNN build (9.25.1.1 here vs 9.25.0.28), the CUDA 13.4
  packages install at the repo's current patch, and the pypa image ships CPython 3.12.14 vs
  3.12.13 (hence the Step 2 symlink).
- **The PyTorch backend uses the public torch wheel** (`2.14.0+cu132`) rather than NVIDIA's own
  PyTorch build, and is built without torchvision (Step 3).
- **RHEL 8 hosts can't run these artifacts.** `manylinux_2_34` needs glibc 2.34 (EL9); releases
  up to 26.07 shipped `manylinux_2_28`, which also ran on EL8.

For background on `build.py` and its options, see the server repo's
[build documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md).
