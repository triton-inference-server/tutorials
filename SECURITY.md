<!--
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Security Policy

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.

To report a potential security vulnerability in any NVIDIA product, use one of
the following channels:

* **NVIDIA Vulnerability Disclosure Program** (preferred):
  [https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/)
* **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt
  sensitive reports with NVIDIA's
  [PGP key](https://www.nvidia.com/en-us/security/pgp-key).
* **GitHub Private Vulnerability Reporting (where enabled):** use the "Report a vulnerability"
  button on this repository's Security tab.

OEM partners should contact their NVIDIA Customer Program Manager.

Please include:

1. Product or repository name and the version, branch or commit affected
2. Type of vulnerability (for example code execution, denial of service,
   information disclosure)
3. Step-by-step instructions to reproduce the issue
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit the issue

NVIDIA PSIRT acknowledges reports, assesses them, and coordinates remediation
and disclosure with the reporter. Past advisories are published at
[https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/).

## Security Architecture and Context

**What this repository is.** Triton Tutorials is a collection of guides and
example code for the Triton Inference Server: Python and shell scripts, model
configurations (`config.pbtxt`), Dockerfiles, Kubernetes and Helm manifests,
and Markdown documentation. It is a **documentation and sample-code
repository**, not a shipped product. Nothing in it is a supported production
artifact.

**Classification:** SDK / sample code (examples and reference deployments).

**Primary security responsibility:** examples should not teach insecure
defaults, and sample code, container build files and scripts should not
contain vulnerabilities or embed credentials.

**Key interfaces and boundaries** (exposed by the examples when a user runs
them, not by this repository itself):

* Triton Inference Server HTTP (8000), gRPC (8001) and metrics (8002) endpoints
  started by the guides.
* Example Gradio web client (`Conceptual_Guide/Part_6-building_complex_pipelines/gui`).
* Example Ray Serve and Kafka integrations under
  `Triton_Inference_Server_Python_API/examples`.
* Docker build and run scripts (`build.sh`, `run.sh`) under
  `Popular_Models_Guide/StableDiffusion` and `Triton_Inference_Server_Python_API`.
* Kubernetes and Helm deployments under `Deployment/Kubernetes`.
* Model artifacts and weights downloaded from third-party hubs at run time.

**Repository Exposure Classification:** Public. Basis: the repository is
publicly visible on GitHub.

**Service Exposure Classification:** Internal-Isolated (low confidence). Basis:
the repository ships no running service; deployments are created by users in
their own environments from the examples.

## Threat Model

1. **Untrusted model artifacts executed at load time.** Example scripts such as
   `Conceptual_Guide/Part_5-Model_Ensembles/utils/export_text_recognition.py`
   call `torch.load` on downloaded checkpoints, and the guides download models
   from public hubs. A tampered or malicious checkpoint can execute arbitrary
   code on the machine running the export step.
2. **Unauthenticated, unencrypted inference endpoints.** The guides start
   Triton listening on all interfaces (HTTP, gRPC, metrics) without TLS or
   authentication, and some `run.sh` scripts use `docker run --network host`.
   Anyone who can reach the host can submit inference requests, read metrics
   and model metadata, or exhaust GPU capacity.
3. **Credential exposure through build and run scripts.** The Stable Diffusion
   and Python API `build.sh` scripts pass `HF_TOKEN` as a Docker `--build-arg`,
   which can persist in image history, and `run.sh` passes it as a container
   environment variable. Images built this way and pushed to a registry can
   leak the token.
4. **Exposed management and UI surfaces.** The Gradio example client binds to
   `0.0.0.0` and the Ray example starts the Ray head node with its dashboard
   on `0.0.0.0`. On a shared network these expose a web UI and a cluster
   control plane without authentication.
5. **Supply-chain risk in example dependencies and containers.** Dockerfiles,
   `requirements.txt` files and guides pull packages, base images and
   third-party tooling (for example `eksctl`, `huggingface-cli`, model
   weights) that are not all pinned to digests or hashes. A compromised
   upstream package or image affects anyone following the guide.
6. **Over-privileged sample deployments.** Kubernetes and Helm examples under
   `Deployment/Kubernetes` and multi-node launch scripts that start the
   server as a subprocess may run with broad container privileges and
   cluster-wide defaults that are unsuitable for production.

## Critical Security Assumptions

* **Examples are for evaluation, not production.** Users are expected to add
  TLS, authentication and authorization, network policy, and resource limits
  before exposing any deployment built from these guides.
* **A trusted network is assumed.** Example endpoints (Triton, Gradio, Ray,
  Kafka) have no authentication and are assumed to run on an isolated
  development network or behind an authenticating proxy.
* **Model artifacts are assumed trusted.** Checkpoints and weights fetched
  from public hubs are loaded without signature or hash verification; users
  must verify provenance, prefer safe serialization formats, and avoid loading
  untrusted pickled files.
* **Secrets are supplied by the user.** Tokens such as `HF_TOKEN` are expected
  to come from the user's environment or a secret manager, are never committed
  to the repository, and should not be baked into built images.
* **Dependencies are the user's responsibility to update.** Pinned versions in
  examples reflect the time of writing and may contain known vulnerabilities.
  Users should rebuild with current, scanned base images and packages.
* **The host and container runtime are trusted.** The examples assume the
  Docker daemon, GPU drivers and Kubernetes cluster are correctly configured
  and isolated.
