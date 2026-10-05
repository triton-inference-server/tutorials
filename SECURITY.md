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

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

To report a potential security vulnerability, please use one of the following channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred): https://www.nvidia.com/en-us/security/
2. **Web form:** [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
3. **Email:** [NVIDIA PSIRT](mailto:psirt@nvidia.com). Please encrypt sensitive reports with NVIDIA's [PGP key](https://www.nvidia.com/en-us/security/pgp-key).
4. **GitHub Private Vulnerability Reporting (where enabled):** use the "Report a vulnerability" button on the Security tab of this repository.

**Do not open a public issue or pull request to report a vulnerability.**

Please include:

* Product or component name and version or branch
* Type of vulnerability
* Steps to reproduce
* Proof of concept, if available
* Potential impact and how it could be exploited

See https://www.nvidia.com/en-us/security/ for past NVIDIA Security Bulletins and Notices.

## Security Architecture and Context

**Project:** Tutorials and examples for Triton Inference Server.

**Software type:** Examples and bundled third-party source packages.

**Security boundaries:** The main security boundary is between this code and the environments, credentials and networks where it is built or run.

**Repository Exposure Classification:** Public.

**Service Exposure Classification:** Deployment-dependent. Exposure depends on how the software is deployed and configured by the operator.

## Threat Model

1. **Not hardened:** Examples and modified third-party sources are for development and reference, and may omit production security controls.
2. **Vulnerable or outdated dependencies:** Bundled or referenced third-party code may contain known vulnerabilities or lag behind upstream fixes.
3. **Supply chain:** Sources and models fetched at build or run time may be tampered with or unpinned.
4. **Exposure by default:** Example deployments may expose services without authentication or encryption.
5. **Credentials and sensitive data:** Credentials and data handled by examples may leak through logs, environment variables or build artifacts.

## Critical Security Assumptions

* The code is used for development and evaluation, and is reviewed before any production use.
* Deployers add authentication, authorization and TLS before exposing services.
* Dependencies are kept up to date and obtained from trusted sources.
* Credentials used with the examples are protected and rotated.
* Host operating system, driver and hardware security are the operator's responsibility.
