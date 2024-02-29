<!--
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
-->

# Client Command Line Options

```bash
python3 client.py --help
usage: client.py [-h] [--clients CLIENTS] [--requests REQUESTS] [--static-batch-size STATIC_BATCH_SIZE] [--prompt PROMPT] [--save-image] [--launch-nvidia-smi] [--model MODEL]

Example client demonstrating sending prompts to generative AI models

options:
  -h, --help            show this help message and exit
  --clients CLIENTS     Number of concurrent clients. Each client sends --requests number of requests. (default: 1)
  --requests REQUESTS   Number of requests to send. (default: 1)
  --static-batch-size STATIC_BATCH_SIZE
                        Number of prompts to send in a single request (default: 1)
  --prompt PROMPT       Prompt. All requests and batches will use the same prompt (default: skeleton sitting by the side of a river looking soulful, concert poster, 4k, artistic)
  --save-image          If provided, generated images will be saved as jpeg files (default: False)
  --launch-nvidia-smi   Launch nvidia smi in daemon mode and log data to nvidia_smi_output.txt (default: False)
  --model MODEL         model name (default: stable_diffusion_xl)
```

