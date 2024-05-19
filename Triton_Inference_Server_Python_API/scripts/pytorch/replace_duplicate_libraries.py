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

import hashlib
import os
import pathlib
import shutil

backends = ["pytorch"]
exclude = ["model.py"]

source_path = pathlib.Path("/usr")


def hash_file(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_candidate(file_path):
    candidates = list(source_path.rglob(file_path.name))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    file_hash = hash_file(file_path)
    for candidate in candidates:
        candidate_file_hash = hash_file(candidate)
        if file_hash == candidate_file_hash:
            return candidate


if __name__ == "__main__":
    for backend in backends:
        backend_path = pathlib.Path(f"/opt/tritonserver/backends/{backend}")

        for file_path in backend_path.glob("*"):
            if file_path.name in exclude:
                continue
            candidate = get_candidate(file_path)

            if candidate:
                print(f"replacing {file_path} with symlink to {candidate}")
                file_path.unlink()
                file_path = pathlib.Path(file_path)
                file_path.symlink_to(candidate)


#            print(file_path.name)
#           print(file_path.stat())
#          print(hash_file(file_path))

#  files = os.listdir("/opt/tritonserver/backends/pytorch")
# print(files)

# print(list(pathlib.Path("/opt/tritonserver/backends/pytorch").glob("*")))

# print(list(pathlib.Path("/usr").rglob(files[1])))
