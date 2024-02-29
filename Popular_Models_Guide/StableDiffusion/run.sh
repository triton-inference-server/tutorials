#!/bin/bash -e
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

TAG=
RUN_PREFIX=

# Frameworks
declare -A FRAMEWORKS=(["DIFFUSION"]=1)
DEFAULT_FRAMEWORK=DIFFUSION

SOURCE_DIR=$(dirname "$(readlink -f "$0")")

# Base Images
IMAGE=
IMAGE_TAG_DIFFUSERS=diffusion

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
	--framework)
            if [ "$2" ]; then
                FRAMEWORK=$2
                shift
            else
                error 'ERROR: "--framework" requires an argument.'
            fi
            ;;
        --image)
            if [ "$2" ]; then
                BASE_IMAGE=$2
                shift
            else
                error 'ERROR: "--base" requires an argument.'
            fi
            ;;
        --dry-run)
            RUN_PREFIX="echo"
            echo ""
            echo "=============================="
            echo "DRY RUN: COMMANDS PRINTED ONLY"
            echo "=============================="
            echo ""
            ;;
        --)
            shift
            break
            ;;
         -?*)
	    error 'ERROR: Unknown option: ' $1
            ;;
	 ?*)
	    error 'ERROR: Unknown option: ' $1
            ;;
        *)
            break
            ;;
        esac

        shift
    done

    if [ -z "$FRAMEWORK" ]; then
	FRAMEWORK=$DEFAULT_FRAMEWORK
    fi

    if [ ! -z "$FRAMEWORK" ]; then
	FRAMEWORK=${FRAMEWORK^^}
	if [[ ! -n "${FRAMEWORKS[$FRAMEWORK]}" ]]; then
	    error 'ERROR: Unknown framework: ' $FRAMEWORK
	fi
    fi

    if [ -z "$IMAGE" ]; then
        IMAGE="tritonserver:r24.01"

	if [[ $FRAMEWORK == "DIFFUSION" ]]; then
	    IMAGE+="-diffusion"
	fi
    fi

}

show_help() {
    echo "usage: run.sh"
    echo "  [--image image]"
    echo "  [--framework framework one of ${!FRAMEWORKS[@]}]"
    echo "  [--dry-run print docker commands without running]"
    exit 0
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

# RUN the image

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

$RUN_PREFIX docker run --gpus all -it --rm --network host --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 -eHF_TOKEN -eGITHUB_TOKEN -eAWS_DEFAULT_REGION -eAWS_ACCESS_KEY_ID -eAWS_SECRET_ACCESS_KEY -eS3_BUCKET_URL -v/tmp:/tmp -v ${SOURCE_DIR}:/workspace -v${SOURCE_DIR}/.cache/huggingface:/root/.cache/huggingface -v${SOURCE_DIR}/backend/diffusion:/opt/tritonserver/backends/diffusion -w /workspace  $IMAGE

{ set +x; } 2>/dev/null


