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
BUILD_MODELS=()

# Frameworks
declare -A FRAMEWORKS=(["DIFFUSION"]=1)
DEFAULT_FRAMEWORK=DIFFUSION

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
DOCKERFILE=${SOURCE_DIR}/docker/Dockerfile


# Base Images
BASE_IMAGE=nvcr.io/nvidia/tritonserver
BASE_IMAGE_TAG_DIFFUSION=24.01-py3

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
	--build-models)
	    if [ "$2" ]; then
                BUILD_MODELS+=("$2")
                shift
            else
		BUILD_MODELS+=("all")
            fi
            ;;
        --base)
            if [ "$2" ]; then
                BASE_IMAGE=$2
                shift
            else
                error 'ERROR: "--base" requires an argument.'
            fi
            ;;
	--base-image-tag)
            if [ "$2" ]; then
                BASE_IMAGE_TAG=$2
                shift
            else
                error 'ERROR: "--base" requires an argument.'
            fi
            ;;
        --build-arg)
            if [ "$2" ]; then
                BUILD_ARGS+="--build-arg $2 "
                shift
            else
                error 'ERROR: "--build-arg" requires an argument.'
            fi
            ;;
        --tag)
            if [ "$2" ]; then
                TAG=$2
                shift
            else
                error 'ERROR: "--tag" requires an argument.'
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
	--no-cache)
	    NO_CACHE=" --no-cache"
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
	if [ -z $BASE_IMAGE_TAG ]; then
	    BASE_IMAGE_TAG=BASE_IMAGE_TAG_${FRAMEWORK}
	    BASE_IMAGE_TAG=${!BASE_IMAGE_TAG}
	fi
    fi

    if [ -z "$TAG" ]; then
        TAG="tritonserver:r24.01"

	if [[ $FRAMEWORK == "DIFFUSION" ]]; then
	    TAG+="-diffusion"
	fi

    fi

}


show_image_options() {
    echo ""
    echo "Building Triton Inference Server Image: '${TAG}'"
    echo ""
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base_Image_Tag: '${BASE_IMAGE_TAG}'"
    echo "   Build Context: '${SOURCE_DIR}'"
    echo "   Build Options: '${BUILD_OPTIONS}'"
    echo "   Build Arguments: '${BUILD_ARGS}'"
    echo "   Framework: '${FRAMEWORK}'"
    echo ""
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base base image]"
    echo "  [--base-imge-tag base image tag]"
    echo "  [--framework framework one of ${!FRAMEWORKS[@]}]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--tag tag for image]"
    echo "  [--dry-run print docker commands without running]"
    exit 0
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

# BUILD RUN TIME IMAGE

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG --build-arg FRAMEWORK=$FRAMEWORK "

if [ ! -z ${GITHUB_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} "
fi

if [ ! -z ${HF_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg HF_TOKEN=${HF_TOKEN} "
fi

show_image_options

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

$RUN_PREFIX docker build -f $DOCKERFILE $BUILD_OPTIONS $BUILD_ARGS -t $TAG $SOURCE_DIR $NO_CACHE

{ set +x; } 2>/dev/null


if [[ $FRAMEWORK == DIFFUSION ]]; then
    if [ -z "$RUN_PREFIX" ]; then
	set -x
    fi
    $RUN_PREFIX mkdir -p $PWD/backend/diffusion
    $RUN_PREFIX docker run --rm -it -v $PWD:/workspace $TAG /bin/bash -c "cp -rf /tmp/TensorRT/demo/Diffusion /workspace/backend/diffusion"

    { set +x; } 2>/dev/null

    for model in "${BUILD_MODELS[@]}"
    do
	if [ -z "$RUN_PREFIX" ]; then
	    set -x
	fi

	$RUN_PREFIX docker run --rm -it -v $PWD:/workspace $TAG /bin/bash -c "/workspace/scripts/build_models.sh --model $model"

	{ set +x; } 2>/dev/null
    done
fi



