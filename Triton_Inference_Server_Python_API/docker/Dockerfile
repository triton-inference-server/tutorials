ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=23.12-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as triton-python-api

RUN apt-get update; apt-get install -y gdb

COPY ./deps/requirements.txt /tmp/requirements.txt

RUN pip install --timeout=2000 -r /tmp/requirements.txt

# Finish pyright install

 RUN pyright --help

ARG FRAMEWORK=HF_DIFFUSERS

COPY ./deps/requirements_torch.txt /tmp/requirements_torch.txt
RUN if [[ "$FRAMEWORK" == "HF_DIFFUSERS" ]] ; then pip install --timeout=2000 -r /tmp/requirements_torch.txt ; fi

COPY ./deps/tritonserver-2.41.0.dev0-py3-none-any.whl /tmp/tritonserver-2.41.0.dev0-py3-none-any.whl

RUN pip3 install --force-reinstall --upgrade /tmp/tritonserver-2.41.0.dev0-py3-none-any.whl[all]

RUN ln -sf /bin/bash /bin/sh

COPY . /workspace

ARG RUN_TESTS=FALSE

RUN if [[ "$RUN_TESTS" == "TRUE" ]] ; then pytest /workspace/deps ; fi

