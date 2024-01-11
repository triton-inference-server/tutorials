ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=23.12-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as rayserve-triton

ARG FRAMEWORK=HF_DIFFUSERS
ARG RUN_TESTS=FALSE


RUN apt-get update; apt-get install -y gdb

COPY ./deps/requirements.txt /tmp/requirements.txt

RUN pip install --timeout=2000 -r /tmp/requirements.txt

# Finish pyright install

RUN pyright --help

COPY ./deps/requirements_torch.txt /tmp/requirements_torch.txt
RUN if [[ "$FRAMEWORK" == "HF_DIFFUSERS" ]] ; then pip install --timeout=2000 -r /tmp/requirements_torch.txt ; fi

COPY . /workspace

RUN /workspace/deps -maxdepth 1 -type f -name \
    "tritonserver-*.whl" | xargs pip3 install --force-reinstall --upgrade

RUN ln -sf /bin/bash /bin/sh

RUN if [[ "$RUN_TESTS" == "TRUE"]] ; pytest /workspace/deps; fi

