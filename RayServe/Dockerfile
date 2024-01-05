ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=23.12-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as rayserve-triton
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --timeout=2000 -r /tmp/requirements.txt
COPY ./requirements_torch.txt /tmp/requirements_torch.txt
RUN pip install --timeout=2000 -r /tmp/requirements_torch.txt
RUN apt-get update; apt-get install -y gdb
RUN pyright --help
COPY ./deps /tmp/deps
COPY . /workspace
RUN pip install --force-reinstall /tmp/deps/tritonserver-2.42.0.dev0-py3-none-any.whl
# COPY ./deps/libtritonserver.so /opt/tritonserver/lib/libtritonserver.so
# COPY ./deps/backends/python /opt/tritonserver/backends/python
COPY ./models_test /workspace/models
RUN ln -sf /bin/bash /bin/sh
# RUN pytest /tmp/deps
