ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=23.11-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as triton-ray
RUN pip install -U ray[all]==2.8
# RUN pip install /opt/tritonserver/python/tritonserver-2.39.0-py3-none-any.whl || true
RUN pip install pytest
RUN pip install cupy-cuda12x
COPY ./deps /tmp/deps
RUN ls -al /tmp
RUN pip install --force-reinstall /tmp/deps/tritonserver-2.41.0.dev0-py3-none-any.whl
RUN pip install awscli
COPY . /workspace
COPY ./models_test /workspace/models
RUN pytest /tmp/deps
