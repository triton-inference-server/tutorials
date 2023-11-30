ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=23.10-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as triton-ray

RUN pip install -U ray[all]==2.7
RUN pip install /opt/tritonserver/python/tritonserver-2.39.0-py3-none-any.whl || true
RUN pip install pytest
RUN pip install cupy-cuda12x