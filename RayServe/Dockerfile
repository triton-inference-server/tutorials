ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_IMAGE_TAG=23.09-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} as triton-ray

RUN pip install -U ray[all]
RUN pip install /opt/tritonserver/python/tritonserver-2.38.0-py3-none-any.whl