# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_CONTAINER_IMAGE=ubuntu:jammy

FROM ${BASE_CONTAINER_IMAGE}

# Set a set of useful labels.
LABEL "base"="${BASE_CONTAINER_IMAGE}"
LABEL "role"="client"

# Stop APT (Debian package manager) from complaining about interactivity.
ENV DEBIAN_FRONTEND=noninteractive
# Set additional environment values that make usage more pleasant.
ENV TERM=xterm-256color

RUN apt update \
 && apt install --fix-missing --no-install-recommends --yes \
    ca-certificates \
    wget \
    apt-transport-https \
    software-properties-common \
    python3 \
    python3-pip \
    icu-devtools \
    curl \
    git \
 && apt autoremove --yes \
 && apt purge --yes \
 && rm -rf /var/lib/apt/lists/*

COPY client.py .

ENTRYPOINT [ "/bin/bash" ]
