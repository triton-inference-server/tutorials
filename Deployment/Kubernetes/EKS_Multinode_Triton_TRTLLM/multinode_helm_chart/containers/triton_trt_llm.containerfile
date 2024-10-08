# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_CONTAINER_IMAGE=nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
FROM ${BASE_CONTAINER_IMAGE}

ENV EFA_INSTALLER_VERSION=1.33.0
ENV AWS_OFI_NCCL_VERSION=1.9.2-aws
ENV NCCL_VERSION=2.21.5-1
ENV NCCL_TESTS_VERSION=2.13.9

# Set a set of useful labels.
LABEL "base"="${BASE_CONTAINER_IMAGE}"
LABEL "role"="server"

# Stop APT (Debian package manager) from complaining about interactivity.
ENV DEBIAN_FRONTEND=noninteractive
# Set additional environment values that make usage more pleasant.
ENV TERM=xterm-256color

RUN apt update \
 && apt install --yes \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    cgroup-tools \
 && rm -rf /var/lib/apt/lists/*

# Install kubectl because server.py script depends on it.
# Step 1: acquire the Kubernetes APT GPG key.
RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key \
  | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg \
 && chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

# Step 2: Acquire the API sources list for Kubernetes.
RUN echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' \
  | tee /etc/apt/sources.list.d/kubernetes.list \
 && chmod 644 /etc/apt/sources.list.d/kubernetes.list

# Step 3: Install kubectl.
RUN apt update \
 && apt install --yes \
    kubectl \
 && apt autoremove --yes \
 && apt purge --yes \
 && rm -rf /var/lib/apt/lists/*

###############################################
#### For EFA and NCCL Test
RUN apt-get update -y
RUN apt-get remove -y --allow-change-held-packages \
                      libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1

RUN rm -rf /opt/hpcx/ompi \
    && rm -rf /usr/local/mpi \
    && rm -fr /opt/hpcx/nccl_rdma_sharp_plugin \
    && ldconfig
ENV OPAL_PREFIX=
RUN apt-get install -y --allow-unauthenticated \
    git \
    gcc \
    vim \
    kmod \
    openssh-client \
    openssh-server \
    build-essential \
    curl \
    autoconf \
    libtool \
    gdb \
    automake \
    cmake \
    apt-utils \
    libhwloc-dev \
    aptitude && \
    apt autoremove -y

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# NCCL
RUN apt-get remove -y libnccl2 libnccl-dev \
   && cd /tmp \
   && git clone https://github.com/NVIDIA/nccl.git -b v${NCCL_VERSION} \
   && cd nccl \
   && make -j src.build BUILDDIR=/usr/local \
   NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90" \
   && rm -rf /tmp/nccl

# EFA
RUN apt-get update && \
    apt-get install -y libhwloc-dev && \
    cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz  && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*

# NCCL EFA Plugin
RUN mkdir -p /tmp && \
    cd /tmp && \
    curl -LO https://github.com/aws/aws-ofi-nccl/archive/refs/tags/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    tar -xzf /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    rm /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    mv aws-ofi-nccl-${AWS_OFI_NCCL_VERSION} aws-ofi-nccl && \
    cd /tmp/aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure --prefix=/opt/amazon/efa \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
        --with-mpi=/opt/amazon/openmpi && \
    make -j$(nproc) install && \
    rm -rf /tmp/aws-ofi/nccl

# NCCL ENV setup
RUN echo "/usr/local/lib"      >> /etc/ld.so.conf.d/local.conf && \
    echo "/opt/amazon/openmpi/lib" >> /etc/ld.so.conf.d/efa.conf && \
    ldconfig

ENV OMPI_MCA_pml=^cm,ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0 \
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo

# NCCL-tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && git checkout v${NCCL_TESTS_VERSION} \
    && make MPI=1 \
    MPI_HOME=/opt/amazon/openmpi \
    CUDA_HOME=/usr/local/cuda \
    # nvcc to target p5 and p4 instances
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90"

###############################################

# Set the active working directory.
WORKDIR /workspace

# Copy kubessh script w/ executable permissions for everyone.
# This enables the script to be executed no matter the user the container is run as.
# This works around the issue of the file being non-executable when the container is build on a Windows host.
COPY --chmod=555 kubessh .
COPY server.py .

RUN apt list --installed \
 && pip list --version

ENTRYPOINT [ "/bin/bash" ]
