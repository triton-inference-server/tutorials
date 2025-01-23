<!--
# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Multi-Node Triton + TRT-LLM Deployment on EKS

This repository provides instructions for multi-node deployment of LLMs on EKS (Amazon Elastic Kubernetes Service). This includes instructions for building custom image to enable features like EFA, Helm chart and associated Python script. This deployment flow uses NVIDIA TensorRT-LLM as the inference engine and NVIDIA Triton Inference Server as the model server.

We have 1 pod per node, so the main challenge in deploying models that require multi-node is that one instance of the model spans multiple nodes hence multiple pods. Consequently, the atomic unit that needs to be ready before requests can be served, as well as the unit that needs to be scaled becomes group of pods. This example shows how to get around these problems and provides code to set up the following

 1. **LeaderWorkerSet for launching Triton+TRT-LLM on groups of pods:**  To launch Triton and TRT-LLM across nodes you use MPI to have one node launch TRT-LLM processes on all the nodes (including itself) that will make up one instance of the model. Doing this requires knowing the hostnames of all involved nodes. Consequently we need to spawn groups of pods and know which model instance group they belong to. To achieve this we use [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws/tree/main), which lets us create "megapods" that consist of a group of pods - one leader pod and a specified number of worker pods -  and provides pod labels identifying group membership. We configure the LeaderWorkerSet and launch Triton+TRT-LLM via MPI in [`deployment.yaml`](multinode_helm_chart/chart/templates/deployment.yaml) and [server.py](multinode_helm_chart/containers/server.py).
 2. **Gang Scheduling:** Gang scheduling simply means ensuring all pods that make up a model instance are ready before Triton+TRT-LLM is launched. We show how to use `kubessh` to achieve this in the `wait_for_workers` function of [server.py](multinode_helm_chart/containers/server.py).
 3. **Autoscaling:** By default the Horizontal Pod Autoscaler (HPA) scales individual pods, but LeaderWorkerSet makes it possible to scale each "megapod". However, since these are GPU workloads we don't want to use cpu and host memory usage for autoscaling. We show how to leverage the metrics Triton Server exposes through Prometheus and set up GPU utilization recording rules in [`triton-metrics_prometheus-rule.yaml`](multinode_helm_chart/triton-metrics_prometheus-rule.yaml). We also demonstrate how to properly set up PodMonitors and an HPA in [`pod-monitor.yaml`](multinode_helm_chart/chart/templates/pod-monitor.yaml) and [`hpa.yaml`](multinode_helm_chart/chart/templates/hpa.yaml) (the key is to only scrape metrics from the leader pods). Instructions for properly setting up Prometheus and exposing GPU metrics are found in [Configure EKS Cluster and Install Dependencies](https://github.com/Wenhan-Tan/EKS_Multinode_Triton_TRTLLM/blob/main/Cluster_Setup_Steps.md). To enable deployment to dynamically add more nodes in response to HPA, we also setup [Cluster Autoscaler](https://github.com/Wenhan-Tan/EKS_Multinode_Triton_TRTLLM/blob/main/Cluster_Setup_Steps.md#10-install-cluster-autoscaler)
 4. **LoadBalancer Setup:** Although there are multiple pods in each instance of the model, only one pod within each group accepts requests. We show how to correctly set up a LoadBalancer Service to allow external clients to submit requests in [`service.yaml`](multinode_helm_chart/chart/templates/service.yaml)


## Setup and Installation

 1. [Create EKS Cluster](1.%20Create_EKS_Cluster.md)
 2. [Configure EKS Cluster](2.%20Configure_EKS_Cluster.md)
 3. [Deploy Triton](3.%20Deploy_Triton.md)
