<!---
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
--->

# Multi-Node Generative AI w/ Triton Server and TensorRT-LLM

It almost goes without saying that large language models (LLM) are large.
LLMs often are too large to fit in the memory of a single GPU.
Therefore we need a solution which enables multiple GPUs to cooperate to enable inference serving for this very large models.

This guide aims to explain how to perform multi-GPU, multi-node deployment of large language models using Triton Server and
TRT-LLM in a Kubernetes cluster.
Setting up multi-node LLM support using Triton Inference Server, TensorRT-LLM, and Kubernetes is not difficult, but it does
require preparation.

We'll cover the following topics:

* [Cluster Setup](#cluster-setup)
  * [Persistent Volume Setup](#persistent-volume-setup)
  * [Core Cluster Services](#core-cluster-services)
    * [Kubernetes Node Feature Discovery service](#kubernetes-node-feature-discovery-service)
    * [NVIDIA Device Plugin for Kubernetes](#nvidia-device-plugin-for-kubernetes)
    * [NVIDIA GPU Feature Discovery service](#nvidia-gpu-feature-discovery-service)
  * [Hugging Face Authorization](#hugging-face-authorization)
* [Triton Preparation](#triton-preparation)
  * [Model Preparation Script](#model-preparation-script)
  * [Custom Container Image](#custom-container-image)
  * [Kubernetes Pull Secrets](#kubernetes-pull-secrets)
* [Triton Deployment](#triton-deployment)
  * [How It Works](#how-it-works)
  * [Potential Improvements](#potential-improvements)
    * [Autoscaling and Gang Scheduling](#autoscaling-and-gang-scheduling)
    * [Network Topology Aware Scheduling](#network-topology-aware-scheduling)
* [Developing this Guide](#developing-this-guide)

Prior to beginning this guide/tutorial you will need a couple of things.

* Kubernetes Control CLI (`kubectl`)
  [ [documentation](https://kubernetes.io/docs/reference/kubectl/introduction/)
  | [download](https://kubernetes.io/releases/download/) ]
* Helm CLI (`helm`)
  [ [documentation](https://helm.sh/)
  | [download](https://helm.sh/docs/intro/install) ]
* Docker CLI (`docker`)
  [ [documentation](https://docs.docker.com/)
  | [download](https://docs.docker.com/get-docker/) ]
* Decent text editing software for editing YAML files.
* Kubernetes cluster.
* Fully configured `kubectl` with administrator permissions to the cluster.



## Cluster Setup

The following instructions are setting up a Kubernetes cluster for the deployment of LLMs using Triton Server and TRT-LLM.


### Prerequisites

This guide assumes that all nodes with NVIDIA GPUs have the following:
- A node label of `nvidia.com/gpu=present` to more easily identify nodes with NVIDIA GPUs.
- A node taint of `nvidia.com/gpu=present:NoSchedule` to prevent non-GPU pods from being deployed to GPU nodes.

> [!Tip]
> When using a Kubernetes provider like AKS, EKA, or GKE, it is usually best to use their interface when configuring nodes
> instead of using `kubectl` to do it directly.


### Persistent Volume Setup

To enable multiple pods deployed to multiple nodes to load shards of the same model so that they can used in coordination to
serve inference request too large to loaded by a single GPU, we'll need a common, shared storage location.
In Kubernetes, these common, shared storage locations are referred to as persistent volumes.
Persistent volumes can be volume mapped in to any number of pods and then accessed by processes running inside of said pods
as if they were part of the pod's file system.

Additionally, we will need to create a persistent-volume claim which can use to assign the persistent volume to a pod.

Unfortunately, the creation of a persistent volume will depend on how your cluster is setup, and is outside the scope of this
tutorial.
That said, we will provide a basic overview of the process.

#### Create a Persistent Volume

If your cluster is hosted by a cloud service provider, (CSP) like Amazon (EKS), Azure (AKS), or gCloud (GKE)
step-by-step instructions are available online for how to setup a persistent volume for your cluster.
Otherwise, you will need to work with your cluster administrator or find a separate guide online on how to setup a
persistent volume for your cluster.

The following resources can assist with the setting up of persistent volumes for your cluster.

* [Kubernetes Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
* [AKS Persistent Volumes](https://learn.microsoft.com/en-us/azure/aks/azure-csi-disk-storage-provision)
* [EKS Persistent Volumes](https://aws.amazon.com/blogs/storage/persistent-storage-for-kubernetes/)
* [GKE Persistent Volumes](https://cloud.google.com/kubernetes-engine/docs/concepts/persistent-volumes)
* [OKE Persistent Volumes](https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengcreatingpersistentvolumeclaim.htm)

> [!Important]
> It is important to consider the storage requirements of the models you expect your cluster to host, and be sure to
> sufficiently size the persistent volume for the combined storage size of all models.

Below are some example values gathered from internal testing of this tutorial.

| Model           | Parallelism | Raw Size | Converted Size | Total Size |
| :-------------- | ----------: | -------: | -------------: | ---------: |
| **Llama-3-8B**  | 2           | 15Gi     | 32Gi           | 47Gi       |
| **Llama-3-8B**  | 4           | 15Gi     | 36Gi           | 51Gi       |
| **Llama-3-70B** | 8           | 90Gi     | 300Gi          | 390Gi      |

#### Create a Persistent-Volume Claim

In order to connect the Triton Server pods to the persistent volume created above, we need to create a persistent-volume
claim (PVC). You can use the [pvc.yaml](./pvc.yaml) file provided as part of this tutorial to create one.

> [!Important]
> The `volumeName` property must match the `metadata.name` property of the persistent volume created above.


### Core Cluster Services

Once all nodes are correctly labeled and tainted, use the following steps to prepare the cluster deploying large language
models across multiple nodes with Triton Server.

The following series of steps are intended to prepare a fresh cluster.
For clusters in varying states, it is best to coordinate with your cluster administrator before installing new services and
capabilities.

#### Kubernetes Node Feature Discovery service

1.  Add the Kubernetes Node Feature Discovery chart repository to the local cache.

    ```bash
    helm repo add kube-nfd https://kubernetes-sigs.github.io/node-feature-discovery/charts \
      && helm repo update
    ```

2.  Run the command below to install the service.

    ```bash
    helm install -n kube-system node-feature-discovery kube-nfd/node-feature-discovery \
      --set nameOverride=node-feature-discovery \
      --set worker.tolerations[0].key=nvidia.com/gpu \
      --set worker.tolerations[0].operator=Exists \
      --set worker.tolerations[0].effect=NoSchedule
    ```

    > [!Note]
    > The above command sets toleration values which allow for the deployment of a pod onto a node with
    > a matching taint.
    > See this document's [prerequisites](#prerequisites) for the taints this document expected to have been applied to GPU
    > nodes in the cluster.

#### NVIDIA Device Plugin for Kubernetes

1.  This step is unnecessary if the Device Plugin has already been installed in your cluster.
    Cloud provider turnkey Kubernetes clusters, such as those from AKS, EKS, and GKE, often have the Device Plugin
    automatically once a GPU node as been added to the cluster.

    To check if your cluster requires the NVIDIA Device Plugin for Kubernetes, run the following command and inspect
    the output for `nvidia-device-plugin-daemonset`.

    ```bash
    kubectl get daemonsets --all-namespaces
    ```

    Example output:
    ```text
    NAMESPACE     NAME         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
    kube-system   kube-proxy   6         6         6       6            6
    ```

2.  If `nvidia-device-plugin-daemonset` is not listed, run the command below to install the plugin.
    Once installed it will provide containers access to GPUs in your clusters.

    For additional information, see
    [Github/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin/blob/main/README.md).

    ```bash
    kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
    ```

#### NVIDIA GPU Feature Discovery Service

1.  This step is unnecessary if the service has already been installed in your cluster.

    To check if your cluster requires the NVIDIA Device Plugin for Kubernetes, run the following command and inspect
    the output for `nvidia-device-plugin-daemonset`.

    ```bash
    kubectl get daemonsets --all-namespaces
    ```

    Example output:
    ```text
    NAMESPACE     NAME                             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
    kube-system   kube-proxy                       6         6         6       6            6
    kube-system   nvidia-device-plugin-daemonset   6         6         6       6            6
    ```

2.  If `gpu-feature-discovery` is listed, skip this step and the next.

    Otherwise, use the YAML file below to install the GPU Feature Discovery service.

    > [nvidia_gpu-feature-discovery_daemonset.yaml](nvidia_gpu-feature-discovery_daemonset.yaml)

    The file above was created by downloading its contents from
    [GitHub/NVIDIA](https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.8.2/deployments/static/gpu-feature-discovery-daemonset.yaml)
    and modified specifically for this tutorial.

    ```bash
    curl https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.8.2/deployments/static/gpu-feature-discovery-daemonset.yaml \
      >  nvidia_gpu-feature-discovery_daemonset.yaml
    ```

3.  Then run the command below to install the

    ```bash
    kubectl apply -f ./nvidia_gpu-feature-discovery_daemonset.yaml
    ```


### Hugging Face Authorization

In order to download models from Hugging Face, your pods will require an access token with the appropriate permission to
download models from their servers.

1.  If you do not already have a Hugging Face access token, you will need to created one.
    To create a Hugging Face access token,
    [follow their guide](https://huggingface.co/docs/hub/en/security-tokens).

2.  Once you have a token, use the command below to persist the token as a secret named `hf-model-pull` in your cluster.

    ```bash
    kubectl create secret generic hf-model-pull '--from-literal=password=<access_token>'
    ```

3.  To verify that your secret has been created, use the following command and inspect the output for your secret.

    ```bash
    kubectl get secrets
    ```



## Triton Preparation


### Model Preparation Script

The intention of this script to handle the acquisition of the model file from Hugging Face, the generation of the TensorRT
engine and plan files, and the caching of said generated files.
The script depends on the fact that the Kubernetes deployment scripts we'll be using rely on the persistent volume backing the
persistent-volume claim provided as part of the Helm chart.

Specially, the model and engine directories will me mapped to folders in the persistent volume and remapped to all subsequent
pods deployed as part of the Helm chart.
This enables the generation script to detect that the plan and engine generation steps have been completed and not repeat work.

> [!Tip]
> This script will executed as a job every time the Helm chart is installed unless the `.model.skipConversion` property is
> set to `false`.

When Triton Server is started, the same persistent volume folders will be mounted to its container and Triton will use the
pre-generated model plan and engine files.
Not only does this enable pods on separate nodes to share the same model engine and plan files, it drastically reduces the time
required for subsequent pod starts on the same node.

> [!Note]
> You can look at the code used to acquire and convert the models in [containers/server.py](containers/server.py).
> This file is copied into the server container image (see below) during its creation and then executed when the conversion
> job pod is deployed.

#### Custom Container Image

1.  Using the file below, we'll create a custom container image in the next step.

    > [triton_trt-llm.containerfile](containers/triton_trt-llm.containerfile)

2.  Run the following command to create a custom Triton Inference Server w/ all necessary tools to generate TensorRT-LLM
    plan and engine files. In this example we'll use the tag `24.04` to match the date portion of `24.04-trtllm-python-py3`
    from the base image.

    ```bash
    docker build \
      --file ./triton_trt-llm.containerfile \
      --rm \
      --tag triton_trt-llm:24.04 \
      .
    ```

    ##### Custom Version of Triton CLI

    This custom Triton Server container image makes use of a custom version of the Triton CLI.
    The relevant changes have been made available as a
    [topic branch](https://github.com/triton-inference-server/triton_cli/tree/jwyman/aslb-mn) in the Triton CLI repository on
    GitHub.
    The changes in the branch can be
    [inspected](https://github.com/triton-inference-server/triton_cli/compare/main...jwyman/aslb-mn) using the GitHub
    interface, and primarily contain the addition of the ability to specify tensor parallelism when optimizing models for
    TensorRT-LLM and enable support for additional models.

3.  Upload the Container Image to a Cluster Visible Repository.

    In order for your Kubernetes cluster to be able to download out new container image, it will need to be pushed to a
    container image repository that nodes in your cluster can reach.
    In this example, we'll use the fictional `nvcr.io/example` repository for demonstration purposes.
    You will need to determine which repositories you have write access to that your cluster can also access.

    1. First, re-tag the container image with the repository's name like below.

        ```bash
        docker tag \
          triton_trt-llm:24.04 \
          nvcr.io/example/triton_trt-llm:24.04
        ```

    2. Next, upload the container image to your repository.

        ```bash
        docker push nvcr.io/example/triton_trt-llm:24.04
        ```

#### Kubernetes Pull Secrets

If your container image repository requires credentials to download images from, then you will need to create a Kubernetes
docker-registry secret.
We'll be using the `nvcr.io` container image repository example above for demonstration purposes.
Be sure to properly escape any special characters such as `$` in the password or username values.

1.  Use the command below to create the necessary secret.  Secrets for your repository should be similar, but not be identical
to the example below.

    ```bash
    kubectl create secret docker-registry ngc-container-pull \
      --docker-password='dGhpcyBpcyBub3QgYSByZWFsIHNlY3JldC4gaXQgaXMgb25seSBmb3IgZGVtb25zdHJhdGlvbiBwdXJwb3Nlcy4=' \
      --docker-server='nvcr.io' \
      --docker-username='\$oauthtoken'
    ```

2.  The above command will create a secret in your cluster named `ngc-container-pull`.
    You can verify that the secret was created correctly using the following command and inspecting its output for the secret
    you're looking for.

    ```bash
    kubectl get secrets
    ```

3.  Ensure the contents of the secret are correct, you can run the following command.

    ```bash
    kubectl get secret/ngc-container-pull -o yaml
    ```

    You should see an output similar to the following.

    ```yaml
    apiVersion: v1
    data:
      .dockerconfigjson: eyJhdXRocyI6eyJudmNyLmlvIjp7InVzZXJuYW1lIjoiJG9hdXRodG9rZW4iLCJwYXNzd29yZCI6IlZHaHBjeUJwY3lCdWIzUWdZU0J5WldGc0lITmxZM0psZEN3Z2FYUWdhWE1nYjI1c2VTQm1iM0lnWkdWdGIyNXpkSEpoZEdsdmJpQndkWEp3YjNObGN5ND0iLCJhdXRoIjoiSkc5aGRYUm9kRzlyWlc0NlZrZG9jR041UW5CamVVSjFZak5SWjFsVFFubGFWMFp6U1VoT2JGa3pTbXhrUTNkbllWaFJaMkZZVFdkaU1qVnpaVk5DYldJelNXZGFSMVowWWpJMWVtUklTbWhrUjJ4MlltbENkMlJZU25kaU0wNXNZM2swWjFWSGVHeFpXRTVzU1VjMWJHUnRWbmxKU0ZaNldsTkNRMWxZVG14T2FsRm5aRWM0WjJGSGJHdGFVMEo1V2xkR2MwbElUbXhaTTBwc1pFaE5hQT09In19fQ==
    kind: Secret
    metadata:
      name: ngc-container-pull
      namespace: default
    type: kubernetes.io/dockerconfigjson
    ```

    The value of `.dockerconfigjson` is a base-64 encoded string which can be decoded into the following.

    ```json
    {
      "auths": {
        "nvcr.io": {
          "username":"$oauthtoken",
          "password":"VGhpcyBpcyBub3QgYSByZWFsIHNlY3JldCwgaXQgaXMgb25seSBmb3IgZGVtb25zdHJhdGlvbiBwdXJwb3Nlcy4gUGxlYXNlIG5ldmVyIHVzZSBCYXNlNjQgdG8gaGlkZSByZWFsIHNlY3JldHMh",
          "auth":"JG9hdXRodG9rZW46VkdocGN5QnBjeUJ1YjNRZ1lTQnlaV0ZzSUhObFkzSmxkQ3dnYVhRZ2FYTWdiMjVzZVNCbWIzSWdaR1Z0YjI1emRISmhkR2x2YmlCd2RYSndiM05sY3k0Z1VHeGxZWE5sSUc1bGRtVnlJSFZ6WlNCQ1lYTmxOalFnZEc4Z2FHbGtaU0J5WldGc0lITmxZM0psZEhNaA=="
        }
      }
    }
    ```

    You can use this compact command line to get the above output with a single command.

    ```bash
    kubectl get secret/ngc-container-pull -o json | jq -r '.data[".dockerconfigjson"]' | base64 -d | jq
    ```

    > [!Note]
    > The values of `password` and `auth` are also base-64 encoded string.
    > We recommend inspecting the values of the following values:
    >
    > * Value of `.auths['nvcr.io'].username`.
    > * Base64 decoded value of `.auths['nvcr.io'].password`.
    > * Base64 decoded value of `.auths['nvcr.io'].auths`.



## Triton Deployment

> [!Note]
> Deploying Triton Server with a model that fits on a single GPU is straightforward but not explained by this guide.
> For instructions and examples of deploying a model using a single GPU or multiple GPUs on a single node, use the
> [Autoscaling and Load Balancing Generative AI w/ Triton Server and TensorRT-LLM Guide](../Kubernetes/TensorRT-LLM_Autoscaling_and_Load_Balancing/README.md) instead.

Given the memory requirements of some AI models it is not possible to host them using a single device.
Triton and TensorRT-LLM provide a mechanism to enable a large model to be hosted by multiple GPU devices working in concert.
The provided sample Helm [chart](./chart/) provides a mechanism for taking advantage of this capability.

To enable this feature, adjust the `model.tensorrtLlm.parallelism.tensor` value to an integer greater than 1.
Configuring a model to use tensor parallelism enables the TensorRT-LLM runtime to effectively combine the memory of multiple
GPUs to host a model too large to fit on a single GPU.

Similarly, changing the value of `model.tensorrtLlm.parallelism.pipeline` will enable pipeline parallelism.
Pipeline parallelism is used to combine the compute capacity of multiple GPUs to process inference requests in parallel.

> [!Important]
> The product of the values of `.tensor` and `.pipeline` should be a power of 2 greater than `0` and less than or equal to
> `32`.

The number of GPUs required to host the model is equal to product of the values of `.tensor` and `.pipeline`.
When the model is deployed, one pod per GPU required will be created.
The Helm chart will create a leader pod and one or more work pods, depending on the number of additional pods required to
host the model.
Additionally, a model conversion job will be created to download the model from Hugging Face and then convert the downloaded
model into TRT-LLM engin and plan files.
To disable the creation of a conversion job by the Helm chart, set the values file's `model.skipConversion` property to
`false`.

> [!Warning]
> If your cluster has insufficient resources to create the conversion job, the leader pod, and the required worker pods,
> and the job pod is not scheduled to execute first, it is possible for the example Helm chart to become "hung" due to the
> leader pod waiting on the job pod's completion and there being insufficient resources to schedule the job pod.
>
> If this occurs, it is best to delete the Helm installation and retry until the job pod is successfully scheduled.
> Once the job pod completes, it will release its resources and make them available for the other pods to start.

### Deploying Single GPU Models

Deploying Triton Server with a model that fits on a single GPU is straightforward using the steps below.

1.  Create a custom values file with required values:

    * Container image name.
    * Model name.
    * Supported / available GPU.
    * Image pull secrets (if necessary).
    * Hugging Face secret name.

    The provided sample Helm [chart](./chart/) include several example values files such as
    [llama-3-8b_values.yaml](chart/llama-3-8b-instruct_values.yaml).

2.  Deploy LLM on Triton + TRT-LLM.

    Apply the custom values file to override the exported base values file using the command below, and create the Triton
    Server Kubernetes deployment.

    > [!Tip]
    > The order that the values files are specified on the command line is important with values are applied and
    > override existing values in the order they are specified.

    ```bash
    helm install <installation_name> \
      --values ./chart/values.yaml \
      --values ./chart/<custom_values>.yaml \
      --set 'triton.image.name=<custom_image_name>' \
      ./chart/.
    ```

    > [!Important]
    > Be sure to substitute the correct values for `<installation_name>` and `<custom_values>` in the example above.

3.  Verify the Chart Installation.

    Use the following commands to inspect the installed chart and to determine if everything is working as intended.

    ```bash
    kubectl get deployments,pods,services,jobs --selector='app=<installation_name>'
    ```

    > [!Important]
    > Be sure to substitute the correct value for `<installation_name>` in the example above.

    You should output similar to below (assuming the installation name of "llama-3"):

    ```text
    NAME                      READY   UP-TO-DATE   AVAILABLE
    deployment.apps/llama-3   0/1     1            0

    NAME                          READY   STATUS    RESTARTS
    pod/llama-3-7989ffd8d-ck62t   0/1     Pending   0

    NAME              TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)
    service/llama-3   ClusterIP   10.100.23.237   <none>        8000/TCP,8001/TCP,8002/TCP
    ```

4.  Uninstalling the Chart

    Uninstalling a Helm chart is as straightforward as running the command below.
    This is useful when experimenting with various options and configurations.

    ```bash
    helm uninstall <installation_name>
    ```

### How It Works

The Helm chart creates a model-conversion job and multiple Kubernetes deployments to support the distributed model's tensor parallelism needs.
When a distributed model is deployed, a "leader" pod along with a number of "workers" to meet the model's tensor parallelism requirements are
created.
The leader pod then awaits for the conversion job to complete and for all worker pods to be successfully deployed.

The model-conversion job is responsible for downloading the configured model from Hugging Face and converting that model into a TensorRT-LLM
ready set of engine and plan files.
The model-conversion job will place all downloaded and converted files on the provided persistent volume.

> [!Note]
> Model downloads from Hugging Face are reused when possible.
> Converted TRT-LLM models are GPU and tensor-parallelism specific.
> Therefore a converted model will exist for every GPU the model is deployed on to as well as for every configuration of tensor parallelism.

Once these conditions are met, the leader pod creates an [`mpirun`](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html) process which creates a Triton Server process in each pod of the distributed model.

The leader pod's process is responsible for handling inference request and response functionality, as well as inference request tokenization and
result de-tokenization.
Worker pods' processes provide expanded GPU compute and memory capacity.
All of the processes are coordinated by the original `mpirun` process.
Communications between the processes is accelerated by [NVIDIA Collective Communications Library](https://developer.nvidia.com/nccl) (NCCL).
NCCL enables GPU-to-GPU direct communication and avoids the wasteful data copying from GPU-to-CPU-to-GPU that occur otherwise.


### Potential Improvements

#### Autoscaling and Gang Scheduling

This guide does not provide any solution for autoscaling or load balancing Triton deployments because Kubernetes horizontal pod
autoscaling (HPA) is not capable of managing deployments composed of multiple pods.
Additionally, because the solution provided in this tutorial makes use of multiple deployments, any automation has a high risk of concurrent,
partial deployments exhausting available resources preventing any of the deployments from succeeding.

For an example of concurrent, partial deployments preventing each other from successfully deploying, imagine a cluster with 4 nodes, each with 8 GPUs for a total of 32 available GPUs.
Now consider a model which requires 8 GPUs to be deployed and we attempt to deploy 5 copies of it.
When individually deploying the models, each deployment is assigned 8 GPUs until there are zero available GPUs remaining resulting in the model
being successfully deployed 4 times.
At this point, the system understands that there are no more available resources and the 5 model copy fails to deploy.

However, when attempting to deploy all 5 copies of the model simultaneously, it is highly likely that each copy will get at least 1 GPU resource
assigned to it.
This results in their insufficient resources for at least two of the copies; leaving both deployments stuck in a non-functional, partially
deployed state.

One solution to this problem would be to leverage a gang scheduler for Kubernetes.
Gang scheduling would enable the Kubernetes scheduler to only create a pod if its entire cohort of pods can be created.
This provides a solution to the partial deployment of model pods blocking each other from being fully deployed.

> [!Note]
> Read about [gang scheduling on Wikipedia](https://en.wikipedia.org/wiki/Gang_scheduling) for additional information.

The above solutions, however, does not provide any kind of autoscaling solution.
To achieve this, a custom, gang-schedular-aware autoscaler would be required.

#### Network Topology Aware Scheduling

Triton Server w/ TensorRT-LLM leverage a highly-optimized networking stacked known as the
[NVIDIA Collective Communications Library](https://developer.nvidia.com/nccl) (NCCL) to enable tensor parallelization.
NCCL takes advantage of he ability for modern GPUs to leverage
[remote direct memory access](https://en.wikipedia.org/wiki/Remote_direct_memory_access) (RDMA) based network acceleration to optimize operations
between GPUs regardless if they're on the same or nearby machines.
This means that quality of the network between GPUs on separate machines directly affects the performance of distributed models.

Providing a network topology aware scheduler for Kubernetes, could help ensure that the GPUs assigned to the pods of a model deployment are
relatively local to each other.
Ideally, on the same machine or at least the same networking switch to minimize network latency and the impact of bandwidth limitations.


## Developing this Guide

During the development of this guide, I ran into several problems that needed to be solved before we could provide a useful
guide.
This section will outline and describe the issues I ran into and how we resolved them.

> _This document was developed using a Kubernetes cluster provided by Amazon EKS._
> _Clusters provisioned on-premises or provided by other cloud service providers such as Azure AKS or GCloud GKE might require_
> _modifications to this guide._


### Why This Set of Software Components?

The set of software packages described in this document is close the minimum viable set of packages without handcrafting
custom Helm charts and YAML files for every package and dependency.
Is this the only set of packages and components that can be used to make this solution work?
Definitely not, there are several alternatives which could meet our requirements.
This set of packages and components is just the set I happen to choose for this guide.

Below is a high-level description of why each package is listed in this guide.

#### NVIDIA Device Plugin for Kubernetes

Required to enable GPUs to be treated as resources by the Kubernetes scheduler.
Without this component, GPUs would not be assigned to containers correctly.

#### NVIDIA GPU Discovery Service for Kubernetes

Provides automatic labelling of Kubernetes nodes based on the NVIDIA devices and software available on the node.
Without the provided labels, it would not be possible to specify specific GPU SKUs when deploying models because the
Kubernetes scheduler treats all GPUs as identical (referring to them all with the generic resources name `nvidia.com/gpu`).

#### Kubernetes Node Discovery Service

This is a requirement for the [NVIDIA GPU Discovery Service for Kubernetes](#nvidia-gpu-discovery-service-for-kubernetes).

#### NVIDIA DCGM Exporter

Provides hardware monitoring and metrics for NVIDIA GPUs and other devices present in the cluster.
Without the metrics this provides, monitoring GPU utilization, temperature and other metrics would not be possible.

While Triton Server has the capability to collect and serve NVIDIA hardware metrics, relying on Triton Server to provide this
service is non-optimal for several reasons.

Firstly, many processes on the same machine querying the NVIDIA device driver for current state, filtering the results for
only values that pertain to the individual process, and serving them via Triton's open-metrics server is as wasteful as the
the number of Triton Server process beyond the first on the node.

Secondly, due to the need to interface with the kernel-mode driver to retrieve hardware metrics, queries get serialized adding
additional overhead and latency to the system.

Finally, the rate at which metrics are collected from Triton Server is not the same as the rate at which metrics are collected
from the DCGM Exporter.
Separating the metrics collection from Triton Server allows for customized metric collection rates, which enables us to
further minimize the process overhead placed on the node.

##### Why is the DCGM Exporter Values File Custom?

I decided to use a custom values file when installing the DCGM Exporter Helm chart for several reasons.

Firstly, it is my professional opinion that every container in a cluster should specify resource limits and requests.
Not doing so opens the node up to a number of difficult to diagnose failure conditions related to resource exhaustion.
Out of memory errors are the most obvious and easiest to root cause.
Additionally, difficult to reproduce, transient timeout and timing errors caused CPU over-subscription can easily happen when
any container is unconstrained and quickly waste an entire engineering team's time as they attempt to triage, debug, and
resolve them.

Secondly, the DCGM Exporter process itself spams error logs when it cannot find NVIDIA devices in the system.
This is primarily because the service was originally created for non-Kubernetes environments.
Therefore I wanted to restrict which node the exporter would get deployed to.
Fortunately, the DCGM Helm chart makes this easy by support node selector options.

Thirdly, because nodes with NVIDIA GPUs have been tainted with the `nvidia.com/gpu=present:NoSchedule` that prevents any
pod which does not explicitly tolerate the taint from be assigned to the node, I need to add the tolerations to the DCGM
Exporter pod.

Finally, the default Helm chart for DCGM Exporter is missing the required `--kubernetes=true` option being passed in via
command line options when the process is started.
Without this option, DCGM Exporter does not correctly associate hardware metrics with the pods actually using it, and
there would be mechanism for understand how each pod uses the GPU resources assigned to it.


### Why Use the Triton CLI and Not Other Tools Provided by NVIDIA?

I chose to use the new [Triton CLI](https://github.com/triton-inference-server/triton_cli) tool to optimize models for
TensorRT-LLM instead of other available tools for a couple of reasons.

Firstly, using the Triton CLI simplifies the conversion and optimization of models into a single command.

Secondly, relying on the Triton CLI simplifies the creation of the container because all requirements were met with a single
`pip install` command.

#### Why Use a Custom Branch of Triton CLI Instead of an Official Release?

I decided to use a custom [branch of Triton CLI](https://github.com/triton-inference-server/triton_cli/tree/jwyman/aslb-mn)
because there are features this guide needed that were not present in any of the official releases available.
The branch is not a Merge Request because the method used to add the needed features does not aligned with changes the
maintainers have planned.
Once we can achieve alignment, this guide will be updated to use an official release.


### Why Does the Chart Run a Python Script Instead of Triton Server Directly?

There are two reasons:

1.  In order to retrieve a model from Hugging Face, convert and optimize it for TensorRT-LLM, and cache it on the host, I
    decided that [pod initialization container](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) was the
    most straightforward solution.

    In order to make the best use of the initialization container I chose to use a custom [server.py](./containers/server.py)
    script that made of the new [Triton CLI](https://github.com/triton-inference-server/triton_cli) tool.

2.  Multi-GPU deployments require a rather specialized command line to run, and generating it using Helm chart scripting was
    not something I wanted to deal with.
    Leveraging the custom Python script was the logical, and easiest, solution.

#### Why is the Python Written Like That?

Because I'm not a Python developer, but I am learning!
My background is in C/C++ with plenty of experience with shell scripting languages.


### Why Use a Custom Triton Image?

I decided to use a custom image for a few reasons.

1.  Given the answer above and the use of Triton CLI and a custom Python script, the initialization container needed both
    components pre-installed in it to avoid unnecessary use of ephemeral storage.

    > [!Warning]
    > Use of ephemeral storage can lead to pod eviction, and therefore should be avoided whenever possible.

2.  Since the Triton + TRT-LLM image is already incredibly large, I wanted to avoid consuming additional host storage space
    with yet another container image.

    Additionally, the experience of a pod appearing to be stuck in the `Pending` state while it download a container prior to
    the initialization container is easier to understand compared to a short `Pending` state before the initialization
    container, followed by a much longer `Pending` state before the Triton Server can start.

3.  I wanted a custom, constant environment variable set for `ENGINE_DEST_PATH` that could be used by both the initialization
    and Triton Server containers.

---

Software versions featured in this document:

* Triton Inference Server v2.45.0 (24.04-trtllm-python-py3)
* TensorRT-LLM v0.9.0
* Triton CLI v0.0.7
* NVIDIA Device Plugin for Kubernetes v0.15.0
* NVIDIA GPU Discovery Service for Kubernetes v0.8.2
* NVIDIA DCGM Exporter v3.3.5
* Kubernetes Node Discovery Service v0.15.4
* Prometheus Stack for Kubernetes v58.7.2
* Prometheus Adapter for Kubernetes v4.10.0

---

Author: J Wyman, System Software Architect, AI &amp; Distributed Systems

Copyright &copy; 2024, NVIDIA CORPORATION. All rights reserved.
