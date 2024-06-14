# Triton Tutorials

For users experiencing the "Tensor in" & "Tensor out" approach to Deep Learning Inference, getting started with Triton can lead to many questions. The goal of this repository is to familiarize users with Triton's features and provide guides and examples to ease migration. For a feature by feature explanation, refer to the [Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

#### Getting Started Checklist
| [Overview Video](https://www.youtube.com/watch?v=NQDtfSi5QF4) | [Conceptual Guide: Deploying Models](Conceptual_Guide/Part_1-model_deployment/README.md) |
| ------------ | --------------- |

## Quick Deploy

The focus of these examples is to demonstrate deployment for models trained with various frameworks. These are quick demonstrations made with an understanding that the user is somewhat familiar with Triton.

### Deploy a ...
| [PyTorch Model](./Quick_Deploy/PyTorch/README.md) | [TensorFlow Model](./Quick_Deploy/TensorFlow/README.md) | [ONNX Model](./Quick_Deploy/ONNX/README.md) | [TensorRT Accelerated Model](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton) | [vLLM Model](./Quick_Deploy/vLLM/README.md) | [OpenVINO Model](./Quick_Deploy/OpenVINO/README.md)
| --------------- | ------------ | --------------- | --------------- | --------------- | --------------- |

## LLM Tutorials
The table below contains some popular models that are supported in our tutorials
| Example Models   | Tutorial Link |
| :-------------: | :------------------------------: |
| [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) |[TensorRT-LLM Tutorial](Popular_Models_Guide/Llama2/trtllm_guide.md) |
| [Persimmon-8B](https://www.adept.ai/blog/persimmon-8b) | [HuggingFace Transformers Tutorial](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/HuggingFaceTransformers)  |
[Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) |[HuggingFace Transformers Tutorial](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/HuggingFaceTransformers)   |

**Note:**
This is not an exhausitive list of what Triton supports, just what is included in the tutorials.

## What does this repository contain?
This repository contains the following resources:
* [Conceptual Guide](./Conceptual_Guide/): This guide focuses on building a conceptual understanding of the general challenges faced whilst building inference infrastructure and how to best tackle these challenges with Triton Inference Server.
* [Quick Deploy](#quick-deploy): These are a set of guides about deploying a model from your preferred framework to the Triton Inference Server. These guides assume a basic understanding of the Triton Inference Server. It is recommended to review the getting started material for a complete understanding.
* [HuggingFace Guide](./HuggingFace/): The focus of this guide is to walk the user through different methods in which a HuggingFace model can be deployed using the Triton Inference Server.
* [Feature Guides](./Feature_Guide/): This folder is meant to house Triton's feature-specific examples.
* [Migration Guide](./Migration_Guide/migration_guide.md): Migrating from an existing solution to Triton Inference Server? Get an understanding of the general architecture that might best fit your use case.

## Navigating Triton Inference Server Resources

The Triton Inference Server GitHub organization contains multiple repositories housing different features of the Triton Inference Server. The following is not a complete description of all the repositories, but just a simple guide to build intuitive understanding.

* [Server](https://github.com/triton-inference-server/server) is the main Triton Inference Server Repository.
* [Client](https://github.com/triton-inference-server/client) contains the libraries and examples needed to create Triton Clients
* [Backend](https://github.com/triton-inference-server/backend) contains the core scripts and utilities to build a new Triton Backend. Any repository containing the word "backend" is either a framework backend or an example for how to create a backend.
* Tools like [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) and [Model Navigator](https://github.com/triton-inference-server/model_navigator) provide the tooling to either measure performance, or to simplify model acceleration.

## Adding Requests

Open an issue and specify details for adding a request for an example. Want to make a contribution? Open a pull request and tag an Admin.
