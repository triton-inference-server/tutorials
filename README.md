# Triton Tutorials

For users experiencing the "Tensor in" & "Tensor out" approach to Deep Learning Inference, getting started with Triton can lead to many questions. The goal of this repository is to familiarize users with Triton's features and provide guides and examples to ease migration.

| [Conceptual Guide](Conceptual_Guide/README.md) | [Quick Overview](https://www.youtube.com/watch?v=NQDtfSi5QF4) | [Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) |
| ------------ | --------------- | --------------- |

## Quick Deploy

The focus of these examples is to demonstrate deployment for models trained with various frameworks. These are quick demonstrations made with an understanding that the user is somewhat familiar with Triton. **It is highly recommended to review [Part 1 of the Conceptual Guide](Conceptual_Guide/Part_1-model_deployment/README.md) for a complete understanding**.

| Deploy a | [PyTorch Model](./Quick_Deploy/PyTorch/README.md) | [TensorFlow Model](./Quick_Deploy/TensorFlow/README.md) | [ONNX Model](./Quick_Deploy/ONNX/README.md) | [TensorRT Accelerated Model](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton) |
| ------------ | --------------- | ------------ | --------------- | --------------- |

## Navigating Triton Inference Server Resources

The Triton Inference Server GitHub organization contains multiple repositories housing different features of the Triton Inference Server. The following is not a complete description of all the repositories, but just a simple guide to build intuitive understanding.

* [Server](https://github.com/triton-inference-server/server) is the main Triton Inference Server Repository.
* [Client](https://github.com/triton-inference-server/client) contains the libraries and examples needed to create Triton Clients
* [Backend](https://github.com/triton-inference-server/backend) contains the core scripts and utilities to build a new Triton Backend. Any repository containing the word "backend" is either a framework backend or an example for how to create a backend.
* Tools like [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) and [Model Navigator](https://github.com/triton-inference-server/model_navigator) provide the tooling to either measure performance, or to simplify model acceleration.

## Adding Requests

Open an issue and specify details for adding a request for an example. Want to make a contribution? Open a pull request and tag an Admin.
