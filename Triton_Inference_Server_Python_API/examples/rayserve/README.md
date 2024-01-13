# Triton Inference Server Ray Serve Deployment

Using the Triton Inference Server In-Process Python API you can easily
integrate your triton server based models into any Python framework
including FastAPI and Ray Serve.

This directory contains a simple Triton Inference Server based Ray Serve
deployment.

## Run Container
```bash
./run.sh --framework hf_diffusers
```

## Run Deployment
```bash
python3 examples/rayserve/tritonserver_deployment.py
```
