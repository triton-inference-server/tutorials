# RayServe + Triton Inference Server Prototype

The goal of this project is to demonstrate integration with RayServe
based on Triton's low level python bindings.

A partial "pythonic wrapper" is provided as an early preview of the
type of high level api that will be provided in a coming Triton
release.

>**Note** This code is not meant for production and is subject to
>change.

## Included Files

|File|Description|
|----|-----------|
|build.sh| builds a docker image based on Triton 23.09 + RayServe |
|build_tensorrt_llm.sh| builds a docker image based on Triton 23.09 + TensorRT-LLM + Ray Server |
|Dockerfile | Dockerfile for Triton 23.09 + RayServe |
|Dockerfile.trt_llm| Dockerfile for Triton 23.09 + TensorRT-LLM + RayServe|
|models_test | Simple model repository with simple custom python model |
|models_trt_llm | config files for TensorRT-LLM gpt2 model |
|build_gpt_engine.sh | builds the engine file for gpt 2 |
|run.sh | runs basic image |
|run_trt_llm.sh| runs TensorRT-LLM enabled image |
| triton_deployment.py | RayServe deployment including embedded Triton Server |
| tritonserver_api.py | "Pythonic" wrapper on top of low level bindings

## Python Binding References

| File | Description |
| ---- | ----------- |
| [tritonserver_pybind.cc](https://github.com/triton-inference-server/core/blob/main/python/tritonserver/_c/tritonserver_pybind.cc) | Low level bindings |
| [test_binding.py](https://github.com/triton-inference-server/core/blob/main/python/test/test_binding.py) | Unit tests and example usage |

## Building and Running within Docker

The prototype is designed to be run within a docker container using
volume mounting for interactive local development.

### Triton 23.09 + RayServe

#### Build Image
```bash
   ./build.sh
```

#### Supported Backends
```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt
```

#### Run
```bash
  ./run.sh
```

#### Within Docker Container

```bash
python3 triton_deployment.py
```

##### Expected Result
```
(ServeReplica:default:TritonDeployment pid=2736) {'name': 'test', 'version': 1, 'state': 'READY'}
<SNIP>
Hello Theodore!
<SNIP>
{'text_output': 'Theodore', 'fp16_output': [0.5]}
<SNIP>
```

#### Within Docker Container
```bash
serve run triton_deployment:triton_app
```

##### Expected Result
```bash
<SNIP>
ServeReplica:default:TritonDeployment pid=10347) {'name': 'test', 'version': 1, 'state': 'READY'}
<SNIP>
```

#### Interact
```bash
curl "localhost:8000/test?text_input="who%20is%20Triton%20Inference%20Server?"&fp16_input=0.5"
```

##### Expected Result
```bash
<SNIP>
{"text_output":"who is Triton Inference Server?","fp16_output":[0.5]}
```

### Triton 23.09 + TensorRT-LLM + RayServe

#### Build Image
>**Note:** First time image build will take several minutes

```bash
   ./build_tensorrt_llm.sh
```

#### Supported Backends
```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt  tensorrtllm
```

#### Run
```bash
  ./run_trt_llm.sh
```

#### Build TensorRT-LLM Engine

>**Note:** First time engine build will a few minutes

```bash
./build_gpt_engine.sh
```

#### Within Docker Container

```bash
python3 triton_deployment.py
```

##### Expected Result
```
(ServeReplica:default:TritonDeployment pid=3556) [TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 957, GPU 3183 (MiB)
(ServeReplica:default:TritonDeployment pid=3556) [TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 774 (MiB)
(ServeReplica:default:TritonDeployment pid=3556) [TensorRT-LLM][INFO] Using 410880 tokens in paged KV cache.
(ServeReplica:default:TritonDeployment pid=3556) {'name': 'ensemble', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=3556) {'name': 'postprocessing', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=3556) {'name': 'preprocessing', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=3556) {'name': 'tensorrt_llm', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=3556) {'name': 'test', 'version': 1, 'state': 'READY'}
Hello Theodore!
(ServeReplica:default:TritonDeployment pid=3556) INFO 2023-10-20 06:56:12,295 TritonDeployment default#TritonDeployment#jgXhTy 54e56361-b027-4dbd-b211-589d6b270589 /hello default replica.py:749 - __CALL__ OK 5.5ms
Theodore Roosevelt, who was president from 1933 to 1945, was a staunch advocate of the use of nuclear weapons. He was also a staunch opponent of the use of atomic weapons.

"I am not a pacifist," he said in a speech in 1945. "I am a pacifist because I believe that the use of nuclear weapons is a terrible thing. I believe that the use of nuclear weapons is a terrible thing. I believe that the use of nuclear weapons is a terrible thing. I believe
{'text_output': 'Theodore', 'fp16_output': [0.5]}
```

#### Within Docker Container
```bash
serve run triton_deployment:triton_app
```

##### Expected Result
```bash
(ServeReplica:default:TritonDeployment pid=7540) [TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 774 (MiB)
(ServeReplica:default:TritonDeployment pid=7540) [TensorRT-LLM][INFO] Using 410880 tokens in paged KV cache.
(ServeReplica:default:TritonDeployment pid=7540) {'name': 'ensemble', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=7540) {'name': 'postprocessing', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=7540) {'name': 'preprocessing', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=7540) {'name': 'tensorrt_llm', 'version': 1, 'state': 'READY'}
(ServeReplica:default:TritonDeployment pid=7540) {'name': 'test', 'version': 1, 'state': 'READY'}
2023-10-20 06:57:30,122	SUCC scripts.py:519 -- Deployed Serve app successfully.
```

#### Interact
```bash
curl "localhost:8000/generate?text_input='who%20is%20groot?'"
```

##### Expected Result
```bash
<SNIP>
"'who is groot?'\n\n'I am groot,' said the old man, 'and I am groot.'\n\n'And who is groot?'\n\n'I am groot,' said the old man, 'and I am groot.'\n\n'And who is groot?'\n\n'I am groot,' said the old man, 'and I am groot.'\n\n'And who is groot?'\n\n'I am groot,' said the old man,"
```

## Limitations

* Currently, while models are loaded and executed on the GPU - input tensors are passed via CPU memory

* No asyncio version of response iterator

* Incomplete wrapper of APIs

* Robust error handling

* Performance not tested
