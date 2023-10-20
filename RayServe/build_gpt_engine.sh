#!/bin/bash

cd /workspace/.tensorrt_llm-build/tensorrtllm_backend/tensorrt_llm/examples/gpt/
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
python3 build.py --model_dir=./c-model/gpt2/1-gpu/ \
                 --world_size=1 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --parallel_build \
                 --output_dir=engines/fp16/1-gpu

# cp -rf /workspace/.tensorrt_llm-build/tensorrtllm_backend/all_models/inflight_batcher_llm/* /workspace/models/

cp -rf /workspace/.tensorrt_llm-build/tensorrtllm_backend/tensorrt_llm/examples/gpt/engines/fp16/1-gpu/* /workspace/models/tensorrt_llm/1

chmod -R a+wr /workspace/models/


