<!--
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Stable Diffusion Model Configuration Options

The example python based backend
[`/backend/diffusion/model.py`](../backend/diffusion/model.py) supports
the following configuration parameters to customize the model being served.

## Full Configuration Examples

   * [Stable Diffusion v1.5](../diffusion-models/stable_diffusion_1_5/config.pbtxt)
   * [Stable Diffusion XL](../diffusion-models/stable_diffusion_xl/config.pbtxt)

## Batch Size and Dynamic Batching

You can select the batch size and dynamic batching queue delay. With
batch size 1 dynamic batching is disabled.

> [!Note]
> Changing the batch size requires rebuilding the TensorRT Engines


```bash
max_batch_size: 1

dynamic_batching {
 max_queue_delay_microseconds: 100000
}

```

## Engine Building Parameters

The following configuration parameters affect the engine build.

Please see the [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/9.2/demo/Diffusion)
for more information.

```
{
  key: "onnx_opset"
  value: {
    string_value: "18"
  }
},
{
  key: "image_height"
  value: {
    string_value: "512"
  }
},
{
  key: "image_width"
  value: {
    string_value: "512"
  }
},
{
  key: "version"
  value: {
    string_value: "1.5"
  }
}
```

## Forcing Engine Build

Setting the following parameter to a non empty value will force an
engine rebuild.

```
{
  key: "force_engine_build"
  value: {
    string_value: ""
  }
}
```

## Runtime Settings

The following configuration parameters affect the runtime behavior of the model.
Please see the [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/9.2/demo/Diffusion)
for more information.

Setting a non null integer value for `seed` will result in
deterministic results.

```
{
  key: "steps"
  value: {
    string_value: "50"
  }
},
{
  key: "scheduler"
  value: {
    string_value: ""
  }
},
{
  key: "guidance_scale"
  value: {
    string_value: "7.5"
  }
},
{
  key: "seed"
  value: {
    string_value: ""
  }
}
```
