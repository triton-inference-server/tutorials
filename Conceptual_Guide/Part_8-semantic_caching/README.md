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

# Semantic Caching

When deploying large language models (LLMs) or LLM-based workflows
there are two key factors to consider: the performance and cost-efficiency
of your application. Generating language model outputs requires significant
computational resources, for example GPU time, memory usage, and other
infrastructure costs. These resource-intensive requirements create a
pressing need for optimization strategies that can maintain
high-quality outputs while minimizing operational expenses.

Semantic caching emerges as a powerful solution to reduce computational costs
for LLM-based applications.

## Definition and Benefits

**_Semantic caching_** is a caching mechanism that takes into account
the semantics of the incoming request, rather than just the raw data itself.
It goes beyond simple key-value pairs and considers the content or
context of the data.

This approach offers several benefits including, but not limited to:

+ **Cost Optimization**

    - Semantic caching can substantially reduce operational expenses associated
    with LLM deployments. By storing and reusing responses for semantically
    similar queries, it minimizes the number of actual LLM calls required.

+ **Reduced Latency**

    - One of the primary benefits of semantic caching is its ability to
    significantly improve response times. By retrieving cached responses for
    similar queries, the system can bypass the need for full model inference,
    resulting in reduced latency.

+ **Increased Throughput**

    - Semantic caching allows for more efficient utilization of computational
    resources. By serving cached responses for similar queries, it reduces the
    load on infrastructure components. This efficiency enables the system
    to handle a higher volume of requests with the same hardware, effectively
    increasing throughput.

+ **Scalability**

    - As the user base and the volume of queries grow, the probability of cache
    hits increases, provided that there is adequate storage and resources
    available to support this scaling. The improved resource efficiency and
    reduced computational demands allows applications to serve more users
    without a proportional increase in infrastructure costs.

+ **Consistency in Responses**

    - For certain applications, maintaining consistency in responses to
    similar queries can be beneficial. Semantic caching ensures that analogous
    questions receive uniform answers, which can be particularly useful
    in scenarios like customer service or educational applications.

## Sample Reference Implementation

In this tutorial we provide a reference implementation for a Semantic Cache in
[semantic_caching.py](./artifacts/semantic_caching.py). There are 3 key
dependencies:
* [SentenceTransformer](https://sbert.net/): a Python framework for computing
dense vector representations (embeddings) of sentences, paragraphs, and images.
    - We use this library and `all-MiniLM-L6-v2` in particular to convert
    incoming prompt into an embedding, enabling semantic comparison.
    - Alternatives include [semantic search models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models),
    OpenAI Embeddings, etc.
* [Faiss](https://github.com/facebookresearch/faiss/wiki): an open-source library
developed by Facebook AI Research for efficient similarity search and
clustering of dense vectors.
    - This library is used for the embedding store and extracting the most
    similar embedded prompt from the cached requests (or from the index store).
    - This is a mighty library with a great variety of CPU and GPU accelerated
    algorithms.
    - Alternatives include [annoy](https://github.com/spotify/annoy), or
    [cuVS](https://github.com/rapidsai/cuvs). However, note that cuVS already
    has an integration in Faiss, more on this can be found [here](https://docs.rapids.ai/api/cuvs/nightly/integrations/faiss/).
* [Theine](https://github.com/Yiling-J/theine): High performance in-memory
cache.
    - We will use it as our exact match cache backend. After the most similar
    prompt is identified, the corresponding cached response is extracted from
    the cache. This library supports multiple eviction policies, in this
    tutorial we use "LRU".
    - One may also look into [MemCached](https://memcached.org/about) as a
    potential alternative.

Provided [script](./artifacts/semantic_caching.py) is heavily annotated and we
encourage users to look through the code to gain better clarity in all
the necessary stages.

## Incorporating Semantic Cache into your workflow

For this tutorial, we'll use the [vllm backend](https://github.com/triton-inference-server/vllm_backend)
as our example, focusing on demonstrating how to cache responses for the
non-streaming case. The principles covered here can be extended to handle
streaming scenarios as well.

### Customising vLLM Backend

First, let's start by cloning Triton's vllm backend repository. This will
provide the necessary codebase to implement our semantic caching example.

```bash
git clone https://github.com/triton-inference-server/vllm_backend.git
cd vllm_backend
```

With the repository successfully cloned, the next step is to apply all
necessary modifications. To simplify this process, we've prepared a
[semantic_cache.patch](tutorials/Conceptual_Guide/Part_8-semantic_caching/artifacts/semantic_cache.patch)
that consolidates all changes into a single step:

```bash
curl https://raw.githubusercontent.com/triton-inference-server/tutorials/refs/heads/main/Conceptual_Guide/Part_8-semantic_caching/artifacts/semantic_cache.patch | git apply -v
```

If you're eager to start using Triton with the optimized vLLM backend,
you can skip ahead to the
[Launching Triton with Optimized vLLM Backend](#launching-triton-with-optimized-vllm-backend)
section. However, for those interested in understanding the specifics,
let's explore what this patch includes.

The patch introduces a new script,
[semantic_caching.py](./artifacts/semantic_caching.py), which is added to the
appropriate directory. This script implements the core logic for our
semantic caching functionality.

Next, the patch integrates semantic caching into the model. Let's walk through
these changes step-by-step.

Firstly, it imports the necessary classes from
[semantic_caching.py](./artifacts/semantic_caching.py) into the codebase:

```diff
...

from utils.metrics import VllmStatLogger
+from utils.semantic_caching import SemanticCPUCacheConfig, SemanticCPUCache
```

Next, it sets up the semantic cache during the initialization step.
This setup will prepare your model to utilize semantic caching during
its operations.

```diff
    def initialize(self, args):
        self.args = args
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        ...

        # Starting asyncio event loop to process the received requests asynchronously.
        self._loop = asyncio.get_event_loop()
        self._event_thread = threading.Thread(
            target=self.engine_loop, args=(self._loop,)
        )
        self._shutdown_event = asyncio.Event()
        self._event_thread.start()
+       config = SemanticCPUCacheConfig()
+       self.semantic_cache = SemanticCPUCache(config=config)

```

Finally, the patch incorporates logic to query and update the semantic cache
during request processing. This ensures that cached responses are efficiently
utilized whenever possible.

```diff
    async def generate(self, request):
        ...
        try:
            request_id = random_uuid()
            prompt = pb_utils.get_input_tensor_by_name(
                request, "text_input"
            ).as_numpy()[0]
            ...

            if prepend_input and stream:
                raise ValueError(
                    "When streaming, `exclude_input_in_output` = False is not allowed."
                )
+           cache_hit = self.semantic_cache.get(prompt)
+           if cache_hit:
+               try:
+                   response_sender.send(
+                   self.create_response(cache_hit, prepend_input),
+                   flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
+                   )
+                   if decrement_ongoing_request_count:
+                       self.ongoing_request_count -= 1
+               except Exception as err:
+                   print(f"Unexpected {err=} for prompt {prompt}")
+               return None
            ...

            async for output in response_iterator:
                ...

            last_output = output

            if not stream:
                response_sender.send(
                    self.create_response(last_output, prepend_input),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
+               self.semantic_cache.set(prompt, last_output)

```

### Launching Triton with Optimized vLLM Backend

To evaluate or optimized vllm backend, let's start vllm docker container and
mount our implementation to `/opt/tritonserver/backends/vllm`. We'll
also mount sample model repository, provided in
`vllm_backend/samples/model_repository`. Feel free to set up your own.
Use the following docker command to start Triton's vllm docker container,
but make sure to specify proper paths to the cloned `vllm_backend`
repository and replace `<xx.yy>` with the latest release of Triton.

```bash
docker run --gpus all -it --net=host --rm \
    --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/vllm_backend/src/:/opt/tritonserver/backends/vllm \
    -v /path/to/vllm_backend/samples/model_repository:/workspace/model_repository \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3
```

When inside the container, make sure to install required dependencies:
```bash
pip install sentence_transformers faiss_gpu theine
```

Finally, let's launch Triton
```bash
tritonserver --model-repository=model_repository/
```

After you start Triton you will see output on the console showing
the server starting up and loading the model. When you see output
like the following, Triton is ready to accept inference requests.

```
I1030 22:33:28.291908 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1030 22:33:28.292879 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1030 22:33:28.335154 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

### Evaluation

After you [start Triton](#launching-triton-with-optimized-vllm-backend)
with the sample model_repository, you can quickly run your first inference
request with the
[generate endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md).

We'll also time this query:

```bash
time curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "Tell me, how do I create model repository for Triton Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens":100}, "exclude_input_in_output":true}'
```

Upon success, you should see a response from the server like this one:
```
{"model_name":"vllm_model","model_version":"1","text_output": <MODEL'S RESPONSE>}
real	0m1.128s
user	0m0.000s
sys	0m0.015s
```

Now, let's try a different response, but keep the semantics:

```bash
time curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "How do I set up model repository for Triton Inference Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens":100}, "exclude_input_in_output":true}'
```

Upon success, you should see a response from the server like this one:
```
{"model_name":"vllm_model","model_version":"1","text_output": <SAME MODEL'S RESPONSE>}
real	0m0.038s
user	0m0.000s
sys	0m0.017s
```

Let's try one more:

```bash
time curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "How model repository should be set up for Triton Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens":100}, "exclude_input_in_output":true}'
```

Upon success, you should see a response from the server like this one:
```
{"model_name":"vllm_model","model_version":"1","text_output": <SAME MODEL'S RESPONSE>}
real	0m0.059s
user	0m0.016s
sys	0m0.000s
```

Clearly, the latter 2 requests are semantically similar to the first one, which
resulted in a cache hit scenario, which reduced the latency of our model from
approx 1.1s to the average of 0.048s per request.

## Current Limitations

* The current implementation of the Semantic Cache only considers the prompt
itself for cache hits, without accounting for additional request parameters
such as `max_tokens` and `temperature`. As a result, these parameters are not
included in the cache hit evaluation, which may affect the accuracy of cached
responses when different configurations are used.

* Semantic Cache effectiveness is heavily reliant on the choice of embedding
model and application context. For instance, queries like "How to set up model
repository for Triton Inference Server?" and "How not to set up model
repository for Triton Inference Server?" may have high cosine similarity
despite differing semantically. This makes it challenging to set an optimal
threshold for cache hits, as a narrow similarity range might exclude useful
cache entries.

## Interested in This Feature?

While this reference implementation provides a glimpse into the potential
of semantic caching, it's important to note that it's not an officially
supported feature in Triton Inference Server.

We value your input! If you're interested in seeing semantic caching as a
supported feature in future releases, we invite you to join the ongoing
[discussion](https://github.com/triton-inference-server/server/discussions/7742).
Provide details about why you think semantic caching would
be valuable for your use case. Your feedback helps shape our product roadmap,
and we appreciate your contributions to making our software better for everyone.