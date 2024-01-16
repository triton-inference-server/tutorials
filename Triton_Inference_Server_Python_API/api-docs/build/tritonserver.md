<a id="tritonserver-package"></a>

# tritonserver package

<a id="module-tritonserver"></a>

<a id="module-contents"></a>

## Module contents

Triton Inference Server In-Process Python API

The Triton Inference Server In-Process Python API enables developers to easily
embed a Triton inference server within their python
application. Developers can load and interact with models, send
inference requests, query metrics, etc. Everything available through
the C API for emebedding a Triton Inference Server in C / C++
applications has been provided within a Python API.

Note: The Python API is currently in BETA and interfaces and
capabilities are subject to change. Any feedback is welcome.

Note: Any objects not explicitly exported here are considered private.
Note: Any methods, properties, or arguments prefixed with \_ are
considered private.

<a id="tritonserver.Server"></a>

### *class* tritonserver.Server(options: Optional[[Options](#tritonserver.Options)] = None, \*\*kwargs: Unpack[[Options](#tritonserver.Options)])

Bases: `object`

Triton Inference Server

Server objects allow users to instantiate and interact with an
in-process Triton Inference Server. Server objects provide methods
to access models, server metadata, and metrics.

<a id="tritonserver.Server.live"></a>

#### live()

Returns true if the server is live.

See :c:func\`TRITONSERVER_ServerIsLive()\`

* **Returns:**
  True if server is live. False
  otherwise.
* **Return type:**
  bool

### Examples

```pycon
>>> server.live()
server.live()
True
```

<a id="tritonserver.Server.load"></a>

#### load(model_name: str, parameters: Optional[dict[str, str | int | bool | bytes]] = None)

Load a model

Load a model from the repository and wait for it to be
ready. Only available if ModelControlMode is EXPLICIT.

See c:func:TRITONSERVER_ServerLoadModel

* **Parameters:**
  * **model_name** (*str*) – model name
  * **parameters** (*Optional**[**dict**[**str**,* *str* *|* *int* *|* *bool* *|* *bytes**]**]*) – parameters to override settings and upload model artifacts
* **Returns:**
  Model object
* **Return type:**
  [Model](#tritonserver.Model)

### Examples

```pycon
>>> server.load("new_name")
server.load("new_name")
{'name': 'new_name', 'version': -1, 'state': None}
>>> server.models()
server.models()
{('new_name', 1): {'name': 'new_name', 'version': 1, 'state':
'READY'}, ('resnet50_libtorch', -1): {'name':
'resnet50_libtorch', 'version': -1, 'state': None}, ('test_2',
-1): {'name': 'test_2', 'version': -1, 'state': None}}
```

<a id="tritonserver.Server.metadata"></a>

#### metadata()

Returns metadata for server

Returns metadata for server including name, version and
enabled extensions.

See :c:func\`TRITONSERVER_ServerMetadata\`

* **Returns:**
  Dictionary of key value pairs of metadata information
* **Return type:**
  dict[str, Any]

### Examples

```pycon
>>> server.metadata()
server.metadata()
{'name': 'triton', 'version': '2.41.0', 'extensions':
['classification', 'sequence', 'model_repository',
'model_repository(unload_dependents)', 'schedule_policy',
'model_configuration', 'system_shared_memory',
'cuda_shared_memory', 'binary_tensor_data', 'parameters',
'statistics', 'trace', 'logging']}
```

<a id="tritonserver.Server.metrics"></a>

#### metrics(metric_format: ~tritonserver._c.triton_bindings.TRITONSERVER_MetricFormat = <TRITONSERVER_MetricFormat.PROMETHEUS: 0>)

Return server and custom metrics

See c:func:TRITONSERVER_ServerMetrics()

* **Parameters:**
  **metric_format** (*MetricFormat*) – format for metrics
* **Returns:**
  string containing metrics in specified format
* **Return type:**
  str

<a id="tritonserver.Server.model"></a>

#### model(model_name: str, model_version: int = -1)

Factory method for creating Model objects

Creates and returns a Model object that can be used to
interact with a model. See Model documentation for more
details.

Note: Model is not validated until it is used.

* **Parameters:**
  * **model_name** (*str*) – name of model
  * **model_version** (*int*) – model version, default -1
* **Returns:**
  Model object
* **Return type:**
  [Model](#tritonserver.Model)
* **Raises:**
  **InvalidArgumentError** – If server isn’t started.

### Examples

```pycon
>>> server.model("test")
server.model("test")
{'name': 'test', 'version': -1, 'state': None}
>>> server.model("test").metadata()
server.model("test").metadata()
{'name': 'test', 'versions': ['1'], 'platform': 'python',
'inputs': [{'name': 'text_input', 'datatype': 'BYTES',
'shape': [-1]}, {'name': 'fp16_input', 'datatype': 'FP16',
'shape': [-1, 1]}], 'outputs': [{'name': 'text_output',
'datatype': 'BYTES', 'shape': [-1]}, {'name': 'fp16_output',
'datatype': 'FP16', 'shape': [-1, 1]}]}
```

<a id="tritonserver.Server.models"></a>

#### models(exclude_not_ready: bool = False)

Returns a dictionary of known models in the model repository

See c:func:TRTIONSERVER_ServerModelIndex()

* **Parameters:**
  **exclude_not_ready** (*bool*) – exclude any models which are not in a ready state
* **Returns:**
  Dictionary mapping model name, version to Model objects
* **Return type:**
  ModelDictionary
* **Raises:**
  **InvalidArgumentError** – If server is not started

### Examples

```pycon
>>> server.models()
server.models()
{('new_name', -1): {'name': 'new_name', 'version': -1, 'state': None},
('resnet50_libtorch', -1): {'name': 'resnet50_libtorch', 'version':
-1, 'state': None}, ('test_2', -1): {'name': 'test_2', 'version': -1,
'state': None}}
>>> server.models(exclude_not_ready=True)
server.models(exclude_not_ready=True)
{}
```

<a id="tritonserver.Server.poll_model_repository"></a>

#### poll_model_repository()

Poll model repository for changes

Only available if ModelControlMode.POLL is enabled.

See c:func:TRITONSERVER_ServerPollModelRepository

* **Return type:**
  [Server](#tritonserver.Server)

<a id="tritonserver.Server.ready"></a>

#### ready()

Returns True if the server is ready

See c:func:TRITONSERVER_ServerIsReady()

* **Returns:**
  True if server is ready. False otherwise.
* **Return type:**
  bool

### Examples

```pycon
>>> server.ready()
server.ready()
True
```

<a id="tritonserver.Server.register_model_repository"></a>

#### register_model_repository(repository_path: str, name_mapping: Optional[dict[str, str]] = None)

Add a new model repository.

Adds a new model repository.

Only available when ModelControlMode is set to explicit

See `TRITONSERVER_ServerRegisterModelRepository()`

* **Parameters:**
  * **repository_path** (*str*) – repository path
  * **name_mapping** (*Optional**[**dict**[**str**,* *str**]**]*) – override model names

### Examples

```pycon
>>> options = tritonserver.Options()
>>> options.model_control_mode=tritonserver.ModelControlMode.EXPLICIT
>>> options.model_repository="/workspace/models"
>>> options.startup_models=["test"]
>>> server = tritonserver.Server(options)
>>> server.start()
>>> server.models()
{('resnet50_libtorch', -1): {'name': 'resnet50_libtorch',
'version': -1, 'state': None}, ('test', 1): {'name': 'test',
'version': 1, 'state': 'READY'}, ('test_2', -1): {'name':
'test_2', 'version': -1, 'state': None}}
>>> server.unregister_model_repository("/workspace/models")
>>> server.models()
{}
```

```pycon
>>> server.register_model_repository("/workspace/models",{"test":"new_model"})
>>> server.models()
{('new_name', -1): {'name': 'new_name', 'version': -1,
'state': None}, ('resnet50_libtorch', -1): {'name':
'resnet50_libtorch', 'version': -1, 'state': None}, ('test_2',
-1): {'name': 'test_2', 'version': -1, 'state': None}}
```

<a id="tritonserver.Server.start"></a>

#### start(wait_until_ready: bool = False, polling_interval: float = 0.1, timeout: Optional[float] = None)

Start the in-process server

Starts the in-process server and loads models (depending on
the ModelControlMode setting). Configuration options are
validated and any errors raised as exceptions.

* **Parameters:**
  * **wait_until_ready** (*bool**,* *default False*) – Wait for the server to reach a ready state before
    returning.
  * **polling_interval** (*float**,* *default 0.1*) – Time to sleep between polling for server ready. Only
    applicable if wait_until_ready is set to True.
  * **timeout** (*Optional**[**float**]*) – Timeout when waiting for server to be ready. Only
    applicable if wait_until_ready is set to True.
* **Return type:**
  [Server](#tritonserver.Server)
* **Raises:**
  * **UnavailableError** – If timeout reached before server ready.
  * **InvalidArgumentError** – Raised on invalid configuration or if server already
        started.

### Examples

```pycon
>>> server = tritonserver.Server(model_repository="/workspace/models")
server = tritonserver.Server(model_repository="/workspace/models")
```

```pycon
>>> server.start()
server.start()
```

<a id="tritonserver.Server.stop"></a>

#### stop()

Stop server and unload models

See c:func:TRITONSERVER_ServerStop

* **Return type:**
  [Server](#tritonserver.Server)

### Examples

```pycon
>>> server.stop()
server.stop()
```

<a id="tritonserver.Server.unload"></a>

#### unload(model: str | [tritonserver._api._model.Model](#tritonserver.Model), unload_dependents: bool = False, wait_until_unloaded: bool = False, polling_interval: float = 0.1, timeout: Optional[float] = None)

Unload model

Unloads a model and its dependents (optional).

See c:func:TRITONSERVER_ServerUnloadModel()

* **Parameters:**
  * **model** (*str* *|* [*Model*](#tritonserver.Model)) – model name or model object
  * **unload_dependents** (*bool*) – if True dependent models will also be unloaded
  * **wait_until_unloaded** (*bool*) – if True call will wait until model is unloaded before
    returning.
  * **polling_interval** (*float*) – time to wait in between polling if model is unloaded
  * **timeout** (*Optional**[**float**]*) – timeout to wait for the model to become unloaded
* **Raises:**
  **InvalidArgumentError** – if server is not started

### Examples

```pycon
>>> server.unload("new_name", wait_for_unloaded=True)
server.unload("new_name", wait_for_unloaded=True)
>>> server.models()
server.models()
{('new_name', 1): {'name': 'new_name', 'version': 1, 'state':
'UNAVAILABLE'}, ('resnet50_libtorch', -1): {'name':
'resnet50_libtorch', 'version': -1, 'state': None}, ('test_2',
-1): {'name': 'test_2', 'version': -1, 'state': None}}
```

<a id="tritonserver.Server.unregister_model_repository"></a>

#### unregister_model_repository(repository_path: str)

Unregister model repository

Only available when ModelControlMode is set to explicit

See c:func:TRITONSERVER_ServerUnregisterModelRepository

* **Parameters:**
  **repository_path** (*str*) – path to unregister
* **Return type:**
  [Server](#tritonserver.Server)

### Examples

```pycon
>>> options = tritonserver.Options()
>>> options.model_control_mode=tritonserver.ModelControlMode.EXPLICIT
>>> options.model_repository="/workspace/models"
>>> options.startup_models=["test"]
>>> server = tritonserver.Server(options)
>>> server.start()
>>> server.models()
{('resnet50_libtorch', -1): {'name': 'resnet50_libtorch',
'version': -1, 'state': None}, ('test', 1): {'name': 'test',
'version': 1, 'state': 'READY'}, ('test_2', -1): {'name':
'test_2', 'version': -1, 'state': None}}
>>> server.unregister_model_repository("/workspace/models")
>>> server.models()
{}
```

<a id="tritonserver.Options"></a>

### *class* tritonserver.Options(model_repository: str | list[str] = <factory>, server_id: str = 'triton', model_control_mode: ~tritonserver._c.triton_bindings.TRITONSERVER_ModelControlMode = <TRITONSERVER_ModelControlMode.NONE: 0>, startup_models: list[str] = <factory>, strict_model_config: bool = True, rate_limiter_mode: ~tritonserver._c.triton_bindings.TRITONSERVER_RateLimitMode = <TRITONSERVER_RateLimitMode.OFF: 0>, rate_limiter_resources: list[tritonserver._api._server.RateLimiterResource] = <factory>, pinned_memory_pool_size: int = 268435456, cuda_memory_pool_sizes: dict[int, int] = <factory>, cache_config: dict[str, dict[str, typing.Any]] = <factory>, cache_directory: str = '/opt/tritonserver/caches', min_supported_compute_capability: float = 6.0, exit_on_error: bool = True, strict_readiness: bool = True, exit_timeout: int = 30, buffer_manager_thread_count: int = 0, model_load_thread_count: int = 4, model_namespacing: bool = False, log_file: ~typing.Optional[str] = None, log_info: bool = False, log_warn: bool = False, log_error: bool = False, log_format: ~tritonserver._c.triton_bindings.TRITONSERVER_LogFormat = <TRITONSERVER_LogFormat.DEFAULT: 0>, log_verbose: int = 0, metrics: bool = True, gpu_metrics: bool = True, cpu_metrics: bool = True, metrics_interval: int = 2000, backend_directory: str = '/opt/tritonserver/backends', repo_agent_directory: str = '/opt/tritonserver/repoagents', model_load_device_limits: list[tritonserver._api._server.ModelLoadDeviceLimit] = <factory>, backend_configuration: dict[str, dict[str, str]] = <factory>, host_policies: dict[str, dict[str, str]] = <factory>, metrics_configuration: dict[str, dict[str, str]] = <factory>)

Bases: `object`

Server Options.

* **Parameters:**
  * **model_repository** (*str* *|* *list**[**str**]**,* *default* *[**]*) – Model repository path(s).
    At least one path is required.
    See `TRITONSERVER_ServerOptionsSetModelRepositoryPath()`
  * **server_id** (*str**,* *default 'triton'*) – Textural ID for the server.
    See `TRITONSERVER_ServerOptionsSetServerId()`
  * **model_control_mode** (*ModelControlMode**,* *default ModelControlModel.NONE*) –

    Model control mode.
    ModelControlMode.NONE : All models in the repository are loaded on startup.
    ModelControlMode.POLL : All models in the repository are loaded on startup.
    > Model repository changes can be applied using poll_model_repository.
    ModelControlMode.EXPLICIT
    : using model control APIs load_model, unload_model.

    See `TRITONSERVER_ServerOptionsSetModelControlMode()`
  * **startup_models** (*list**[**str**]**,* *default* *[**]*) – List of models to load at startup. Only relevant with ModelControlMode.EXPLICIT.
    See `TRITONSERVER_ServerOptionsSetStartupModel()`
  * **strict_model_config** (*bool**,* *default True*) – Enable or disable strict model configuration.
    See `TRITONSERVER_ServerOptionsSetStrictModelConfig()`
  * **rate_limiter_mode** (*RateLimitMode**,* *default RateLimitMode.OFF*) –

    Rate limit mode.
    RateLimitMode.EXEC_COUNT : Rate limiting prioritizes execution based on
    > the number of times each instance has run and if
    > resource constraints can be satisfied.

    RateLimitMode.OFF : Rate limiting is disabled.
    See `TRITONSERVER_ServerOptionsSetRateLimiterMode()`
  * **rate_limiter_resources** (*list**[**RateLimiterResource**]**,* *default* *[**]*) – Resource counts for rate limiting.
    See `TRITONSERVER_ServerOptionsAddRateLimiterResource()`
  * **pinned_memory_pool_size** (*uint**,* *default 1 << 28*) – Total pinned memory size.
    See `TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize()`
  * **cuda_memory_pool_sizes** (*dict**[**int**,* *uint**]**,* *default {}*) – Total CUDA memory pool size per device.
    See `TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize()`
  * **cache_config** (*dict**[**str**,* *dict**[**str**,* *Any**]**]**,* *default {}*) – Key-value configuration parameters for the cache provider.
    See `TRITONSERVER_ServerOptionsSetCacheConfig()`
  * **cache_directory** (*str**,* *default "/opt/tritonserver/caches"*) – Directory for cache provider implementations.
    See `TRITONSERVER_ServerOptionsSetCacheDirectory()`
  * **min_supported_compute_capability** (*float**,* *default 6.0*) – Minimum required CUDA compute capability.
    See `TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability()`
  * **exit_on_error** (*bool**,* *default True*) – Whether to exit on an initialization error.
    See `TRITONSERVER_ServerOptionsSetExitOnError()`
  * **strict_readiness** (*bool**,* *default True*) – Enable or disable strict readiness handling.
    See `TRITONSERVER_ServerOptionsSetStrictReadiness()`
  * **exit_timeout** (*uint**,* *default 30*) – Exit timeout for the server in seconds.
    See `TRITONSERVER_ServerOptionsSetExitTimeout()`
  * **buffer_manager_thread_count** (*uint**,* *default 0*) – Number of threads used by the buffer manager.
    See `TRITONSERVER_ServerOptionsSetBufferManagerThreadCount()`
  * **model_load_thread_count** (*uint**,* *default 4*) – Number of threads used to load models concurrently.
    See `TRITONSERVER_ServerOptionsSetModelLoadThreadCount()`
  * **model_namespacing** (*bool**,* *default False*) – Enable or disable model namespacing.
    See `TRITONSERVER_ServerOptionsSetModelNamespacing()`
  * **log_file** (*Optional**[**str**]**,* *default None*) – Path to the log file. If None, logs are written to stdout.
    See `TRITONSERVER_ServerOptionsSetLogVerbose()`
  * **log_info** (*bool**,* *default False*) – Enable or disable logging of INFO level messages.
    See `TRITONSERVER_ServerOptionsSetLogInfo()`
  * **log_warn** (*bool**,* *default False*) – Enable or disable logging of WARNING level messages.
    See `TRITONSERVER_ServerOptionsSetLogWarn()`
  * **log_error** (*bool**,* *default False*) – Enable or disable logging of ERROR level messages.
    See `TRITONSERVER_ServerOptionsSetLogError()`
  * **log_format** (*LogFormat**,* *default LogFormat.DEFAULT*) – Log message format.
    See `TRITONSERVER_ServerOptionsSetLogFormat()`
  * **log_verbose** (*uint**,* *default 0*) – Verbose logging level. Level zero disables logging.
    See `TRITONSERVER_ServerOptionsSetLogVerbose()`
  * **metrics** (*bool**,* *default True*) – Enable or disable metric collection.
    See `TRITONSERVER_ServerOptionsSetMetrics()`
  * **gpu_metrics** (*bool**,* *default True*) – Enable or disable GPU metric collection.
    See `TRITONSERVER_ServerOptionsSetGpuMetrics()`
  * **cpu_metrics** (*bool**,* *default True*) – Enable or disable CPU metric collection.
    See `TRITONSERVER_ServerOptionsSetCpuMetrics()`
  * **metrics_interval** (*uint**,* *default 2000*) – Interval, in milliseconds, for metric collection.
    See `TRITONSERVER_ServerOptionsSetMetricsInterval()`
  * **backend_directory** (*str**,* *default "/opt/tritonserver/backends"*) – Directory containing backend implementations.
    See `TRITONSERVER_ServerOptionsSetBackendDirectory()`
  * **repo_agent_directory** (*str**,* *default "/opt/tritonserver/repoagents"*) – Directory containing repository agent implementations.
    See `TRITONSERVER_ServerOptionsSetRepoAgentDirectory()`
  * **model_load_device_limits** (*list**[**ModelLoadDeviceLimit**]**,* *default* *[**]*) – Device memory limits for model loading.
    See `TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit()`
  * **backend_configuration** (*dict**[**str**,* *dict**[**str**,* *str**]**]**,* *default {}*) – Configuration for backend providers.
    See `TRITONSERVER_ServerOptionsSetBackendConfig()`
  * **host_policies** (*dict**[**str**,* *dict**[**str**,* *str**]**]**,* *default {}*) – Host policies for given policy name
    See `TRITONSERVER_ServerOptionsSetHostPolicy()`
  * **metrics_configuration** (*dict**[**str**,* *dict**[**str**,* *str**]**]**,* *default {}*) – Configuration for metric reporting.
    See `TRITONSERVER_ServerOptionsSetMetricsConfig()`

### Notes

The class represents server options with various configurable parameters for Triton Inference Server.
Please refer to the Triton Inference Server documentation for more details on each option.

<a id="tritonserver.Options.backend_configuration"></a>

#### backend_configuration*: dict[str, dict[str, str]]*

<a id="tritonserver.Options.backend_directory"></a>

#### backend_directory*: str*

<a id="tritonserver.Options.buffer_manager_thread_count"></a>

#### buffer_manager_thread_count*: int*

<a id="tritonserver.Options.cache_config"></a>

#### cache_config*: dict[str, dict[str, Any]]*

<a id="tritonserver.Options.cache_directory"></a>

#### cache_directory*: str*

<a id="tritonserver.Options.cpu_metrics"></a>

#### cpu_metrics*: bool*

<a id="tritonserver.Options.cuda_memory_pool_sizes"></a>

#### cuda_memory_pool_sizes*: dict[int, int]*

<a id="tritonserver.Options.exit_on_error"></a>

#### exit_on_error*: bool*

<a id="tritonserver.Options.exit_timeout"></a>

#### exit_timeout*: int*

<a id="tritonserver.Options.gpu_metrics"></a>

#### gpu_metrics*: bool*

<a id="tritonserver.Options.host_policies"></a>

#### host_policies*: dict[str, dict[str, str]]*

<a id="tritonserver.Options.log_error"></a>

#### log_error*: bool*

<a id="tritonserver.Options.log_file"></a>

#### log_file*: Optional[str]*

<a id="tritonserver.Options.log_format"></a>

#### log_format*: TRITONSERVER_LogFormat*

<a id="tritonserver.Options.log_info"></a>

#### log_info*: bool*

<a id="tritonserver.Options.log_verbose"></a>

#### log_verbose*: int*

<a id="tritonserver.Options.log_warn"></a>

#### log_warn*: bool*

<a id="tritonserver.Options.metrics"></a>

#### metrics*: bool*

<a id="tritonserver.Options.metrics_configuration"></a>

#### metrics_configuration*: dict[str, dict[str, str]]*

<a id="tritonserver.Options.metrics_interval"></a>

#### metrics_interval*: int*

<a id="tritonserver.Options.min_supported_compute_capability"></a>

#### min_supported_compute_capability*: float*

<a id="tritonserver.Options.model_control_mode"></a>

#### model_control_mode*: TRITONSERVER_ModelControlMode*

<a id="tritonserver.Options.model_load_device_limits"></a>

#### model_load_device_limits*: list[tritonserver._api._server.ModelLoadDeviceLimit]*

<a id="tritonserver.Options.model_load_thread_count"></a>

#### model_load_thread_count*: int*

<a id="tritonserver.Options.model_namespacing"></a>

#### model_namespacing*: bool*

<a id="tritonserver.Options.model_repository"></a>

#### model_repository*: str | list[str]*

<a id="tritonserver.Options.pinned_memory_pool_size"></a>

#### pinned_memory_pool_size*: int*

<a id="tritonserver.Options.rate_limiter_mode"></a>

#### rate_limiter_mode*: TRITONSERVER_RateLimitMode*

<a id="tritonserver.Options.rate_limiter_resources"></a>

#### rate_limiter_resources*: list[tritonserver._api._server.RateLimiterResource]*

<a id="tritonserver.Options.repo_agent_directory"></a>

#### repo_agent_directory*: str*

<a id="tritonserver.Options.server_id"></a>

#### server_id*: str*

<a id="tritonserver.Options.startup_models"></a>

#### startup_models*: list[str]*

<a id="tritonserver.Options.strict_model_config"></a>

#### strict_model_config*: bool*

<a id="tritonserver.Options.strict_readiness"></a>

#### strict_readiness*: bool*

<a id="tritonserver.Model"></a>

### *class* tritonserver.Model(server: [Server](#tritonserver.Server), name: str, version: int = -1, state: Optional[str] = None, reason: Optional[str] = None)

Bases: `object`

Class for interacting with Triton Inference Server models

Model objects are returned from server factory methods and allow
users to query metadata and execute inference
requests.

<a id="tritonserver.Model.async_infer"></a>

#### async_infer(inference_request: Optional[[InferenceRequest](#tritonserver.InferenceRequest)] = None, raise_on_error: bool = True, \*\*kwargs: Unpack[[InferenceRequest](#tritonserver.InferenceRequest)])

Send an inference request to the model for execution

Sends an inference request to the model. Responses are
returned using an asyncio compatible iterator. See
c:func:TRITONSERVER_ServerInferAsync

* **Parameters:**
  * **inference_request** (*Optional**[*[*InferenceRequest*](#tritonserver.InferenceRequest)*]*) – inference request object. If not provided inference
    request will be created using remaining keyword
    arguments.
  * **raise_on_error** (*bool**,* *default True*) – if True iterator will raise an error on any response
    errors returned from the model. If False errors will be
    returned as part of the response object.
  * **kwargs** (*Unpack**[*[*InferenceRequest*](#tritonserver.InferenceRequest)*]*) – If a request object is not provided, a new object will be
    created with remaining keyword arguments. See
    InferenceRequest documentation for valid arguments.
* **Returns:**
  asyncio compatible iterator
* **Return type:**
  [AsyncResponseIterator](#tritonserver.AsyncResponseIterator)
* **Raises:**
  **InvalidArgumentError** – if any invalid arguments are provided

<a id="tritonserver.Model.batch_properties"></a>

#### batch_properties()

Returns the batch properties of the model

See `TRITONSERVER_ServerModelBatchProperties()`

* **Returns:**
  ModelBatchFlag.UNKNOWN or ModelBatchFlag.FIRST_DIM
* **Return type:**
  ModelBatchFlag

### Examples

```pycon
>>> server.model("resnet50_libtorch").batch_properties()
server.model("resnet50_libtorch").batch_properties()
<TRITONSERVER_ModelBatchFlag.FIRST_DIM: 2>
```

<a id="tritonserver.Model.config"></a>

#### config(config_version: int = 1)

Returns model configuration

See `TRITONSERVER_ServerModelConfiguration()`

* **Parameters:**
  **config_version** (*int*) – configuration version in case multiple are supported
* **Returns:**
  Dictionary of key value pairs for model configuration
* **Return type:**
  dict[str, Any]

### Examples

```pycon
>>> server.model("test").config()
server.model("test").config()
```

{‘name’: ‘test’, ‘platform’:
‘’, ‘backend’: ‘python’, ‘version_policy’: {‘latest’:
{‘num_versions’: 1}}, ‘max_batch_size’: 0, ‘input’: [{‘name’:
‘text_input’, ‘data_type’: ‘TYPE_STRING’, ‘format’:
‘FORMAT_NONE’, ‘dims’: [-1], ‘is_shape_tensor’: False,
‘allow_ragged_batch’: False, ‘optional’: True}, {‘name’:
‘fp16_input’, ‘data_type’: ‘TYPE_FP16’, ‘format’:
‘FORMAT_NONE’, ‘dims’: [-1, 1], ‘is_shape_tensor’: False,
‘allow_ragged_batch’: False, ‘optional’: True}], ‘output’:
[{‘name’: ‘text_output’, ‘data_type’: ‘TYPE_STRING’, ‘dims’:
[-1], ‘label_filename’: ‘’, ‘is_shape_tensor’: False},
{‘name’: ‘fp16_output’, ‘data_type’: ‘TYPE_FP16’, ‘dims’: [-1,
1], ‘label_filename’: ‘’, ‘is_shape_tensor’: False}],
‘batch_input’: [], ‘batch_output’: [], ‘optimization’:
{‘priority’: ‘PRIORITY_DEFAULT’, ‘input_pinned_memory’:
{‘enable’: True}, ‘output_pinned_memory’: {‘enable’: True},
‘gather_kernel_buffer_threshold’: 0, ‘eager_batching’: False},
‘instance_group’: [{‘name’: ‘test_2’, ‘kind’: ‘KIND_GPU’,
‘count’: 1, ‘gpus’: [0], ‘secondary_devices’: [], ‘profile’:
[], ‘passive’: False, ‘host_policy’: ‘’}],
‘default_model_filename’: ‘model.py’, ‘cc_model_filenames’:
{}, ‘metric_tags’: {}, ‘parameters’: {}, ‘model_warmup’: [],
‘model_transaction_policy’: {‘decoupled’: True}}

<a id="tritonserver.Model.create_request"></a>

#### create_request(\*\*kwargs: Unpack[[InferenceRequest](#tritonserver.InferenceRequest)])

Inference request factory method

Return an inference request object that can be used with
model.infer() ro model.async_infer()

* **Parameters:**
  **kwargs** (*Unpack**[*[*InferenceRequest*](#tritonserver.InferenceRequest)*]*) – Keyword arguments passed to InferenceRequest constructor. See
  InferenceRequest documentation for details.
* **Returns:**
  Inference request associated with this model
* **Return type:**
  [InferenceRequest](#tritonserver.InferenceRequest)

### Examples

```pycon
>>> server.model("test").create_request()
server.model("test").create_request()
InferenceRequest(model={'name': 'test', 'version': -1,
'state': None},
_server=<tritonserver._c.triton_bindings.TRITONSERVER_Server
object at 0x7f5827156bf0>, request_id=None, flags=0,
correlation_id=None, priority=0, timeout=0, inputs={},
parameters={}, output_memory_type=None,
output_memory_allocator=None, response_queue=None,
_serialized_inputs={})
```

<a id="tritonserver.Model.infer"></a>

#### infer(inference_request: Optional[[InferenceRequest](#tritonserver.InferenceRequest)] = None, raise_on_error: bool = True, \*\*kwargs: Unpack[[InferenceRequest](#tritonserver.InferenceRequest)])

Send an inference request to the model for execution

Sends an inference request to the model. Responses are
returned asynchronously using an iterator. See
c:func:TRITONSERVER_ServerInferAsync

* **Parameters:**
  * **inference_request** (*Optional**[*[*InferenceRequest*](#tritonserver.InferenceRequest)*]*) – inference request object. If not provided inference
    request will be created using remaining keyword
    arguments.
  * **raise_on_error** (*bool**,* *default True*) – if True iterator will raise an error on any response
    errors returned from the model. If False errors will be
    returned as part of the response object.
  * **kwargs** (*Unpack**[*[*InferenceRequest*](#tritonserver.InferenceRequest)*]*) – If a request object is not provided, a new object will be
    created with remaining keyword arguments. See
    InferenceRequest documentation for valid arguments.
* **Returns:**
  Response iterator
* **Return type:**
  [ResponseIterator](#tritonserver.ResponseIterator)
* **Raises:**
  **InvalidArgumentError** – if any invalid arguments are provided

### Examples

```pycon
>>> responses = server.model("test_2").infer(inputs={"text_input":["hello"]})
responses = list(server.model("test_2").infer(inputs={"text_input":["hello"]}))
```

```pycon
>>> response = responses[0]
print(response)
InferenceResponse(model={'name': 'test_2', 'version': 1,
'state': None},
_server=<tritonserver._c.triton_bindings.TRITONSERVER_Server
object at 0x7f5827156bf0>, request_id='', parameters={},
outputs={'text_output':
Tensor(data_type=<TRITONSERVER_DataType.BYTES: 13>,
shape=array([1]),
memory_buffer=MemoryBuffer(data_ptr=140003384498080,
memory_type=<TRITONSERVER_MemoryType.CPU: 0>,
memory_type_id=0, size=9, owner=array([ 5, 0, 0, 0, 104, 101,
108, 108, 111], dtype=int8)))}, error=None,
classification_label=None, final=False)
```

```pycon
>>> response.outputs["text_output"].to_bytes_array()
response.outputs["text_output"].to_bytes_array()
array([b'hello'], dtype=object)
```

<a id="tritonserver.Model.metadata"></a>

#### metadata()

Returns medatadata about a model and its inputs and outputs

See c:func:TRITONSERVER_ServerModelMetadata()

* **Returns:**
  Model metadata as a dictionary of key value pairs
* **Return type:**
  dict[str, Any]

### Examples

server.model(“test”).metadata()
{‘name’: ‘test’, ‘versions’: [‘1’], ‘platform’: ‘python’,
‘inputs’: [{‘name’: ‘text_input’, ‘datatype’: ‘BYTES’,
‘shape’: [-1]}, {‘name’: ‘fp16_input’, ‘datatype’: ‘FP16’,
‘shape’: [-1, 1]}], ‘outputs’: [{‘name’: ‘text_output’,
‘datatype’: ‘BYTES’, ‘shape’: [-1]}, {‘name’: ‘fp16_output’,
‘datatype’: ‘FP16’, ‘shape’: [-1, 1]}]}

<a id="tritonserver.Model.ready"></a>

#### ready()

Returns whether a model is ready to accept requests

See `TRITONSERVER_ServerModelIsReady()`

* **Returns:**
  True if model is ready. False otherwise.
* **Return type:**
  bool

### Examples

```pycon
>>> server.model("test").ready()
server.model("test").ready()
True
```

<a id="tritonserver.Model.statistics"></a>

#### statistics()

Returns model statistics

See `TRITONSERVER_ServerModelStatistics()`

* **Returns:**
  Dictionary of key value pairs representing model
  statistics
* **Return type:**
  dict[str, Any]

### Examples

```pycon
>>> server.model("test").statistics()
server.model("test").statistics()
```

{‘model_stats’: [{‘name’:
‘test’, ‘version’: ‘1’, ‘last_inference’: 1704731597736,
‘inference_count’: 2, ‘execution_count’: 2, ‘inference_stats’:
{‘success’: {‘count’: 2, ‘ns’: 3079473}, ‘fail’: {‘count’: 0, ‘ns’:
0}, ‘queue’: {‘count’: 2, ‘ns’: 145165}, ‘compute_input’: {‘count’: 2,
‘ns’: 124645}, ‘compute_infer’: {‘count’: 2, ‘ns’: 2791809},
‘compute_output’: {‘count’: 2, ‘ns’: 10240}, ‘cache_hit’: {‘count’: 0,
‘ns’: 0}, ‘cache_miss’: {‘count’: 0, ‘ns’: 0}}, ‘batch_stats’:
[{‘batch_size’: 1, ‘compute_input’: {‘count’: 2, ‘ns’: 124645},
‘compute_infer’: {‘count’: 2, ‘ns’: 2791809}, ‘compute_output’:
{‘count’: 2, ‘ns’: 10240}}], ‘memory_usage’: []}]}

<a id="tritonserver.Model.transaction_properties"></a>

#### transaction_properties()

Returns the transaction properties of the model

See `TRITONSERVER_ServerModelTransactionProperties()`

* **Returns:**
  ModelTxnPropertyFlag.ONE_TO_ONE or
  ModelTxnPropertyFlag.DECOUPLED
* **Return type:**
  ModelTxnPropertyFlag

### Examples

```pycon
>>> server.model("resnet50_libtorch").transaction_properties()
server.model("resnet50_libtorch").transaction_properties()
<TRITONSERVER_ModelTxnPropertyFlag.ONE_TO_ONE: 1>
```

<a id="tritonserver.InferenceRequest"></a>

### *class* tritonserver.InferenceRequest(model: ~tritonserver._api._model.Model, request_id: ~typing.Optional[str] = None, flags: int = 0, correlation_id: ~typing.Optional[~typing.Union[int, str]] = None, priority: int = 0, timeout: int = 0, inputs: dict[str, typing.Union[tritonserver._api._tensor.Tensor, typing.Any]] = <factory>, parameters: dict[str, str | int | bool] = <factory>, output_memory_type: ~typing.Optional[~typing.Union[tuple[tritonserver._c.triton_bindings.TRITONSERVER_MemoryType, int], ~tritonserver._c.triton_bindings.TRITONSERVER_MemoryType, tuple[tritonserver._api._dlpack.DLDeviceType, int], str]] = None, output_memory_allocator: ~typing.Optional[~tritonserver._api._allocators.MemoryAllocator] = None, response_queue: ~typing.Optional[~typing.Union[~_queue.SimpleQueue, ~asyncio.queues.Queue]] = None)

Bases: `object`

Dataclass representing an inference request.

Inference request objects are created using Model factory
methods. They contain input parameters and input data as well as
configuration for response output memory allocation.

See c:func:TRITONSERVER_InferenceRequest for more details

* **Parameters:**
  * **model** ([*Model*](#tritonserver.Model)) – Model instance associated with the inference request.
  * **request_id** (*Optional**[**str**]**,* *default None*) – Unique identifier for the inference request.
  * **flags** (*int**,* *default 0*) – Flags indicating options for the inference request.
  * **correlation_id** (*Optional**[**Union**[**int**,* *str**]**]**,* *default None*) – Correlation ID associated with the inference request.
  * **priority** (*int**,* *default 0*) – Priority of the inference request.
  * **timeout** (*int**,* *default 0*) – Timeout for the inference request in microseconds.
  * **inputs** (*Dict**[**str**,* *Union**[*[*Tensor*](#tritonserver.Tensor)*,* *Any**]**]**,* *default {}*) – Dictionary of input names and corresponding input tensors or data.
  * **parameters** (*Dict**[**str**,* *Union**[**str**,* *int**,* *bool**]**]**,* *default {}*) – Dictionary of parameters for the inference request.
  * **output_memory_type** (*Optional**[**DeviceOrMemoryType**]**,* *default None*) – output_memory_type : Optional[DeviceOrMemoryType], default
    None Type of memory to allocate for inference response
    output. If not provided memory type will be chosen based on
    backend / model preference with MemoryType.CPU as
    fallback. Memory type can be given as a string, MemoryType,
    tuple [MemoryType, memory_type_\_id], or tuple[DLDeviceType,
    device_id].
  * **output_memory_allocator** (*Optional**[*[*MemoryAllocator*](#tritonserver.MemoryAllocator)*]**,* *default None*) – Memory allocator to use for inference response output. If not
    provided default allocators will be used to allocate
    MemoryType.GPU or MemoryType.CPU memory as set in
    output_memory_type or as requested by backend / model
    preference.
  * **response_queue** (*Optional**[**Union**[**queue.SimpleQueue**,* *asyncio.Queue**]**]**,* *default None*) – Queue for asynchronous handling of inference responses. If
    provided Inference responses will be added to the queue in
    addition to the response iterator. Must be queue.SimpleQueue
    for non asyncio requests and asyncio.Queue for asyncio
    requests.

### Examples

# Creating a request explicitly

```pycon
>>> request = server.model("test").create_request()
request = server.model("test").create_request()
request.inputs["fp16_input"] = numpy.array([[1.0]]).astype(numpy.float16)
for response in server.model("test_2").infer(request):
   print(numpy.from_dlpack(response.outputs["fp16_output"]))
[[1.]]
```

# Creating a request implicitly

for response in server.model(“test_2”).infer(
: inputs={“fp16_input”: numpy.array([[1.0]]).astype(numpy.float16)}

):
: print(numpy.from_dlpack(response.outputs[“fp16_output”]))

[[1.]]

<a id="tritonserver.InferenceRequest.correlation_id"></a>

#### correlation_id*: Optional[Union[int, str]]* *= None*

<a id="tritonserver.InferenceRequest.flags"></a>

#### flags*: int* *= 0*

<a id="tritonserver.InferenceRequest.inputs"></a>

#### inputs*: dict[str, Union[[tritonserver._api._tensor.Tensor](#tritonserver.Tensor), Any]]*

<a id="tritonserver.InferenceRequest.model"></a>

#### model*: [Model](#tritonserver.Model)*

<a id="tritonserver.InferenceRequest.output_memory_allocator"></a>

#### output_memory_allocator*: Optional[[MemoryAllocator](#tritonserver.MemoryAllocator)]* *= None*

<a id="tritonserver.InferenceRequest.output_memory_type"></a>

#### output_memory_type*: Optional[Union[tuple[tritonserver._c.triton_bindings.TRITONSERVER_MemoryType, int], TRITONSERVER_MemoryType, tuple[tritonserver._api._dlpack.DLDeviceType, int], str]]* *= None*

<a id="tritonserver.InferenceRequest.parameters"></a>

#### parameters*: dict[str, str | int | bool]*

<a id="tritonserver.InferenceRequest.priority"></a>

#### priority*: int* *= 0*

<a id="tritonserver.InferenceRequest.request_id"></a>

#### request_id*: Optional[str]* *= None*

<a id="tritonserver.InferenceRequest.response_queue"></a>

#### response_queue*: Optional[Union[SimpleQueue, Queue]]* *= None*

<a id="tritonserver.InferenceRequest.timeout"></a>

#### timeout*: int* *= 0*

<a id="tritonserver.InferenceResponse"></a>

### *class* tritonserver.InferenceResponse(model: ~tritonserver._api._model.Model, request_id: ~typing.Optional[str] = None, parameters: dict[str, str | int | bool] = <factory>, outputs: dict[str, tritonserver._api._tensor.Tensor] = <factory>, error: ~typing.Optional[~tritonserver.TritonError] = None, classification_label: ~typing.Optional[str] = None, final: bool = False)

Bases: `object`

Dataclass representing an inference response.

Inference response objects are returned from response iterators
which are in turn returned from model inference methods. They
contain output data, output parameters, any potential errors
reported and a flag to indicate if the response is the final one
for a request.

See c:func:TRITONSERVER_InferenceResponse for more details

* **Parameters:**
  * **model** ([*Model*](#tritonserver.Model)) – Model instance associated with the response.
  * **request_id** (*Optional**[**str**]**,* *default None*) – Unique identifier for the inference request (if provided)
  * **parameters** (*dict**[**str**,* *str* *|* *int* *|* *bool**]**,* *default {}*) – Additional parameters associated with the response.
  * **outputs** (*dict* *[**str**,* [*Tensor*](#tritonserver.Tensor)*]**,* *default {}*) – Output tensors for the inference.
  * **error** (*Optional**[**TritonError**]**,* *default None*) – Error (if any) that occurred in the processing of the request.
  * **classification_label** (*Optional**[**str**]**,* *default None*) – Classification label associated with the inference. Not currently supported.
  * **final** (*bool**,* *default False*) – Flag indicating if the response is final

<a id="tritonserver.InferenceResponse.classification_label"></a>

#### classification_label*: Optional[str]* *= None*

<a id="tritonserver.InferenceResponse.error"></a>

#### error*: Optional[TritonError]* *= None*

<a id="tritonserver.InferenceResponse.final"></a>

#### final*: bool* *= False*

<a id="tritonserver.InferenceResponse.model"></a>

#### model*: [Model](#tritonserver.Model)*

<a id="tritonserver.InferenceResponse.outputs"></a>

#### outputs*: dict[str, [tritonserver._api._tensor.Tensor](#tritonserver.Tensor)]*

<a id="tritonserver.InferenceResponse.parameters"></a>

#### parameters*: dict[str, str | int | bool]*

<a id="tritonserver.InferenceResponse.request_id"></a>

#### request_id*: Optional[str]* *= None*

<a id="tritonserver.ResponseIterator"></a>

### *class* tritonserver.ResponseIterator(model: [Model](#tritonserver.Model), request: TRITONSERVER_InferenceRequest, user_queue: Optional[queue.SimpleQueue] = None, raise_on_error: bool = False)

Bases: `object`

Response iterator

Response iterators are returned from model inference methods and
allow users to process inference responses in the order they were
received for a request.

<a id="tritonserver.ResponseIterator.cancel"></a>

#### cancel()

Cancel an inflight request

Cancels an in-flight request. Cancellation is handled on a
best effort basis and may not prevent execution of a request
if it is already started or completed.

See c:func:TRITONSERVER_ServerInferenceRequestCancel

### Examples

responses = server.model(“test”).infer(inputs={“text_input”:[“hello”]})

responses.cancel()

<a id="tritonserver.AsyncResponseIterator"></a>

### *class* tritonserver.AsyncResponseIterator(model: [Model](#tritonserver.Model), request: TRITONSERVER_InferenceRequest, user_queue: Optional[Queue] = None, raise_on_error: bool = False, loop: Optional[AbstractEventLoop] = None)

Bases: `object`

Asyncio compatible response iterator

Response iterators are returned from model inference methods and
allow users to process inference responses in the order they were
received for a request.

<a id="tritonserver.AsyncResponseIterator.cancel"></a>

#### cancel()

Cancel an inflight request

Cancels an in-flight request. Cancellation is handled on a
best effort basis and may not prevent execution of a request
if it is already started or completed.

See c:func:TRITONSERVER_ServerInferenceRequestCancel

### Examples

responses = server.model(“test”).infer(inputs={“text_input”:[“hello”]})

responses.cancel()

<a id="tritonserver.Tensor"></a>

### *class* tritonserver.Tensor(data_type: TRITONSERVER_DataType, shape: Sequence[int], memory_buffer: [MemoryBuffer](#tritonserver.MemoryBuffer))

Bases: `object`

Class representing a Tensor.

* **Parameters:**
  * **data_type** (*DataType*) – Data type of the tensor.
  * **shape** (*Sequence**[**int**]*) – Shape of the tensor.
  * **memory_buffer** ([*MemoryBuffer*](#tritonserver.MemoryBuffer)) – Memory buffer containing the tensor data.

<a id="tritonserver.Tensor.data_ptr"></a>

#### *property* data_ptr*: int*

Get the pointer to the tensor’s data.

* **Returns:**
  The pointer to the tensor’s data.
* **Return type:**
  int

<a id="tritonserver.Tensor.data_type"></a>

#### data_type*: TRITONSERVER_DataType*

<a id="tritonserver.Tensor.from_bytes_array"></a>

#### *static* from_bytes_array(bytes_array: list[str] | list[bytes] | numpy.ndarray)

Create Triton BYTES Tensor from numpy array or list

Creates a Triton tensor of type BYTES from a list of strings,
bytes or a numpy array of type

```
object_
```

,

```
bytes_
```

, or

```
str_
```

. The
method allocates new host memory to store the serialized
tensor. For more details on the format of Triton BYTES Tensors
please see Triton Inference Server documentation.

* **Parameters:**
  **bytes_array** (*list**[**str* *|* *bytes**]* *|* *numpy.ndarray*) – an array like object to convert
* **Return type:**
  [Tensor](#tritonserver.Tensor)
* **Raises:**
  **InvalidArgumentError** – If the given object can not be converted.

### Examples

tensor = Tensor.from_bytes_array(numpy.array([“hello”]))

tensor = Tensor.from_bytes_array([“hello”])

<a id="tritonserver.Tensor.from_dlpack"></a>

#### *static* from_dlpack(obj: Any)

Create a tensor from a DLPack-compatible object.

* **Parameters:**
  **obj** (*Any*) – The DLPack-compatible object.
* **Returns:**
  A new tensor created from the DLPack-compatible object.
* **Return type:**
  [Tensor](#tritonserver.Tensor)

### Examples

tensor = Tensor.from_dlpack(numpy.array([0,1,2], dtype=numpy.float16))

tensor = Tensor.from_dlpack(torch.zeros(100, dtype=torch.float16))

<a id="tritonserver.Tensor.from_string_array"></a>

#### *static* from_string_array(string_array: list[str] | numpy.ndarray)

Create Triton BYTES Tensor from numpy array of strings or list of strings.

Creates a Triton tensor of type BYTES from a list of strings,
or numpy array of type

```
str_
```

. The
method allocates new host memory to store the serialized
tensor. For more details on the format of Triton BYTES Tensors
please see Triton Inference Server documentation.

* **Parameters:**
  **string_array** (*list**[**str**]* *|* *numpy.ndarray*) – an array like object to convert
* **Return type:**
  [Tensor](#tritonserver.Tensor)
* **Raises:**
  **InvalidArgumentError** – If the given object can not be converted.

### Examples

tensor = Tensor.from_string_array(numpy.array([“hello”]))

tensor = Tensor.from_string_array([“hello”])

<a id="tritonserver.Tensor.memory_buffer"></a>

#### memory_buffer*: [MemoryBuffer](#tritonserver.MemoryBuffer)*

<a id="tritonserver.Tensor.memory_type"></a>

#### *property* memory_type*: TRITONSERVER_MemoryType*

Get the memory type of the tensor.

* **Returns:**
  The memory type of the tensor.
* **Return type:**
  MemoryType

<a id="tritonserver.Tensor.memory_type_id"></a>

#### *property* memory_type_id*: int*

Get the ID representing the memory type of the tensor.

* **Returns:**
  The ID representing the memory type of the tensor.
* **Return type:**
  int

<a id="tritonserver.Tensor.shape"></a>

#### shape*: Sequence[int]*

<a id="tritonserver.Tensor.size"></a>

#### *property* size*: int*

Get the size of the tensor’s data in bytes.

* **Returns:**
  The size of the tensor’s data in bytes.
* **Return type:**
  int

<a id="tritonserver.Tensor.to_bytes_array"></a>

#### to_bytes_array()

Deserialize Triton BYTES Tensor into numpy array.

If memory is not on the host the tensor data will be copied to
the host before deserialization. For more details on the
format of Triton BYTES Tensors please see Triton Inference
Server documentation.

* **Returns:**
  A numpy array of objects representing the BYTES tensor.
* **Return type:**
  numpy.ndarray

### Examples

numpy_ndarray = response.outputs[“text_output”].to_bytes_array()

<a id="tritonserver.Tensor.to_device"></a>

#### to_device(device: tuple[tritonserver._c.triton_bindings.TRITONSERVER_MemoryType, int] | tritonserver._c.triton_bindings.TRITONSERVER_MemoryType | tuple[tritonserver._api._dlpack.DLDeviceType, int] | str)

Move the tensor to the specified device.

* **Parameters:**
  **device** (*DeviceOrMemoryType*) – The target device. Device can be specified as a string,
  MemoryType, tuple [MemoryType, memory_type_\_id], or
  tuple[DLDeviceType, device_id].
* **Returns:**
  The tensor moved to the specified device.
* **Return type:**
  [Tensor](#tritonserver.Tensor)

### Examples

tensor_cpu = tritonserver.Tensor.from_dlpack(numpy.array([0,1,2], dtype=numpy.float16))

# Different ways to specify the device

tensor_gpu = tensor_cpu.to_device(MemoryType.GPU)

tensor_gpu = tensor_cpu.to_device((MemoryType.GPU,0))

tensor_gpu = tensor_cpu.to_device((DLDeviceType.kDLCUDA,0))

tensor_gpu = tensor_cpu.to_device(“gpu”)

tensor_gpu = tensor_cpu.to_device(“gpu:0”)

ndarray_gpu = cupy.from_dlpack(tensor_gpu)

ndarray_gpu[0] = ndarray_gpu.mean()

tensor_cpu = tensor_gpu.to_device(“cpu”)

ndarray_cpu = numpy.from_dlpack(tensor_cpu)

assert ndarray_cpu[0] == ndarray_gpu[0]

<a id="tritonserver.Tensor.to_host"></a>

#### to_host()

Move the tensor to CPU memory from device memory

* **Returns:**
  The tensor moved to the CPU.
* **Return type:**
  [Tensor](#tritonserver.Tensor)

### Examples

tensor = Tensor.from_dlpack(torch.zeros(100, dtype=torch.float16).to(“cuda”))

numpy_nd_array = numpy.array(tensor.to_host())

<a id="tritonserver.Tensor.to_string_array"></a>

#### to_string_array()

Deserialize Triton BYTES Tensor into numpy array of strings.

If memory is not on the host the tensor data will be copied to
the host before deserialization. For more details on the
format of Triton BYTES Tensors please see Triton Inference
Server documentation.

* **Returns:**
  A numpy array of objects representing the BYTES tensor.
* **Return type:**
  numpy.ndarray

### Examples

numpy_ndarray = response.outputs[“text_output”].to_string_array()

<a id="tritonserver.MemoryBuffer"></a>

### *class* tritonserver.MemoryBuffer(data_ptr: int, memory_type: TRITONSERVER_MemoryType, memory_type_id: int, size: int, owner: Any)

Bases: `object`

Memory allocated for a Tensor.

This object does not own the memory but holds a reference to the
owner.

* **Parameters:**
  * **data_ptr** (*int*) – Pointer to the allocated memory.
  * **memory_type** (*MemoryType*) – memory type
  * **memory_type_id** (*int*) – memory type id (typically the same as device id)
  * **size** (*int*) – Size of the allocated memory in bytes.
  * **owner** (*Any*) – Object that owns or manages the memory buffer.  Allocated
    memory must not be freed while a reference to the owner is
    held.

### Examples

```pycon
>>> buffer = MemoryBuffer.from_dlpack(numpy.array([100],dtype=numpy.uint8))
```

<a id="tritonserver.MemoryBuffer.data_ptr"></a>

#### data_ptr*: int*

<a id="tritonserver.MemoryBuffer.from_dlpack"></a>

#### *static* from_dlpack(owner: Any)

<a id="tritonserver.MemoryBuffer.memory_type"></a>

#### memory_type*: TRITONSERVER_MemoryType*

<a id="tritonserver.MemoryBuffer.memory_type_id"></a>

#### memory_type_id*: int*

<a id="tritonserver.MemoryBuffer.owner"></a>

#### owner*: Any*

<a id="tritonserver.MemoryBuffer.size"></a>

#### size*: int*

<a id="tritonserver.MemoryAllocator"></a>

### *class* tritonserver.MemoryAllocator

Bases: `ABC`

Abstract interface to allow for custom memory allocation strategies

Classes implementing the MemoryAllocator interface have to provide
an allocate method returning MemoryBuffer objects.  A memory
allocator implementation does not need to match the requested
memory type or memory type id.

### Examples

class TorchAllocator(tritonserver.MemoryAllocator):
: def allocate(self,
  : > size,
    > memory_type,
    > memory_type_id):
    <br/>
    device = “cpu”
    <br/>
    if memory_type == tritonserver.MemoryType.GPU:
    : device = “cuda”
    <br/>
    tensor = torch.zeros(size,dtype=torch.uint8,device=device)
    return tritonserver.MemoryBuffer.from_dlpack(tensor)

<a id="tritonserver.MemoryAllocator.allocate"></a>

#### *abstract* allocate(size: int, memory_type: TRITONSERVER_MemoryType, memory_type_id: int)

Allocate memory buffer for tensor.

Note: A memory allocator implementation does not need to honor
the requested memory type or memory type id

* **Parameters:**
  * **size** (*int*) – number of bytes requested
  * **memory_type** (*MemoryType*) – type of memory requested (CPU, GPU, etc.)
  * **memory_type_id** (*int*) – memory type id requested (typically device id)
* **Returns:**
  memory buffer with requested size
* **Return type:**
  [MemoryBuffer](#tritonserver.MemoryBuffer)

### Examples

memory_buffer = allocator.allocate(100, MemoryType.CPU, 0)

<a id="tritonserver.MemoryType"></a>

### tritonserver.MemoryType

alias of `TRITONSERVER_MemoryType`

<a id="tritonserver.DataType"></a>

### tritonserver.DataType

alias of `TRITONSERVER_DataType`
