from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import inspect
import json
import queue
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Dict, List

import _datautils
import _dlpack
import numpy
from tritonserver import _c as triton_bindings
from tritonserver._c import TRITONSERVER_MetricFamily as MetricFamily

# Rename module for exceptions to simplify stack trace
exceptions = [
    triton_bindings.TritonError,
    triton_bindings.NotFoundError,
    triton_bindings.UnknownError,
    triton_bindings.InternalError,
    triton_bindings.InvalidArgumentError,
    triton_bindings.UnavailableError,
    triton_bindings.UnsupportedError,
    triton_bindings.AlreadyExistsError,
]

for exception in exceptions:
    exception.__module__ = "tritonserver_api"
    globals()[exception.__name__] = exception


class ModelBatchFlag(triton_bindings.TRITONSERVER_ModelBatchFlag):
    pass


class ModelIndexFlag(triton_bindings.TRITONSERVER_ModelIndexFlag):
    pass


class ModelTxnPropertyFlag(triton_bindings.TRITONSERVER_ModelTxnPropertyFlag):
    pass


class MetricFormat(triton_bindings.TRITONSERVER_MetricFormat):
    pass


class ModelControlMode(triton_bindings.TRITONSERVER_ModelControlMode):
    pass


class RateLimitMode(triton_bindings.TRITONSERVER_RateLimitMode):
    pass


class LogFormat(triton_bindings.TRITONSERVER_LogFormat):
    pass


class InstanceGroupKind(triton_bindings.TRITONSERVER_InstanceGroupKind):
    pass


class MetricKind(triton_bindings.TRITONSERVER_MetricKind):
    pass


@dataclass
class RateLimiterResource:
    name: str
    count: Annotated[int, ctypes.c_uint]
    device: Annotated[int, ctypes.c_uint]


@dataclass
class ModelLoadDeviceLimit:
    kind: InstanceGroupKind
    device: Annotated[int, ctypes.c_uint]
    fraction: float


@dataclass
class Options:
    """Server Options

    Parameters
    ----------
    server_id: str
            Id for server.

    """

    server_id: str = "triton"
    model_repository_paths: List[str] = dataclasses.field(default_factory=list[str])
    model_control_mode: ModelControlMode = ModelControlMode.POLL
    startup_models: List[str] = dataclasses.field(default_factory=list[str])
    strict_model_config: bool = True

    rate_limiter_mode: RateLimitMode = RateLimitMode.OFF
    rate_limiter_resources: List[RateLimiterResource] = dataclasses.field(
        default_factory=list[RateLimiterResource]
    )

    pinned_memory_pool_size: Annotated[int, ctypes.c_uint] = 1 << 28
    cuda_memory_pool_sizes: Dict[
        Annotated[int, ctypes.c_uint], Annotated[int, ctypes.c_uint]
    ] = dataclasses.field(
        default_factory=dict[
            Annotated[int, ctypes.c_uint], Annotated[int, ctypes.c_uint]
        ]
    )

    #   response_cache_size: Annotated[int, ctypes.c_uint] = 0
    cache_config: Dict[str, Dict[str, str]] = dataclasses.field(
        default_factory=dict[str, dict[str, str]]
    )
    cache_directory: str = "/opt/tritonserver/caches"

    min_supported_compute_capability: float = 6.0

    exit_on_error: bool = True
    strict_readiness: bool = True
    exit_timeout: Annotated[int, ctypes.c_uint] = 30
    buffer_manager_thread_count: Annotated[int, ctypes.c_uint] = 0
    model_load_thread_count: Annotated[int, ctypes.c_uint] = 4
    model_namespacing: bool = False

    log_file: str = None
    log_info: bool = False
    log_warn: bool = False
    log_error: bool = False
    log_format: LogFormat = LogFormat.DEFAULT
    log_verbose: bool = False

    metrics: bool = True
    gpu_metrics: bool = True
    cpu_metrics: bool = True
    metrics_interval: Annotated[int, ctypes.c_uint] = 2000

    backend_directory: str = "/opt/tritonserver/backends"
    repo_agent_directory: str = "/opt/tritonserver/repoagents"
    model_load_device_limits: List[ModelLoadDeviceLimit] = dataclasses.field(
        default_factory=list[ModelLoadDeviceLimit]
    )
    backend_configuration: Dict[str, Dict[str, str]] = dataclasses.field(
        default_factory=dict[str, dict[str, str]]
    )
    host_policies: Dict[str, Dict[str, str]] = dataclasses.field(
        default_factory=dict[str, Dict[str, str]]
    )
    metrics_configuration: Dict[str, Dict[str, str]] = dataclasses.field(
        default_factory=dict[str, Dict[str, str]]
    )

    def create_server_options(self):
        options = triton_bindings.TRITONSERVER_ServerOptions()

        options.set_server_id(self.server_id)

        for model_repository_path in self.model_repository_paths:
            options.set_model_repository_path(model_repository_path)
        options.set_model_control_mode(self.model_control_mode)

        for startup_model in self.startup_models:
            options.set_startup_model(startup_model)

        options.set_strict_model_config(self.strict_model_config)
        options.set_rate_limiter_mode(self.rate_limiter_mode)

        for rate_limiter_resource in self.rate_limiter_resources:
            options.set_rate_limiter_resouces(
                rate_limiter_resource.name,
                rate_limiter_resource.count,
                tate_limiter_resource.device,
            )
        options.set_pinned_memory_pool_byte_size(self.pinned_memory_pool_size)

        for device, memory_size in self.cuda_memory_pool_sizes.items():
            options.set_cuda_memory_pool_byte_size(device, memory_size)
        for cache_name, settings in self.cache_config:
            options.set_cache_config(cache, json.dumps(settings))

        options.set_cache_directory(self.cache_directory)
        options.set_min_supported_compute_capability(
            self.min_supported_compute_capability
        )
        options.set_exit_on_error(self.exit_on_error)
        options.set_strict_readiness(self.strict_readiness)
        options.set_exit_timeout(self.exit_timeout)
        options.set_buffer_manager_thread_count(self.buffer_manager_thread_count)
        options.set_model_load_thread_count(self.model_load_thread_count)
        options.set_model_namespacing(self.model_namespacing)

        if self.log_file:
            options.set_log_file(self.log_file)

        options.set_log_info(self.log_info)
        options.set_log_warn(self.log_warn)
        options.set_log_error(self.log_error)
        options.set_log_format(self.log_format)
        options.set_log_verbose(self.log_verbose)
        options.set_metrics(self.metrics)
        options.set_cpu_metrics(self.cpu_metrics)
        options.set_gpu_metrics(self.gpu_metrics)
        options.set_metrics_interval(self.metrics_interval)
        options.set_backend_directory(self.backend_directory)
        options.set_repo_agent_directory(self.repo_agent_directory)

        for model_load_device_limit in self.model_load_device_limits:
            options.set_model_load_device_limit(
                model_load_device_limit.kind,
                model_load_device_limit.device,
                model_load_device_limit.fraction,
            )

        for host_policy, settings in self.host_policies.items():
            for setting_name, setting_value in settings.items():
                options.set_host_policy(host_policy, setting_name, setting_value)

        for config_name, settings in self.metrics_configuration.items():
            for setting_name, setting_value in settings.items():
                options.set_metrics_config(config_name, setting_name, setting_value)

        for backend, settings in self.backend_configuration.items():
            for setting_name, setting_value in settings.items():
                options.set_backend_config(backend, setting_name, setting_value)

        return options


class Server:
    class UnstartedServer(object):
        def __init__(self):
            pass

        def __getattribute__(self, name):
            raise triton_bindings.triton_bindings.InvalidArgumentError(
                "Server not started"
            )

        def __setattr__(self, name, value):
            raise triton_bindings.triton_bindings.InvalidArgumentError(
                "Server not started"
            )

    def __init__(self, options: Options = None, **kwargs):
        if options is None:
            options = Options(**kwargs)
        self._options = options
        self._server = Server.UnstartedServer()

    def start(self, blocking=False):
        self._server = triton_bindings.TRITONSERVER_Server(
            self._options.create_server_options()
        )
        while blocking and not self.is_ready():
            time.sleep(0.1)

    def stop(self):
        self._server.stop()
        self._server = Server.UnstartedServer()

    def unregister_model_repository(self, repository_path: str):
        self._server.unregister_model_repository(repository_path)

    def register_model_repository(
        self, repository_path: str, name_mapping: Dict[str, str]
    ):
        name_mapping_list = [
            triton_bindings.TRITONSERVER_Parameter(name, value)
            for name, value in name_mapping.items()
        ]

        self._server.register_model_repository(repository_path, name_mapping_list)

    def poll_model_repository(self):
        return self._server.poll_model_repository()

    def metadata(self):
        return json.loads(self._server.metadata().serialize_to_json())

    def is_live(self):
        return self._server.is_live()

    def is_ready(self):
        return self._server.is_ready()

    def get_model(self, model_name, model_version=-1):
        return Model(self._server, model_name, model_version)

    def models(self, ready=False):
        return self._model_index(ready)

    def _model_index(self, ready=False):
        models = json.loads(self._server.model_index(ready).serialize_to_json())

        for model in models:
            if "version" in model:
                model["version"] = int(model["version"])

        return [Model(self._server, **model) for model in models]

    def load_model(
        self, model_name: str, parameters: Dict[str, str | int | bool | bytes] = None
    ):
        if parameters:
            parameter_list = [
                triton_bindings.TRITONSERVER_Parameter(name, value)
                for name, value in parameters.items()
            ]
            self._server.load_model_with_parameters(model_name, parameter_list)
        else:
            self._server.load_model(model_name)
        return self.get_model(model_name)

    def unload_model(self, model_name: str):
        self._server.unload_model(model_name)

    def metrics(self, metric_format: MetricFormat = MetricFormat.PROMETHEUS):
        return self._server.metrics().formatted(metric_format)


class Model:
    def __init__(
        self,
        server: triton_bindings.TRITONSERVER_Server,
        name: str,
        version: int = None,
        state: str = None,
    ):
        self._name = name
        self._version = version
        self._server = server
        self._state = state

    def create_inference_request(self, **kwargs):
        return InferenceRequest(model=self, _server=self._server, **kwargs)

    def async_infer(
        self, inference_request: InferenceRequest = None, **kwargs
    ) -> AsyncResponseIterator:
        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )
        server_request, response_iterator = inference_request.create_server_request(
            use_async_iterator=True
        )
        self._server.infer_async(server_request)
        if inference_request.response_queue is None:
            return response_iterator

    def infer(
        self, inference_request: InferenceRequest = None, **kwargs
    ) -> ResponseIterator:
        if inference_request is None:
            inference_request = InferenceRequest(
                model=self, _server=self._server, **kwargs
            )
        server_request, response_iterator = inference_request.create_server_request()
        self._server.infer_async(server_request)
        if inference_request.response_queue is None:
            return response_iterator

    def metadata(self):
        return json.loads(
            self._server.model_metadata(self._name, self._version).serialize_to_json()
        )

    def is_ready(self):
        return self._server.model_is_ready(self._name, self._version)

    def batch_properties(self):
        flags, _ = self._server.model_batch_properties(self._name, self._version)
        return ModelBatchFlag(flags)

    def transaction_properties(self):
        txn_properties, _ = self._server.model_transaction_properties(
            self._name, self._version
        )
        return ModelTxnPropertyFlag(txn_properties)

    def statistics(self):
        return json.loads(
            self._server.model_statistics(self._name, self._version).serialize_to_json()
        )

    def config(self, config_version=1):
        return json.loads(
            self._server.model_config(
                self._name, self._version, config_version
            ).serialize_to_json()
        )

    def __str__(self):
        return "%s" % (
            {"name": self._name, "version": self._version, "state": self._state}
        )


class AsyncResponseIterator:
    def response_callback(self, response, flags, unused):
        try:
            response = InferenceResponse.set_from_server_response(
                self._server, response, flags
            )
            asyncio.run_coroutine_threadsafe(self._queue.put(response), self._loop)

        except Exception as e:
            triton_bindings.TRITONSERVER_LogMessage(
                triton_bindings.TRITONSERVER_LogLevel.ERROR,
                __file__,
                inspect.currentframe().f_lineno,
                str(e),
            )
            # raise e

    def __init__(self, server, loop=None, response_queue=None):
        self._server = server
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        if response_queue is None:
            response_queue = asyncio.Queue()
        self._queue = response_queue
        self._complete = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._complete:
            raise StopAsyncIteration
        response = await self._queue.get()
        self._complete = response.final
        return response


class ResponseIterator:
    def response_callback(self, response, flags, unused):
        self._queue.put(
            InferenceResponse.set_from_server_response(self._server, response, flags)
        )

    def __init__(self, server, response_queue: queue.SimpleQueue = None):
        if response_queue is None:
            response_queue = queue.SimpleQueue()
        self._queue = response_queue
        self._server = server
        self._complete = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._complete:
            raise StopIteration
        response = self._queue.get()
        self._complete = response.final
        return response


@dataclass
class InferenceResponse:
    request_id: str = None
    parameters: dict = dataclasses.field(default_factory=dict)
    outputs: dict = dataclasses.field(default_factory=dict)
    error: triton_bindings.TritonError = None
    classification_label: str = None
    final: bool = False
    _server: triton_bindings.TRITONSERVER_Server = None
    model: Model = None

    @staticmethod
    def set_from_server_response(server, response, flags):
        values = {}
        try:
            response.throw_if_response_error()
        except triton_bindings.TritonError as error:
            values["error"] = error

        if flags == triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL:
            values["final"] = True

        name, version = response.model
        values["model"] = Model(server, name, version)
        values["request_id"] = response.id
        parameters = {}
        for parameter_index in range(response.parameter_count):
            name, type_, value = response.parameter(parameter_index)
            parameters[name] = value
        values["parameters"] = parameters
        outputs = {}
        for output_index in range(response.output_count):
            (
                name,
                type_,
                shape,
                buffer_,
                byte_size,
                memory_type,
                memory_type_id,
                numpy_array,
            ) = response.output(output_index)
            if type_ == triton_bindings.TRITONSERVER_DataType.BYTES:
                numpy_array = InferenceRequest._deserialize_bytes_array(numpy_array)

            numpy_dtype = _datautils.TRITON_TO_NUMPY_DTYPE[type_]
            outputs[name] = numpy_array.view(numpy_dtype).reshape(shape)
        values["outputs"] = outputs
        values["_server"] = server

        # values["classification_label"] = response.output_classification_label()

        return InferenceResponse(**values)


@dataclass
class InferenceRequest:
    request_id: str = None
    flags: int = 0
    correlation_id: int | str = None
    priority: int = 0
    timeout: int = 0
    inputs: dict = dataclasses.field(default_factory=dict)
    parameters: dict = dataclasses.field(default_factory=dict)
    model: Model = None
    response_queue: queue.SimpleQueue | asyncio.Queue = None
    _server: triton_bindings.TRITONSERVER_Server = None
    _serialized_inputs: dict = dataclasses.field(default_factory=dict)

    def _release_request(self, request, flags, user_object):
        pass

    @staticmethod
    def _allocate_buffer(
        allocator, tensor_name, byte_size, memory_type, memory_type_id, user_object
    ):
        _buffer = numpy.empty(byte_size, numpy.byte)
        return (
            _buffer.ctypes.data,
            _buffer,
            triton_bindings.TRITONSERVER_MemoryType.CPU,
            0,
        )

    @staticmethod
    def _release_buffer(
        allocator, _buffer, user_object, byte_size, memory_type, memory_type_id
    ):
        # No-op
        pass

    @staticmethod
    def _allocator_start(allocator, user_object):
        pass

    @staticmethod
    def _query_preferred_memory_type(
        allocator, user_object, tensor_name, bytes_size, memory_type, memory_type_id
    ):
        return (triton_bindings.TRITONSERVER_MemoryType.CPU, 0)

    @staticmethod
    def _set_buffer_attributes(
        allocator, tensor_name, buffer_attributes, user_object, buffer_user_object
    ):
        buffer_attributes.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
        buffer_attributes.memory_type_id = 0
        buffer_attributes.byte_size = buffer_user_object.size
        return buffer_attributes

    @staticmethod
    def _deserialize_bytes_array(array):
        result = []
        _buffer = memoryview(array)
        offset = 0
        while offset < len(_buffer):
            (item_length,) = struct.unpack_from("@I", _buffer, offset)
            offset += 4
            result.append(_buffer[offset : offset + item_length].tobytes())
            offset += item_length
        return numpy.array(result, dtype=numpy.object_)

    _allocator = _datautils.NumpyAllocator().create_response_allocator()

    #    triton_bindings.TRITONSERVER_ResponseAllocator(
    #       _allocate_buffer, _release_buffer, _allocator_start
    #  )

    def _add_inputs(self, request):
        for name, value in self.inputs.items():
            memory_buffer = _datautils.MemoryBuffer.from_value(value)
            if memory_buffer.data_type == triton_bindings.TRITONSERVER_DataType.BYTES:
                # to ensure lifetime of array
                self._serialized_inputs[name] = memory_buffer.value
            request.add_input(name, memory_buffer.data_type, memory_buffer.shape)

            request.append_input_data_with_buffer_attributes(
                name, memory_buffer.buffer_, memory_buffer.buffer_attributes
            )

    def _set_callbacks(self, request, use_async_iterator=False):
        # allocator.set_buffer_attributes_function(InferenceRequest._set_buffer_attributes)
        # allocator.set_query_function(InferenceRequest._query_preferred_memory_type)
        if use_async_iterator:
            response_iterator = AsyncResponseIterator(
                self._server, response_queue=self.response_queue
            )
        else:
            response_iterator = ResponseIterator(
                self._server, response_queue=self.response_queue
            )
        request.set_release_callback(self._release_request, None)
        request.set_response_callback(
            InferenceRequest._allocator,
            None,
            response_iterator.response_callback,
            None,
        )
        return response_iterator

    def create_server_request(self, use_async_iterator=False):
        request = triton_bindings.TRITONSERVER_InferenceRequest(
            self._server, self.model._name, self.model._version
        )
        if self.request_id is not None:
            request.id = self.request_id
        request.priority_uint64 = self.priority
        request.timeout_microseconds = self.timeout
        if self.correlation_id is not None:
            if isinstance(self.correlation_id, int):
                request.correlation_id = self.correlation_id
            else:
                request.correlation_id_string = self.correlation_id
        request.flags = self.flags

        self._add_inputs(request)

        response_iterator = self._set_callbacks(request, use_async_iterator)

        return request, response_iterator


# MetricFamily = triton_bindings.TRITONSERVER_MetricFamily


class Metric(triton_bindings.TRITONSERVER_Metric):
    def __init__(self, family: MetricFamily, labels: Dict[str, str] = None):
        if labels is not None:
            parameters = [
                triton_bindings.TRITONSERVER_Parameter(name, value)
                for name, value in labels.items()
            ]
        else:
            parameters = []

        triton_bindings.TRITONSERVER_Metric.__init__(self, family, parameters)


def serve(options: Options = None, **kwargs):
    server = Server(options, **kwargs)
    server.start()
    return server
