import asyncio
import dataclasses
import json
import queue
import struct
from collections import defaultdict
from dataclasses import dataclass

import numpy
from tritonserver import _c as triton_bindings


# @dataclass
class ServerOptions:
    def __init__():
        pass


# @dataclass
class Tensor:
    pass

    def __dlpack__(self, stream=None):
        pass

    def __dlpack_device__(self):
        pass


class TritonServer:
    class Options:
        pass

    def __init__(self, options):
        pass

    def start(self):
        options = triton_bindings.TRITONSERVER_ServerOptions()
        options.set_model_repository_path("/workspace/models")
        options.set_model_control_mode(
            triton_bindings.TRITONSERVER_ModelControlMode.POLL
        )
        options.set_log_verbose(0)
        options.set_log_info(0)
        options.set_log_warn(0)
        options.set_exit_timeout(5)
        self._server = triton_bindings.TRITONSERVER_Server(options)

    def model(self, name, version=-1):
        return TritonModel(self._server, name, version)

    def model_index(self, ready=False):
        models = json.loads(self._server.model_index(ready).serialize_to_json())
        return [
            TritonModel(
                self._server, model["name"], int(model["version"]), model["state"]
            )
            for model in models
        ]

    #        return json.loads(self._server.model_index(ready).serialize_to_json())

    def load_model():
        pass

    def unload_model():
        pass

    def stop():
        pass


class TritonModel:
    def __init__(
        self,
        server: triton_bindings.TRITONSERVER_Server,
        name: str,
        version: int,
        state: str = None,
    ):
        self._name = name
        self._version = version
        self._server = server
        self._state = state

    def inference_request(self, **kwargs):
        kwargs["model"] = self
        kwargs["_server"] = self._server
        return InferenceRequest(**kwargs)

    def infer_async(self, inference_request=None, **kwargs):
        if inference_request is None:
            kwargs["model"] = self
            kwargs["_server"] = self._server
            inference_request = InferenceRequest(**kwargs)
        server_request, response_iterator = inference_request.create_server_request()
        self._server.infer_async(server_request)
        return response_iterator

    def metadata(self):
        return json.loads(
            self._server.model_metadata(self._name, self._version).serialize_to_json()
        )

    def ready(self):
        return self._server.model_is_ready(self._name, self._version)

    def batch_properties():
        pass

    def __str__(self):
        return "%s" % (
            {"name": self._name, "version": self._version, "state": self._state}
        )


NUMPY_TO_TRITON_DTYPE = defaultdict(
    lambda: triton_bindings.TRITONSERVER_DataType.INVALID,
    {
        bool: triton_bindings.TRITONSERVER_DataType.BOOL,
        numpy.int8: triton_bindings.triton_bindings.TRITONSERVER_DataType.INT8,
        numpy.int16: triton_bindings.TRITONSERVER_DataType.INT16,
        numpy.int32: triton_bindings.TRITONSERVER_DataType.INT32,
        numpy.int64: triton_bindings.TRITONSERVER_DataType.INT64,
        numpy.uint8: triton_bindings.TRITONSERVER_DataType.UINT8,
        numpy.uint16: triton_bindings.TRITONSERVER_DataType.UINT16,
        numpy.uint32: triton_bindings.TRITONSERVER_DataType.UINT32,
        numpy.uint64: triton_bindings.TRITONSERVER_DataType.UINT64,
        numpy.float16: triton_bindings.TRITONSERVER_DataType.FP16,
        numpy.float32: triton_bindings.TRITONSERVER_DataType.FP32,
        numpy.float64: triton_bindings.TRITONSERVER_DataType.FP64,
        numpy.bytes_: triton_bindings.TRITONSERVER_DataType.BYTES,
        numpy.object_: triton_bindings.TRITONSERVER_DataType.BYTES,
    },
)

TRITON_TO_NUMPY_DTYPE = defaultdict(
    lambda: triton_bindings.TRITONSERVER_DataType.INVALID,
    {value: key for key, value in NUMPY_TO_TRITON_DTYPE.items()},
)


class AsyncResponseIterator:
    @staticmethod
    def response_callback(response, flags, self):
        response = InferenceResponse.set_from_server_response(self._server, response)

        asyncio.run_coroutine_threadsafe(self._queue.put(response), self._loop)

        if flags == triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL:
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)

    def __init__(self, server, loop):
        self._server = server
        self._loop = loop
        self._queue = asyncio.Queue()

    async def __aiter__(self):
        return self

    async def __anext__(self):
        response = self._queue.get()
        if response is None:
            raise StopAsyncIteration
        return response


class ResponseIterator:
    @staticmethod
    def response_callback(response, flags, self):
        self._queue.put(
            InferenceResponse.set_from_server_response(self._server, response)
        )
        if flags == triton_bindings.TRITONSERVER_ResponseCompleteFlag.FINAL:
            self._queue.put(None)

    def __init__(self, server):
        self._queue = queue.SimpleQueue()
        self._server = server

    #        self._request = request

    def __iter__(self):
        return self

    def __next__(self):
        response = self._queue.get()
        if response is None:
            raise StopIteration
        return response


@dataclass
class InferenceResponse:
    id: str = None
    parameters: dict = dataclasses.field(default_factory=dict)
    outputs: dict = dataclasses.field(default_factory=dict)
    error: triton_bindings.TritonError = None
    classification_label: str = None
    _server: triton_bindings.TRITONSERVER_Server = None
    model: TritonModel = None

    @staticmethod
    def set_from_server_response(server, response):
        values = {}

        try:
            response.throw_if_response_error()
        except triton_bindings.TritonError as error:
            values["error"] = error

        name, version = response.model
        values["model"] = TritonModel(server, name, version)
        values["id"] = response.id
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

            numpy_dtype = TRITON_TO_NUMPY_DTYPE[type_]
            outputs[name] = numpy_array.view(numpy_dtype).reshape(shape)
        values["outputs"] = outputs
        values["_server"] = server

        # values["classification_label"] = response.output_classification_label()

        return InferenceResponse(**values)


@dataclass
class InferenceRequest:
    id: str = None
    flags: int = 0
    correlation_id: int | str = None
    priority: int = 0
    timeout: int = 0
    inputs: dict = dataclasses.field(default_factory=dict)
    parameters: dict = dataclasses.field(default_factory=dict)
    model: TritonModel = None
    _server: triton_bindings.TRITONSERVER_Server = None
    _serialized_inputs: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def _release_request(request, flags, user_object):
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

    @staticmethod
    def _serialize_bytes_array(array):
        result = []
        for array_item in numpy.nditer(array, flags=["refs_ok"], order="C"):
            item = array_item.item()
            if type(item) != bytes:
                item = str(item).encode("utf-8")
            result.append(struct.pack("@I", len(item)))
            result.append(item)
        return numpy.frombuffer(b"".join(result), dtype=numpy.byte)

    _allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
        _allocate_buffer, _release_buffer, _allocator_start
    )

    def _add_inputs(self, request):
        for name, value in self.inputs.items():
            if not isinstance(value, (numpy.ndarray, numpy.generic)):
                raise Exception("Invalid Argument")

            triton_datatype = NUMPY_TO_TRITON_DTYPE[value.dtype.type]

            if triton_datatype == triton_bindings.TRITONSERVER_DataType.INVALID:
                raise Exception("Invalid Argument")

            request.add_input(name, triton_datatype, value.shape)

            if triton_datatype == triton_bindings.TRITONSERVER_DataType.BYTES:
                value = InferenceRequest._serialize_bytes_array(value)
                # to ensure lifetime of array
                self._serialized_inputs[name] = value

            _buffer = value.ctypes.data
            buffer_attributes = triton_bindings.TRITONSERVER_BufferAttributes()
            buffer_attributes.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
            buffer_attributes.memory_type_id = 0
            buffer_attributes.byte_size = value.itemsize * value.size
            request.append_input_data_with_buffer_attributes(
                name, _buffer, buffer_attributes
            )

    def _set_callbacks(self, request):
        # allocator.set_buffer_attributes_function(InferenceRequest._set_buffer_attributes)
        # allocator.set_query_function(InferenceRequest._query_preferred_memory_type)
        response_iterator = ResponseIterator(self._server)
        request.set_release_callback(InferenceRequest._release_request, self)
        request.set_response_callback(
            InferenceRequest._allocator,
            None,
            ResponseIterator.response_callback,
            response_iterator,
        )
        return response_iterator

    def create_server_request(self):
        request = triton_bindings.TRITONSERVER_InferenceRequest(
            self._server, self.model._name, self.model._version
        )
        if self.id is not None:
            request.id = self.id
        request.priority_uint64 = self.priority
        request.timeout_microseconds = self.timeout
        if self.correlation_id is not None:
            if isinstance(self.correlation_id, int):
                request.correlation_id = self.correlation_id
            else:
                request.correlation_id_string = self.correlation_id
        request.flags = self.flags

        self._add_inputs(request)

        response_iterator = self._set_callbacks(request)

        return request, response_iterator
