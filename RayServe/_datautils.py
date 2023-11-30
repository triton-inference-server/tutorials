from __future__ import annotations

import ctypes
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, ClassVar, Dict, List

import _dlpack
import cupy
import numpy
from tritonserver import _c as triton_bindings

DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE = defaultdict(
    lambda: None,
    {
        _dlpack.DLDeviceType.kDLCUDA: triton_bindings.TRITONSERVER_MemoryType.GPU,
        _dlpack.DLDeviceType.kDLCPU: triton_bindings.TRITONSERVER_MemoryType.CPU,
    },
)

DLPACK_TO_TRITON_DTYPE = defaultdict(
    lambda: triton_bindings.TRITONSERVER_DataType.INVALID,
    {
        (_dlpack.DLDataTypeCode.kDLBool, 1): triton_bindings.TRITONSERVER_DataType.BOOL,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            16,
        ): triton_bindings.TRITONSERVER_DataType.FP16,
    },
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
        numpy.str_: triton_bindings.TRITONSERVER_DataType.BYTES,
        numpy.object_: triton_bindings.TRITONSERVER_DataType.BYTES,
    },
)

TRITON_TO_NUMPY_DTYPE = defaultdict(
    lambda: triton_bindings.TRITONSERVER_DataType.INVALID,
    {value: key for key, value in NUMPY_TO_TRITON_DTYPE.items()},
)


class MemoryAllocator(ABC):
    @abstractmethod
    def allocate(
        self,
        allocator,
        tensor_name,
        byte_size,
        memory_type,
        memory_type_id,
        user_object,
    ):
        pass

    @abstractmethod
    def release(
        self,
        allocator,
        buffer_,
        buffer_user_object,
        byte_size,
        memory_type,
        memory_type_id,
    ):
        pass

    @abstractmethod
    def start(self, allocator, user_object):
        pass

    def create_response_allocator(self):
        self._allocator = triton_bindings.TRITONSERVER_ResponseAllocator(
            self.allocate, self.release, self.start
        )

        if hasattr(self, "query_preferred_memory_type"):
            self._allocator.set_query_function(self.query_preferred_memory_type)

        if hasattr(self, "set_attributes"):
            self._allocator.set_buffer_attributes_function(self.set_buffer_attributes)
        return self._allocator


class NumpyAllocator(MemoryAllocator):
    def __init__(self):
        pass

    def start(self, allocator, user_object):
        pass

    def allocate(
        self,
        allocator,
        tensor_name,
        byte_size,
        memory_type,
        memory_type_id,
        user_object,
    ):
        _buffer = numpy.empty(byte_size, numpy.byte)
        return (
            _buffer.ctypes.data,
            _buffer,
            triton_bindings.TRITONSERVER_MemoryType.CPU,
            0,
        )

    def release(
        self,
        allocator,
        buffer_,
        buffer_user_object,
        byte_size,
        memory_type,
        memory_type_id,
    ):
        pass


class DLPackObject:
    def __init__(self, value):
        self._capsule = _dlpack.get_dlpack_capsule(value)
        self._tensor = _dlpack.get_managed_tensor(self._capsule).dl_tensor

    @property
    def byte_size(self):
        return _dlpack.get_byte_size(
            self._tensor.dtype, self._tensor.ndim, self._tensor.shape
        )

    @property
    def memory_type(self):
        return DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE[self._tensor.device.device_type]

    @property
    def memory_type_id(self):
        return self._tensor.device.device_id

    @property
    def shape(self):
        return [self._tensor.shape[i] for i in range(self._tensor.ndim)]

    @property
    def triton_data_type(self):
        return DLPACK_TO_TRITON_DTYPE[self.data_type]

    @property
    def data_type(self):
        return (self._tensor.dtype.type_code, self._tensor.dtype.bits)

    @property
    def data_ptr(self):
        return self._tensor.data + self._tensor.byte_offset


@dataclass
class MemoryBuffer:
    data_type: triton_bindings.TRITONSERVER_DataType
    shape: List[int]
    buffer_attributes: triton_bindings.TRITONSERVER_BufferAttributes
    buffer_: ctypes.c_void_p
    value: object

    @staticmethod
    def from_value(value):
        if type(value) in MemoryBuffer._supported_conversions:
            return MemoryBuffer._supported_conversions[type(value)](value)
        elif hasattr(value, "__dlpack__"):
            return MemoryBuffer._from_dlpack(value)
        else:
            raise triton_bindings.InvalidArgumentError(
                f"Input type {type(value)} not supported"
            )

    @staticmethod
    def _from_dlpack(value):
        dlpack_object = DLPackObject(value)
        data_type = dlpack_object.triton_data_type
        if data_type == triton_bindings.TRITONSERVER_DataType.INVALID:
            raise triton_bindings.InvalidArgumentError(
                f"DLPack dtype {dlpack_object.data_type} not supported"
            )

        if data_type == triton_bindings.TRITONSERVER_DataType.BYTES:
            raise triton_bindings.InvalidArgumentError(
                f"DLPack does not support {data_type}"
            )

        shape = dlpack_object.shape

        buffer_ = dlpack_object.data_ptr

        buffer_attributes = triton_bindings.TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type = dlpack_object.memory_type
        buffer_attributes.memory_type_id = dlpack_object.memory_type_id
        buffer_attributes.byte_size = dlpack_object.byte_size

        return MemoryBuffer(data_type, shape, buffer_attributes, buffer_, value)

    @staticmethod
    def _serialize_numpy_bytes_array(array):
        result = []
        for array_item in numpy.nditer(array, flags=["refs_ok"], order="C"):
            item = array_item.item()
            if type(item) != bytes:
                item = str(item).encode("utf-8")
            result.append(struct.pack("@I", len(item)))
            result.append(item)
        return numpy.frombuffer(b"".join(result), dtype=numpy.byte)

    @staticmethod
    def _from_numpy(value: numpy.ndarray | numpy.generic) -> MemoryBuffer:
        data_type = NUMPY_TO_TRITON_DTYPE[value.dtype.type]
        if data_type == triton_bindings.TRITONSERVER_DataType.INVALID:
            raise triton_bindings.InvalidArgumentError(
                f"Numpy type {value.dtype.type} not supported"
            )
        shape = value.shape
        if data_type == triton_bindings.TRITONSERVER_DataType.BYTES:
            value = MemoryBuffer._serialize_numpy_bytes_array(value)
        buffer_ = value.ctypes.data
        buffer_attributes = triton_bindings.TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type = triton_bindings.TRITONSERVER_MemoryType.CPU
        buffer_attributes.memory_type_id = 0
        buffer_attributes.byte_size = value.itemsize * value.size

        return MemoryBuffer(data_type, shape, buffer_attributes, buffer_, value)

    _supported_conversions: ClassVar[Dict] = dict(
        {
            numpy.ndarray: _from_numpy,
            numpy.generic: _from_numpy,
        },
    )
