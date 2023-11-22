###############################################################################
# Copyright (c) 2022, Habana Labs Ltd.  All rights reserved.
###############################################################################
from ctypes import *
import os
import warnings
from enum import Enum

path = os.path.dirname(os.getenv('GC_KERNEL_PATH').split(':')[0])

def return_c_func(library_name, function_name):
    func = eval("cdll.LoadLibrary('{}').{}".format(library_name, function_name))
    return func


class TraceType(Enum):
    TraceAll = 1,
    TraceHost = 2,
    TraceDevice = 3,
    TraceTypeSize = 4


class TraceFormat(Enum):
    TraceFormatTEF = 1,
    TraceFormatSize = 2


class SynapseProfilerApi:

    def __init__(self):
        self.lib_file = 'libSynapse.so'
        self.full_path = os.path.join(path, self.lib_file)
        self.profiler_start_call = return_c_func(self.full_path, 'synProfilerStart')
        self.profiler_start_call.restype = c_int
        self.profiler_stop_call = return_c_func(self.full_path, 'synProfilerStop')
        self.profiler_stop_call.restype = c_int
        self.profiler_get_trace_call = return_c_func(self.full_path, 'synProfilerGetTrace')
        self.profiler_get_trace_call.restype = c_int
        self.profiler_sync_call = return_c_func(self.full_path, 'synDeviceSynchronize')
        self.profiler_sync_call.restype = c_int

    def profiler_start(self, trace_type: TraceType, device_id: int):
        int32_device_id = c_int32(device_id)
        int_trace_type = c_int(trace_type.value[0])
        import habana_frameworks.torch.utils.experimental as htexp
        htexp._set_profiler_tracer_memory(device_id)
        return self.profiler_start_call(int_trace_type, int32_device_id)

    def profiler_sync(self, device_id: int):
        int32_device_id = c_int32(device_id)
        return self.profiler_sync_call(int32_device_id)

    def profiler_stop(self, trace_type: TraceType, device_id: int):
        int32_device_id = c_int32(device_id)
        int_trace_type = c_int(trace_type.value[0])
        return self.profiler_stop_call(int_trace_type, int32_device_id)

    def profiler_get_trace_json(self, trace_type: TraceType, device_id: int):
        int32_device_id = c_int32(device_id)
        int_trace_type = c_int(trace_type.value[0])
        int_trace_format = c_int(TraceFormat.TraceFormatTEF.value[0])
        result = self.profiler_get_trace_call(int_trace_type, int32_device_id, int_trace_format, None, None, None)
        return result

    def profiler_get_trace(self, trace_type: TraceType, device_id: int):
        int32_device_id = c_int32(device_id)
        int_trace_type = c_int(trace_type.value[0])
        int_trace_format = c_int(TraceFormat.TraceFormatTEF.value[0])
        buffer_size = c_size_t(0)
        num_entries = c_size_t(0)
        self.profiler_get_trace_call(int_trace_type, int32_device_id, int_trace_format, None, byref(buffer_size), byref(num_entries))
        buffer = cast((c_byte * buffer_size.value)(), c_char_p)
        result = self.profiler_get_trace_call(int_trace_type, int32_device_id, int_trace_format, buffer,
                                              byref(buffer_size), byref(num_entries))
        with open('profiler_data.bin', 'wb') as f:
            f.write(buffer)

        return result
