# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import atexit
import ctypes
import signal
import sys
from typing import Optional, Union, Iterable, Any

from asc.runtime import config
from . import state, support


def _sigint_handler(sig, frame):
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(signal.SIGINT)


def _lazy_init(need_device=True, need_stream=True) -> None:
    if state.lib is None:
        state.load_lib()
        atexit.register(reset_device, None)
        signal.signal(signal.SIGINT, _sigint_handler)
    if not need_device:
        return
    if state.device_id is None:
        device_id = default_device()
        set_device(device_id)
    if not need_stream:
        return
    if state.streams[state.device_id] is None:
        state.streams[state.device_id] = create_stream()


def _select_device_id(device_id: Optional[int] = None) -> int:
    if device_id is None:
        return current_device()
    check_device_handle(device_id)
    return device_id


def use_model(custom_lib_prefix: Optional[str] = None) -> None:
    state.model = True
    state.custom_lib_prefix = custom_lib_prefix


def use_npu(custom_lib_prefix: Optional[str] = None) -> None:
    state.model = False
    state.custom_lib_prefix = custom_lib_prefix


def set_soc_version(soc_verison: config.Platform):
    state.soc_verison = soc_verison


def get_soc_version() -> config.Platform:
    return state.soc_verison


def is_initialized() -> bool:
    return state.lib is not None


def is_available() -> bool:
    try:
        _lazy_init(need_device=False)
    except Exception:
        state.lib = None
        return False
    return True


def is_model() -> bool:
    return state.model


def magic_elf_value(core_type: support.CoreType) -> int:
    if not isinstance(core_type, support.CoreType):
        raise RuntimeError("Core type is not supported!")
    if core_type == support.CoreType.AiCore:
        return support.MagicElf.RT_DEV_BINARY_MAGIC_ELF.value
    elif core_type == support.CoreType.VectorCore:
        return support.MagicElf.RT_DEV_BINARY_MAGIC_ELF_AIVEC.value
    elif core_type == support.CoreType.CubeCore:
        return support.MagicElf.RT_DEV_BINARY_MAGIC_ELF_AICUBE.value
    else:
        return support.MagicElf.RT_DEV_BINARY_MAGIC_ELF_AICPU.value


def msprof_task_type(core_type: support.CoreType) -> int:
    if not isinstance(core_type, support.CoreType):
        raise RuntimeError("Core type is not supported!")
    if core_type == support.CoreType.AiCore:
        return support.MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
    elif core_type == support.CoreType.VectorCore:
        return support.MsprofTaskType.MSPROF_TASK_TYPE_AIV.value
    elif core_type == support.CoreType.CubeCore:
        return support.MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
    else:
        return support.MsprofTaskType.MSPROF_TASK_TYPE_AI_CPU.value


def check_device_handle(device_id: int) -> None:
    if isinstance(device_id, int) and device_id >= 0:
        return
    raise RuntimeError(f"Device handle should be int >= 0, but '{device_id}' with {type(device_id)} type was given")


def default_device() -> int:
    return 0


def current_device() -> int:
    _lazy_init(need_stream=False)
    return state.device_id


def current_stream(device_id: Optional[int] = None) -> support.Stream:
    _lazy_init()
    return state.streams[_select_device_id(device_id)]


def current_platform() -> str:
    _lazy_init(need_device=False)
    max_len = 32
    soc_version = (ctypes.c_char * max_len)(0)
    state.lib.call(
        "GetSocVersionWrapper",
        ctypes.c_char_p(ctypes.addressof(soc_version)),
        ctypes.c_uint32(max_len),
    )
    return bytes(soc_version.value).decode("utf-8")


def device_count() -> int:
    _lazy_init(need_device=False)
    count = ctypes.c_int32(0)
    state.lib.call(
        "GetDeviceCountWrapper",
        ctypes.c_void_p(ctypes.addressof(count)),
    )
    return int(count.value)


def device_info(module_type: support.DeviceModuleType, info_type: support.DeviceInfoType,
                device_id: Optional[int] = None) -> int:
    _lazy_init(need_stream=False)
    device_id = _select_device_id(device_id)
    result = ctypes.c_int64(0)
    state.lib.call(
        "GetDeviceInfoWrapper",
        ctypes.c_uint32(device_id),
        ctypes.c_int32(module_type.value),
        ctypes.c_int32(info_type.value),
        ctypes.c_void_p(ctypes.addressof(result)),
    )
    return int(result.value)


def free_mem() -> None:
    _lazy_init()
    for kernel_handle in tuple(state.kernels.keys()):
        unregister_device_binary_kernel(ctypes.c_void_p(kernel_handle))
    for mem_handle in tuple(state.allocs.keys()):
        free(ctypes.c_void_p(mem_handle))


def reset_device(device_id: Optional[int] = None) -> None:
    device_id = _select_device_id(device_id)
    if device_id is None:
        return
    _lazy_init()
    for device_id, stream in state.streams.items():
        destroy_stream(stream)
        state.streams[device_id] = None
    state.lib.call(
        "DeviceResetWrapper",
        ctypes.c_int32(state.device_id),
    )
    if device_id == state.device_id:
        state.device_id = None


def set_device(device_id: int) -> None:
    check_device_handle(device_id)
    if state.device_id is not None:
        reset_device(state.device_id)
    _lazy_init(need_device=False)
    state.lib.call(
        "SetDeviceWrapper",
        ctypes.c_int32(device_id),
    )
    state.device_id = device_id
    state.streams[device_id] = None


def create_stream(priority: int = 0) -> support.Stream:
    _lazy_init(need_stream=False)
    stream_handle = ctypes.c_void_p()
    state.lib.call(
        "StreamCreateWrapper",
        ctypes.c_void_p(ctypes.addressof(stream_handle)),
        ctypes.c_int32(priority),
    )
    return stream_handle


def destroy_stream(stream_handle: support.Stream) -> None:
    if stream_handle is None:
        return
    _lazy_init()
    state.lib.call(
        "StreamDestroyWrapper",
        stream_handle,
    )


def register_device_binary_kernel(kernel_binary: bytes, core_type_id: int) -> support.Kernel:
    _lazy_init()
    kernel_size = len(kernel_binary)
    if kernel_size <= 0:
        raise RuntimeError("Kernel size must be greater than 0 when register device binary kernel!")
    device_binary = support.DevBinary(
        data=ctypes.c_char_p(kernel_binary),
        length=ctypes.c_uint64(kernel_size),
        version=ctypes.c_uint32(0),
        magic=ctypes.c_uint32(core_type_id),
    )
    handle = ctypes.c_void_p()
    state.lib.call(
        "DevBinaryRegisterWrapper",
        ctypes.c_void_p(ctypes.addressof(device_binary)),
        ctypes.c_void_p(ctypes.addressof(handle)),
    )
    state.kernels[handle.value] = kernel_binary
    return handle


def unregister_device_binary_kernel(kernel_handle: support.Kernel):
    _lazy_init()
    state.lib.call(
        "DevBinaryUnRegisterWrapper",
        kernel_handle,
    )
    del state.kernels[kernel_handle.value]


def register_function(kernel_handle: support.Kernel, fn_name: str, mode: int) -> support.Function:
    _lazy_init()
    fn_name_bytes = fn_name.encode("utf-8")
    name_ptr = ctypes.c_char_p(fn_name_bytes)
    state.lib.call(
        "FunctionRegisterWrapper",
        kernel_handle,
        name_ptr,
        name_ptr,
        name_ptr,
        ctypes.c_uint32(mode),
    )
    return ctypes.cast(name_ptr, ctypes.c_void_p)


def malloc(size: int, memory_type: support.MemoryType = support.MemoryType.RT_MEMORY_DEFAULT,
           policy: support.MallocPolicy = support.MallocPolicy.RT_MEMORY_POLICY_NONE) -> support.Memory:
    if size <= 0:
        raise RuntimeError("Malloc size must be greater than 0!")
    _lazy_init()
    real_mem_size = size + 512
    c_memory_p = ctypes.c_void_p()
    state.lib.call(
        "MallocWrapper",
        ctypes.c_void_p(ctypes.addressof(c_memory_p)),
        ctypes.c_uint64(real_mem_size),
        memory_type.value | policy.value,
        ctypes.c_int64(33),
    )
    mem_handle = ctypes.c_void_p(512 * ((c_memory_p.value + 512 - 1) // 512))
    state.allocs[mem_handle.value] = c_memory_p.value
    return mem_handle


def memcpy(mem_dst_handle: support.Memory, dst_nbytes: int, mem_src_handle: support.Memory, src_nbytes: int,
           kind: support.MemcpyKind) -> None:
    if dst_nbytes <= 0 or src_nbytes <= 0:
        raise RuntimeError("Memcopy src and dst size must be greater than 0!")
    _lazy_init()
    state.lib.call(
        "MemcpyWrapper",
        ctypes.cast(mem_dst_handle, ctypes.c_void_p),
        ctypes.c_uint64(dst_nbytes),
        ctypes.cast(mem_src_handle, ctypes.c_void_p),
        ctypes.c_uint64(src_nbytes),
        kind.value,
    )


def copy_data_to_device(src_mem_handle: support.Memory, nbytes: int) -> support.Memory:
    _lazy_init()
    dst_mem_handle = malloc(
        nbytes, support.MemoryType.RT_MEMORY_HBM, support.MallocPolicy.RT_MEMORY_POLICY_HUGE_PAGE_ONLY
        if nbytes > 2048 else support.MallocPolicy.RT_MEMORY_POLICY_NONE)
    memcpy(dst_mem_handle, nbytes, src_mem_handle, nbytes, support.MemcpyKind.RT_MEMCPY_HOST_TO_DEVICE)
    return dst_mem_handle


def copy_data_from_device(mem_dst_handle: support.Memory, mem_src_handle: support.Memory, nbytes: int) -> None:
    _lazy_init()
    memcpy(ctypes.cast(mem_dst_handle, ctypes.c_void_p), nbytes, ctypes.cast(mem_src_handle, ctypes.c_void_p), nbytes,
           support.MemcpyKind.RT_MEMCPY_DEVICE_TO_HOST)


def launch_kernel(fn_handle: support.Function, block_num: int, args: Iterable[Any], num_args: Optional[int] = None,
                  sm_desc: Optional[Union[int, ctypes.c_uint64]] = None,
                  stream_handle: Optional[support.Stream] = None) -> None:
    _lazy_init()
    num_args = len(args) if num_args is None else num_args
    args_arr = (ctypes.c_uint64 * num_args)(*(arg.value if isinstance(arg, ctypes.c_void_p) else arg for arg in args))
    if stream_handle is None:
        stream_handle = current_stream()
    state.lib.call(
        "KernelLaunchWrapper",
        fn_handle,
        ctypes.c_uint32(block_num),
        ctypes.c_void_p(ctypes.addressof(args_arr)),
        ctypes.c_uint32(num_args * 8),
        ctypes.c_void_p(sm_desc),
        stream_handle,
    )


def synchronize(stream_handle: Optional[support.Stream] = None, timeout: int = 0) -> None:
    _lazy_init()
    if stream_handle is None:
        stream_handle = current_stream()
    if timeout > 0:
        state.lib.call(
            "StreamSynchronizeWithTimeoutWrapper",
            stream_handle,
            ctypes.c_int32(timeout),
        )
    else:
        state.lib.call(
            "StreamSynchronizeWrapper",
            stream_handle,
        )


def free(mem_handle: support.Memory):
    _lazy_init()
    mem_ptr = ctypes.c_void_p(state.allocs[mem_handle.value])
    state.lib.call(
        "FreeWrapper",
        mem_ptr,
    )
    del state.allocs[mem_handle.value]


def synchronize_device() -> None:
    _lazy_init()
    state.lib.call("DeviceSynchronizeWrapper")


def c2c_ctrl_addr() -> int:
    if is_model():
        return 255086295400448  # magic value for camodel
    if get_soc_version() in [config.Platform.Ascend910_9579, config.Platform.Ascend910_9599]:
        return 0
    _lazy_init(need_stream=False)
    addr = (ctypes.c_uint64 * 8)()
    state.lib.call(
        "GetC2cCtrlAddrWrapper",
        addr,
        (ctypes.c_uint32 * 8)(),
    )
    return int(addr[0])


def set_pro_switch(command_type: support.ProfilingCommandHandle.Type, device_id: Optional[int] = None,
                   conf: int = 0b11111):
    device_ids = (ctypes.c_uint32 * support.ProfilingValues.RT_PROF_MAX_DEV_NUM)(_select_device_id(device_id))
    prof_config = ctypes.c_uint64(conf)
    handle = support.ProfilingCommandHandle(profSwitch=prof_config, profSwitchHi=ctypes.c_uint64(0),
                                            devNums=ctypes.c_uint32(1), devIdList=device_ids,
                                            modelId=ctypes.c_uint32(3), type=ctypes.c_uint32(command_type.value),
                                            cacheFlag=ctypes.c_uint32(0))
    state.lib.call(
        "ProfSetProSwitchWrapper",
        ctypes.c_void_p(ctypes.addressof(handle)),
        ctypes.sizeof(handle),
    )


def acl_init():
    return state.npu_utils.acl_init()


def acl_finalize():
    return state.npu_utils.acl_finalize()


def npu_utils():
    return state.npu_utils


def current_tick() -> Optional[int]:
    if not is_model():
        return None
    _lazy_init(need_stream=False)
    fn = getattr(state.lib.lib, "_ZN9tm_engine7tm_timeEv")
    fn.argtypes = ()
    fn.restype = ctypes.c_uint64
    return int(fn())
