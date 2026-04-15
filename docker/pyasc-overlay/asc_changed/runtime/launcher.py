# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import ctypes
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from . import utils
from .._C import ir
from ..language.core.struct import Struct
from ..lib import runtime as rt
from .compiler import CompiledKernel
from .config import Platform
from .memory_handle import MemoryHandle, resolve_memory_handle


class MsprofLauncher(object):

    def __init__(self, is_model_: bool):
        self.is_model = is_model_
        self.utils = rt.npu_utils()
        self.start_time = 0

    def start(self):
        if self.is_model:
            return

        self.start_time = self.utils.msprof_sys_cycle_time()

    def process(self, kernel_name: str, block_num: int, task_type: int):
        if self.is_model:
            return

        time_stamp = self.utils.msprof_sys_cycle_time()
        self.utils.msprof_report_compact_info(time_stamp, kernel_name, block_num, task_type)

        end_time = self.utils.msprof_sys_cycle_time()
        self.utils.msprof_report_api(self.start_time, end_time, kernel_name)


ub_capacity = {
    Platform.Ascend910B1: 192 * 1024,
    Platform.Ascend910B2: 192 * 1024,
    Platform.Ascend910B2C: 192 * 1024,
    Platform.Ascend910B3: 192 * 1024,
    Platform.Ascend910B4: 192 * 1024,
    Platform.Ascend910B4_1: 192 * 1024,
    Platform.Ascend910_9362: 192 * 1024,
    Platform.Ascend910_9372: 192 * 1024,
    Platform.Ascend910_9381: 192 * 1024,
    Platform.Ascend910_9382: 192 * 1024,
    Platform.Ascend910_9391: 192 * 1024,
    Platform.Ascend910_9392: 192 * 1024,
    Platform.Ascend910_9579: 248 * 1024,
    Platform.Ascend910_9589: 248 * 1024,
    Platform.Ascend910_9599: 248 * 1024,
    Platform.Ascend950PR_9579: 248 * 1024,
    Platform.Ascend950PR_9589: 248 * 1024,
    Platform.Ascend950PR_9599: 248 * 1024,
}


@dataclass(frozen=True)
class LaunchOptions:
    """
    Kernel launch and device runtime options.

    These options can also be used as positional arguments in :code:`[` brackets :code:`]` when launch JIT function:

    .. code-block:: python

        @asc2.jit
        def kernel(x_ptr, y_ptr):
            ...

        def launch(x, y, core_num = 16):
            kernel[core_num](x, y)
    """

    core_num: int = 0
    """
    Number of active execution blocks (AI cores) used to launch the kernel.
    By default, all cores available on the current platform will be used.
    """

    stream: Optional[rt.Stream] = None


class Launcher:

    def __init__(self, options: LaunchOptions):
        self.options = options
        self.msprof = MsprofLauncher(rt.is_model())

    @staticmethod
    def get_core_num(device_id: Optional[int] = None) -> int:
        return rt.device_info(rt.DeviceModuleType.RT_MODULE_TYPE_AICORE, rt.DeviceInfoType.INFO_TYPE_CORE_NUM,
                              device_id)

    @staticmethod
    def get_ub_capacity() -> int:
        return ub_capacity[Platform(rt.current_platform())]

    @staticmethod
    def expand_kernel_args(args: Iterable[Any]) -> List[Union[np.generic, MemoryHandle]]:
        kernel_args = []
        for arg in args:
            if isinstance(arg, int):
                kernel_args.append(np.int32(arg))
            elif isinstance(arg, float):
                kernel_args.append(np.float32(arg))
            elif isinstance(arg, bool):
                kernel_args.append(np.int8(int(arg)))
            elif isinstance(arg, np.generic):
                kernel_args.append(arg)
            elif isinstance(arg, Struct):
                kernel_args.append(resolve_memory_handle(arg.pack()))
            else:
                kernel_args.append(resolve_memory_handle(arg))
        return kernel_args

    def launch_kernel(self, function: rt.Function, kernel_args: List[Union[np.generic, MemoryHandle]],
                      enable_debug: bool, func_name: str, core_type: rt.CoreType) -> None:

        def blobs_size(inputs: List[bytes]) -> int:
            return sum(len(x) for x in input_blobs)

        input_blobs: List[bytes] = []
        memory_args: List[MemoryHandle] = []
        for arg in kernel_args:
            if isinstance(arg, np.generic):
                input_blobs.append(arg.tobytes())
                if arg.itemsize < 4:
                    input_blobs.append(b"\0" * (4 - arg.itemsize))
                elif arg.itemsize > 4 and arg.itemsize < 8:
                    input_blobs.append(b"\0" * (8 - arg.itemsize))
            elif isinstance(arg, MemoryHandle):
                if blobs_size(input_blobs) % 8 != 0:
                    input_blobs.append(b"\0" * 4)
                handle = arg.copy_to_device()
                input_blobs.append(np.uint64(handle).tobytes())
                memory_args.append(arg)
            else:
                raise TypeError(f"Unsupported kernel argument of type {type(arg)}")
        aligned_len = int(np.ceil(blobs_size(input_blobs) / 8)) * 8
        combined_inputs = bytes().join(input_blobs).ljust(aligned_len, b"\0")
        chunks = [combined_inputs[i:i + 8] for i in range(0, len(combined_inputs), 8)]
        inputs = [ctypes.c_uint64(int.from_bytes(x, "little")) for x in chunks]

        stream = self.options.stream or rt.current_stream()

        self.msprof.start()
        rt.launch_kernel(function, self.options.core_num, inputs, stream_handle=stream)
        self.msprof.process(func_name, self.options.core_num, rt.msprof_task_type(core_type))

        rt.synchronize()
        for index, arg in enumerate(memory_args):
            if enable_debug and index == len(memory_args) - 1:
                rt.call_print_interface(inputs[-1], utils.TOTAL_DUMP_SIZE, stream, func_name)
            else:
                arg.copy_from_device()
            arg.release_memory()

    @classmethod
    def check_memory_overflow(cls, memory_consumed: Dict[str, int]) -> None:
        ub_consumed = memory_consumed.get("UB", 0)
        if ub_consumed > cls.get_ub_capacity():
            raise RuntimeError(f"UB overflow: {cls.get_ub_capacity()} is available, {ub_consumed} is used.")
        # TODO: check other memory realms

    def run(self, kernel: CompiledKernel, function_name: str, user_args: Tuple[Any]) -> None:
        if kernel.memory_consumed is not None:
            self.check_memory_overflow(kernel.memory_consumed)
        dry_run = os.environ.get('DRY_RUN')
        if dry_run:
            return
        if not isinstance(kernel.binary, bytes):
            raise RuntimeError("Compiled binary is required to launch the kernel")
        explicit_arg = iter(user_args)
        kernel_args = []
        for kind in kernel.kernel_args:
            if kind == ir.KernelArgument.Explicit:
                kernel_args.append(next(explicit_arg))
            elif kind == ir.KernelArgument.FftsAddr:
                ffts_addr = np.array([rt.c2c_ctrl_addr()], dtype=np.uint64)
                kernel_args.append(ffts_addr)
            else:
                raise ValueError(f"Unexpected KernelArgument value: {kind}")
        if kernel.enable_debug:
            kernel_args.append(np.zeros(utils.TOTAL_DUMP_SIZE, dtype=np.int8))
        kernel_args = self.expand_kernel_args(tuple(kernel_args))
        kernel_handle = rt.register_device_binary_kernel(kernel.binary, rt.magic_elf_value(kernel.core_type))
        function = rt.register_function(kernel_handle, function_name, mode=0)
        if self.options.core_num <= 0:
            raise ValueError("Core number should be large than 0")
        self.launch_kernel(function, kernel_args, kernel.enable_debug, function_name, kernel.core_type)
        rt.free_mem()
