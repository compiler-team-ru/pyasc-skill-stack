# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import Enum
from typing import Optional, Union

from asc.lib import runtime as rt


class Backend(Enum):
    Model = "Model"
    NPU = "NPU"


class Platform(Enum):
    """get soc version"""
    Ascend910B1 = "Ascend910B1"
    Ascend910B2 = "Ascend910B2"
    Ascend910B2C = "Ascend910B2C"
    Ascend910B3 = "Ascend910B3"
    Ascend910B4 = "Ascend910B4"
    Ascend910B4_1 = "Ascend910B4-1"
    Ascend910_9362 = "Ascend910_9362"
    Ascend910_9372 = "Ascend910_9372"
    Ascend910_9381 = "Ascend910_9381"
    Ascend910_9382 = "Ascend910_9382"
    Ascend910_9391 = "Ascend910_9391"
    Ascend910_9392 = "Ascend910_9392"
    Ascend910_9579 = "Ascend910_9579"
    Ascend910_9589 = "Ascend910_9589"
    Ascend910_9599 = "Ascend910_9599"
    Ascend950PR_9579 = "Ascend950PR_9579"
    Ascend950PR_9589 = "Ascend950PR_9589"
    Ascend950PR_9599 = "Ascend950PR_9599"


class KernelType(Enum):
    """get kernel type"""
    AIV_ONLY = 0
    AIC_ONLY = 1
    MIX_AIV_HARD_SYNC = 2
    MIX_AIC_HARD_SYNC = 3
    MIX_AIV_1_0 = 4
    MIX_AIC_1_0 = 5
    MIX_AIC_1_1 = 6
    MIX_AIC_1_2 = 7


def set_platform(
    backend: Union[Backend, str],
    soc_version: Optional[Platform] = None,
    device_id: Optional[int] = None,
    check=True,
) -> None:
    backend = Backend(backend)
    if backend == Backend.Model:
        if soc_version is None:
            soc_version = Platform.Ascend910B1
        rt.use_model()
    elif backend == Backend.NPU:
        soc_ver = Platform(rt.current_platform())
        if soc_version is not None and soc_version != soc_ver:
            raise ValueError(f"Input soc version: {soc_version} is different from actual: {soc_ver}")
        soc_version = soc_ver
        rt.use_npu()
    else:
        raise ValueError(f"Unknown execution backend: {backend}")
    rt.set_soc_version(soc_version)
    if device_id is not None:
        rt.set_device(device_id)
    if check and not rt.is_available():
        raise RuntimeError("Runtime library must be available!")
