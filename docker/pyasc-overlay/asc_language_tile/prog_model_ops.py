# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable

from ..basic.sys_var import get_block_idx, get_block_num
from ..core.ir_value import PlainValue, RuntimeInt
from .tensor import Tensor
from .utils import verify_shape


def block_idx() -> PlainValue:
    return get_block_idx()


def block_num() -> PlainValue:
    return get_block_num()


def num_tiles(tensor: Tensor, axis: RuntimeInt, shape: Iterable[int]) -> RuntimeInt:
    shape = verify_shape(shape)
    tensor_shape = tensor.shape
    if len(tensor_shape) != len(shape):
        raise RuntimeError("rank of 'tensor_shape' must match rank of 'shape'")
    if axis >= len(shape) or axis >= len(tensor_shape):
        raise ValueError(f"axis ({axis}) exceeds number of dimensions")
    dim_size = tensor_shape[axis]
    tile_size = shape[axis]
    return (dim_size + tile_size - 1) // tile_size
