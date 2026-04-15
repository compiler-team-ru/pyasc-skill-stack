# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real
from typing import Iterable, Optional

from ..._C import ir
from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import RuntimeNumeric
from ..core.utils import check_type
from .tile import Tile
from .utils import constant_tile, splat_tile


def full(shape: Iterable[int], value: RuntimeNumeric, dtype: Optional[DataType] = None,
         location: Optional[ir.TileLocation] = ir.TileLocation.UB) -> Tile:
    if not all(isinstance(dim, int) for dim in shape):
        raise RuntimeError("shape must be integers")
    shape = tuple(shape)
    if isinstance(value, Real):
        if dtype is None:
            dtype = KT.int32 if isinstance(value, int) else KT.float32
        return constant_tile(value, shape, dtype, location)
    if dtype is None:
        dtype = value.dtype
    return splat_tile(value, shape, dtype, location)


def full_like(input: Tile, value: RuntimeNumeric, location: Optional[ir.TileLocation] = ir.TileLocation.UB) -> Tile:
    check_type("input", input, Tile)
    return full(input.shape, value, input.dtype, location)


def zeros(shape: Iterable[int], dtype: DataType = KT.int32,
          location: Optional[ir.TileLocation] = ir.TileLocation.UB) -> Tile:
    return full(shape, 0, dtype, location)


def zeros_like(input: Tile, location: Optional[ir.TileLocation] = ir.TileLocation.UB) -> Tile:
    check_type("input", input, Tile)
    return zeros(input.shape, input.dtype, location)
