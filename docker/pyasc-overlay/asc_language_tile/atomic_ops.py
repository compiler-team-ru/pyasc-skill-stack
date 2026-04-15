# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.utils import global_builder
from .tensor import Tensor
from .tile import Tile


def op_atomic_impl(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt], kind: ir.AtomicKind):
    offsets = [_mat(v, KT.int32).to_ir() for v in offsets]
    global_builder.get_ir_builder().create_asctile_AtomicRMWOp(tile.to_ir(), tensor.to_ir(), offsets, kind)


def atomic_add(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt]) -> None:
    return op_atomic_impl(tile, tensor, offsets, ir.AtomicKind.Add)


def atomic_max(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt]) -> None:
    return op_atomic_impl(tile, tensor, offsets, ir.AtomicKind.Max)


def atomic_min(tile: Tile, tensor: Tensor, offsets: Iterable[RuntimeInt]) -> None:
    return op_atomic_impl(tile, tensor, offsets, ir.AtomicKind.Min)
