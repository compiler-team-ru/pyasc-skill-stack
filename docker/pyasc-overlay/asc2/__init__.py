# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.language.tile.tensor import tensor
from asc.language.tile.tile import TileLocation
from asc.language.tile.range import range

# Tile operations
from asc.language.tile.atomic_ops import (
    atomic_add,
    atomic_max,
    atomic_min,
)
from asc.language.tile.binary_ops import (
    add,
    div,
    equal,
    greater,
    greater_equal,
    left_shift,
    less,
    less_equal,
    matmul,
    maximum,
    minimum,
    mul,
    not_equal,
    right_shift,
    sub,
)
from asc.language.tile.creation_ops import (
    full,
    full_like,
    zeros,
    zeros_like,
)
from asc.language.tile.memory_ops import (
    load,
    store,
)
from asc.language.tile.prog_model_ops import (
    block_idx,
    block_num,
    num_tiles,
)
from asc.language.tile.shape_ops import (
    broadcast_to,
    expand_dims,
    reshape,
    squeeze,
)
from asc.language.tile.unary_ops import (
    abs,
    ceil,
    cos,
    cosh,
    erf,
    exp,
    exp2,
    floor,
    log,
    log2,
    negative,
    relu,
    rms_norm,
    rsqrt,
    sin,
    sinh,
    softmax,
    sqrt,
    tan,
    tanh,
)
from asc.language.tile.indexing_ops import (
    mask,
    where,
)
from asc.language.tile.reduction_ops import (
    reduce_max,
    reduce_min,
    reduce_sum,
    reduce_prod,
)

from .jit import jit

__all__ = [
    # tensor
    "tensor",
    # tile
    "TileLocation",
    # range
    "range",
    # atomic_ops
    "atomic_add",
    "atomic_max",
    "atomic_min",
    # binary_ops
    "add",
    "div",
    "equal",
    "greater",
    "greater_equal",
    "left_shift",
    "less",
    "less_equal",
    "matmul",
    "maximum",
    "minimum",
    "mul",
    "not_equal",
    "right_shift",
    "sub",
    # creation_ops
    "full",
    "full_like",
    "zeros",
    "zeros_like",
    # memory_ops
    "load",
    "store",
    # prog_model_ops
    "block_idx",
    "block_num",
    "num_tiles",
    # shape_ops
    "broadcast_to",
    "expand_dims",
    "reshape",
    "squeeze",
    # unary_ops
    "abs",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "exp",
    "exp2",
    "floor",
    "log",
    "log2",
    "negative",
    "relu",
    "rms_norm",
    "rsqrt",
    "sin",
    "sinh",
    "softmax",
    "sqrt",
    "tan",
    "tanh",
    # indexing_ops
    "mask",
    "where",
    # reduction_ops
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    # .jit
    "jit",
]
