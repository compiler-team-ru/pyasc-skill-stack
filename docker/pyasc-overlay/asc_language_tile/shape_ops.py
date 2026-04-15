# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from typing import Tuple

from ..._C import ir
from ..core.utils import global_builder
from .tile import Tile, bind_tile_method
from .utils import verify_shape


def shapes_match(shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    if len(shape) > len(target_shape):
        return False
    src = shape[::-1]
    dst = target_shape[::-1]
    for i in range(0, len(dst)):
        if i < len(src) and dst[i] != src[i] and src[i] != 1:
            return False
    return True


@bind_tile_method
def broadcast_to(input: Tile, *shape: int) -> Tile:
    """
    Creates new tile of a given shape broadcasting data from the input tensor.
    This function works similar to :code:`torch.broadcast_to`.

    Args:
        input: the input tensor
        shape: the target shape

    Examples:
        Broadcast tile to the provided shape: ::

            input = asc2.load(x, [256], offsets=[0])
            result = input.broadcast_to([16,256])

        The code above may act as the following: ::

            input:   [0,1,2,3,4, ... 255]
            result:  [[0,1,2,...255], [0,1,2,...255] ... [0,1,2,..255]]

    When lowering to Ascend C, it first it fills the BroadcastTiling structure calling GetBroadcastTilingInfo:

    .. code-block:: c++

        template <typename T, int constRank = -1, uint32_t* constDstShape = nullptr, uint32_t* constSrcShape = nullptr>
        __aicore__ inline void GetBroadcastTilingInfo(
        uint32_t rank, const uint32_t* dstShape, const uint32_t* srcShape, bool srcInnerPad, BroadcastTiling& tiling);

    Then it perform Broadcast to fill the destination tensor:

    .. code-block:: c++

        template <typename T, int constRank = -1, uint32_t* constDstShape = nullptr, uint32_t* constSrcShape = nullptr,
        bool constSrcInnerPad = false>
        __aicore__ inline void Broadcast(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t* dstShape,
        const uint32_t* srcShape, BroadcastTiling* tiling);
    """
    shape = verify_shape(shape)
    if not shapes_match(input.shape, shape):
        raise RuntimeError(f"Cannot broadcast tile with shape {input.shape} to {shape}")
    result_type = ir.clone_shaped_type(input.to_ir().get_type(), input.dtype.to_ir(), shape)
    handle = global_builder.get_ir_builder().create_asctile_BroadcastOp(result_type, input.to_ir())
    return Tile(handle)


@bind_tile_method
def reshape(input: Tile, *shape: int) -> Tile:
    shape = verify_shape(shape)
    if math.prod(input.shape) != math.prod(shape):
        raise RuntimeError("Result tile must have the same number of elements as input tile")
    builder = global_builder.get_ir_builder()
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), input.dtype.to_ir(), shape)
    handle = builder.create_asctile_ReshapeOp(ir_type, input.to_ir())
    return Tile(handle)


@bind_tile_method
def expand_dims(input: Tile, *axis: int) -> Tile:
    shape = list(input.shape)
    axis = sorted(set(axis))
    for ax in axis:
        shape.insert(ax, 1)
    return reshape(input, *shape)


@bind_tile_method
def squeeze(input: Tile, *axis: int) -> Tile:
    shape = []
    axis = set(axis if axis else (i for i, dim in enumerate(input.shape) if dim == 1))
    for i, dim in enumerate(input.shape):
        if i not in axis:
            shape.append(dim)
            continue
        if dim != 1:
            raise RuntimeError(f"Unable to squeeze the axis {i} since its length must be 1, got {dim}")
    return reshape(input, *shape)
