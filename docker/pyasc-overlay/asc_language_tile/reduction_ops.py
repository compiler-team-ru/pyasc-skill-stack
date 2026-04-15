# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, List, Optional, Tuple, Union, overload

from ..._C import ir
from ..core.ir_value import PlainValue, IRHandle
from ..core.utils import global_builder
from .tile import Tile, bind_tile_method


def get_reduction_shape(tensor_shape: Tuple[int], keep_dims: bool, dims: Tuple[int]) -> List[int]:
    """
    Get tensor shape after reducing dimensions *dims*
    """
    reduce_dims = [False] * len(tensor_shape)
    for dim in dims:
        reduce_dims[dim] = True
    result = []
    for i in range(0, len(reduce_dims)):
        if not reduce_dims[i]:
            result.append(tensor_shape[i])
        elif keep_dims:
            result.append(1)
    return result


def op_reduce_impl(build_1d: Optional[Callable[..., IRHandle]], build: Callable[..., IRHandle], input: Tile,
                   keep_dims: bool, dims: Tuple[int]) -> Union[Tile, PlainValue]:
    if len(dims) == 0:
        if build_1d is None:
            raise RuntimeError("Reduction to scalar not supported")
        handle = build_1d(input.dtype.to_ir(), input.to_ir())
        return PlainValue(handle)
    if not all(isinstance(dim, int) for dim in dims):
        raise RuntimeError("All reduction dimensions must be integers")
    dims_ir = global_builder.get_ir_builder().get_i32_array_attr(dims)
    target_shape = get_reduction_shape(input.shape, keep_dims, dims)
    dtype = ir.get_element_type(input.to_ir().get_type())
    ir_type = ir.clone_shaped_type(input.to_ir().get_type(), dtype, target_shape)
    handle = build(ir_type, input.to_ir(), dims_ir)
    return Tile(handle)


@overload
def reduce_sum(input: Tile, *dims: int, keep_dims: bool = False) -> Tile:
    ...


@overload
def reduce_sum(input: Tile) -> PlainValue:
    ...


@bind_tile_method(name="sum")
def reduce_sum(input: Tile, *dims: int, keep_dims: bool = False) -> Union[Tile, PlainValue]:
    """
    Returns the sum of each row of the :code:`input` tile in the given dimensions :code:`dims`.

    Dimensions :code:`dims` are squeezed, resulting the output tile having fewer dimensions than input,
    unless :code:`keep_dims=True` is provided.
    When dimension is not specified, the entire tile is reduced to a single scalar value.

    Args:
        input: the input tile
        dims: optional, dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: if set to True, then reduced dimensions are kept in the result shape with size of 1

    Examples:
        Reduce tile by first (outermost) dimension, resulting tile having the shape [256],
        each element is sum of 128 elements in corresponding column: ::

            input = asc2.load(x, [128, 256], offsets=[0, 0])
            result = asc2.reduce_sum(input, 0)

        Compute total sum of all numbers in tile, returns single scalar value: ::

            input = asc2.load(x, [256, 256], offsets=[0, 0])
            result = asc2.reduce_sum(input)
    """
    builder = global_builder.get_ir_builder()
    return op_reduce_impl(builder.create_asctile_ReduceSumAs1dOp, builder.create_asctile_ReduceSumOp, input, keep_dims,
                          dims)


@overload
def reduce_max(input: Tile, *dims: int, keep_dims: bool = False) -> Tile:
    ...


@overload
def reduce_max(input: Tile) -> PlainValue:
    ...


@bind_tile_method(name="max")
def reduce_max(input: Tile, *dims: int, keep_dims: bool = False) -> Union[Tile, PlainValue]:
    """
    Returns the maximum value of each row of the :code:`input` tile in the given dimensions :code:`dims`.

    Dimensions :code:`dims` are squeezed, resulting the output tile having fewer dimensions than input,
    unless :code:`keep_dims=True` is provided.
    When dimension is not specified, the entire tile is reduced to a single scalar value.

    Args:
        input: the input tile
        dims: optional, dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: if set to True, then reduced dimensions are kept in the result shape with size of 1

    Examples:
        Reduce tile by first (outermost) dimension, resulting tile having the shape [256],
        each element is a maximum value between 128 elements in corresponding column: ::

            input = asc2.load(x, [128, 256], offsets=[0, 0])
            result = asc2.reduce_max(input, 0)

        Compute the maximum value between all tile elements, returns single scalar value: ::

            input = asc2.load(x, [256, 256], offsets=[0, 0])
            result = asc2.reduce_max(input)
    """
    builder = global_builder.get_ir_builder()
    return op_reduce_impl(builder.create_asctile_ReduceMaxAs1dOp, builder.create_asctile_ReduceMaxOp, input, keep_dims,
                          dims)


@overload
def reduce_min(input: Tile, *dims: int, keep_dims: bool = False) -> Tile:
    ...


@overload
def reduce_min(input: Tile) -> PlainValue:
    ...


@bind_tile_method(name="min")
def reduce_min(input: Tile, *dims: int, keep_dims: bool = False) -> Union[Tile, PlainValue]:
    """
    Returns the minimum value of each row of the :code:`input` tile in the given dimensions :code:`dims`.

    Dimensions :code:`dims` are squeezed, resulting the output tile having fewer dimensions than input,
    unless :code:`keep_dims=True` is provided.
    When dimension is not specified, the entire tile is reduced to a single scalar value.

    Args:
        input: the input tile
        dims: optional, dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: if set to True, then reduced dimensions are kept in the result shape with size of 1

    Examples:
        Reduce tile by first (outermost) dimension, resulting tile having the shape [256],
        each element is a minimum value between 128 elements in corresponding column: ::

            input = asc2.load(x, [128, 256], offsets=[0, 0])
            result = asc2.reduce_min(input, 0)

        Compute the minimum value between all tile elements, returns single scalar value: ::

            input = asc2.load(x, [256, 256], offsets=[0, 0])
            result = asc2.reduce_min(input)
    """
    builder = global_builder.get_ir_builder()
    return op_reduce_impl(builder.create_asctile_ReduceMinAs1dOp, builder.create_asctile_ReduceMinOp, input, keep_dims,
                          dims)


@bind_tile_method(name="prod")
def reduce_prod(input: Tile, *dims: int, keep_dims: bool = False) -> Tile:
    """
    Returns the product of each row of the :code:`input` tile in the given dimensions :code:`dims`.

    Dimensions :code:`dims` are squeezed, resulting the output tile having fewer dimensions than input,
    unless :code:`keep_dims=True` is provided.

    Args:
        input: the input tile
        dims: dimensions to reduce, should be in range of [0..len(input.shape)-1]
        keep_dims: if set to True, then reduced dimensions are kept in the result shape with size of 1

    Examples:
        Reduce tile by first (outermost) dimension, resulting tile having the shape [256],
        each element is product of 128 elements in corresponding column: ::

            input = asc2.load(x, [128, 256], offsets=[0, 0])
            result = asc2.reduce_prod(0)
    """
    builder = global_builder.get_ir_builder()
    return op_reduce_impl(None, builder.create_asctile_ReduceProdOp, input, keep_dims, dims)
