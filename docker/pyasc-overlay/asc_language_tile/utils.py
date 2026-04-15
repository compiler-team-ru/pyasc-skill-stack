# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from numbers import Real
from typing import Iterable, Tuple, Union

from ..._C import ir
from ..core.dtype import DataType, KnownTypes as KT
from ..core.ir_value import PlainValue, RuntimeNumeric
from ..core.tensor import TensorShape
from ..core.utils import check_type, global_builder
from .tile import BinaryOperandTypeError, Tile


def constant_tile(value: Real, shape: TensorShape, dtype: DataType, loc: ir.TileLocation = ir.TileLocation.UB) -> Tile:
    check_type("value", value, Real)
    builder = global_builder.get_ir_builder()
    attr_builders = {
        "int8": builder.get_i8_attr,
        "int16": builder.get_i16_attr,
        "int32": builder.get_i32_attr,
        "int64": builder.get_i64_attr,
        "float16": builder.get_f16_attr,
        "float32": builder.get_f32_attr,
        "float64": builder.get_f64_attr,
    }
    attr_builder = attr_builders.get(str(dtype))
    if attr_builder is None:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ir_type = ir.get_asctile_TileType(shape, dtype.to_ir(), loc)
    splat_attr = ir.get_splat_attr(ir_type, attr_builder(value))
    handle = builder.create_arith_ConstantOp(splat_attr)
    return Tile.from_ir(handle)


def splat_tile(value: PlainValue, shape: TensorShape, dtype: DataType,
               loc: ir.TileLocation = ir.TileLocation.UB) -> Tile:
    ir_type = ir.get_asctile_TileType(shape, dtype.to_ir(), loc)
    handle = global_builder.get_ir_builder().create_asctile_SplatOp(ir_type, value.cast(dtype).to_ir())
    return Tile.from_ir(handle)


def create_tile(value: Union[Tile, RuntimeNumeric], dtype: DataType, shape: Tuple[int, ...]) -> Tile:
    if isinstance(value, Tile):
        if value.shape != shape:
            raise RuntimeError("Tile shape doesn't match the required shape")
        return value.to(dtype)
    if isinstance(value, Real):
        return constant_tile(value, shape, dtype)
    if isinstance(value, PlainValue):
        return splat_tile(value, shape, dtype)
    raise BinaryOperandTypeError(f"Tile cannot be created from {value.__class__.__name__}")


def infer_tile_dtype(value: Union[Tile, PlainValue, Real]) -> DataType:
    if isinstance(value, (Tile, PlainValue)):
        return value.dtype
    if isinstance(value, bool):
        return KT.int1
    if isinstance(value, int):
        return KT.int32
    if isinstance(value, float):
        return KT.float32
    raise BinaryOperandTypeError(f"Unable to obtain dtype of {value.__class__.__name__}")


def infer_common_dtype(lhs: Union[Tile, RuntimeNumeric], rhs: Union[Tile, RuntimeNumeric]) -> DataType:
    lhs_dtype = infer_tile_dtype(lhs)
    rhs_dtype = infer_tile_dtype(rhs)
    if lhs_dtype == rhs_dtype:
        return lhs_dtype
    if not lhs_dtype.is_numeric() or not rhs_dtype.is_numeric():
        raise RuntimeError(f"Operand dtypes must be numeric, got {lhs_dtype} and {rhs_dtype}")
    if lhs_dtype.is_unsigned() or rhs_dtype.is_unsigned():
        raise NotImplementedError(f"Unsigned dtype operands not supported, got {lhs_dtype} and {rhs_dtype}")
    lhs_is_tile = isinstance(lhs, Tile)
    rhs_is_tile = isinstance(rhs, Tile)
    if lhs_is_tile and not rhs_is_tile:
        return lhs.dtype
    if rhs_is_tile and not lhs_is_tile:
        return rhs.dtype
    if lhs_dtype.is_signed() and rhs_dtype.is_signed() and lhs_dtype.bitwidth != rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.bitwidth > rhs_dtype.bitwidth else rhs_dtype
    if lhs_dtype.is_float() and rhs_dtype.is_float() and lhs_dtype.bitwidth != rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.bitwidth > rhs_dtype.bitwidth else rhs_dtype
    if lhs_dtype.bitwidth == rhs_dtype.bitwidth:
        return lhs_dtype if lhs_dtype.is_float() else rhs_dtype
    raise RuntimeError(f"Unable to infer common dtype between {lhs_dtype} and {rhs_dtype}")


def infer_common_shape(lhs: Union[Tile, RuntimeNumeric], rhs: Union[Tile, RuntimeNumeric]) -> Tuple[int, ...]:
    lhs_is_tile = isinstance(lhs, Tile)
    rhs_is_tile = isinstance(rhs, Tile)
    if lhs_is_tile and rhs_is_tile:
        if lhs.shape == rhs.shape:
            return lhs.shape
        raise RuntimeError(f"Shape mismatch: {lhs.shape} vs. {rhs.shape}")
    return lhs.shape if lhs_is_tile else rhs.shape


def verify_shape(shape: Iterable[int], name: str = "shape") -> Tuple[int]:
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    if len(shape) < 1:
        raise RuntimeError(f"'{name}' must have at least one value")
    if not all(isinstance(dim, int) for dim in shape):
        raise RuntimeError(f"All values in '{name}' must be integers")
    if any(dim <= 0 for dim in shape):
        raise RuntimeError(f"All values in '{name}' must be positive")
    return shape


def check_data_alignment(shape: Tuple[int, ...], dtype: DataType) -> None:
    try:
        dtype_size = dtype.sizeof()
    except ValueError:  # sizeof might be not supported
        return
    if shape[-1] % (ir.ub_block_size // dtype_size) != 0:
        raise RuntimeError(f"Last dimension of tile must be aligned by {ir.ub_block_size} bytes, "
                           f"got {shape[-1]} x {dtype_size} bytes")
