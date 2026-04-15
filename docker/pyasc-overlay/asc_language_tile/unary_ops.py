# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, TypeVar, Union, overload

from ..core.dtype import KnownTypes as KT
from ..core.ir_value import PlainValue, RuntimeFloat, RuntimeNumeric, IRHandle, materialize_ir_value as _mat
from ..core.utils import check_type, global_builder
from .tile import Tile, bind_tile_method

T = TypeVar("T")


def op_unary_impl(input: Union[Tile, RuntimeNumeric], build: Callable[..., IRHandle],
                  support_scalar: bool = False) -> Union[Tile, PlainValue]:
    constraint = Union[Tile, RuntimeNumeric] if support_scalar else Tile
    check_type("input", input, constraint)
    is_scalar = not isinstance(input, Tile)
    if is_scalar:
        input = _mat(input, KT.float32)
    result_dtype = input.dtype
    if result_dtype.is_float():
        handle = build(input.to_ir())
    else:
        raise RuntimeError(f"Operation not support this dtype: {result_dtype}")
    if is_scalar:
        return PlainValue(handle)
    return Tile(handle)


def set_docstring(name: str, support_scalar: bool = False) -> Callable[[T], T]:

    def decorator(fn: T) -> T:
        doc = """
    Computes the element-wise {name} of :code:`input`.

    Args:
        input: the input value ({tile_info})
    """
        fn.__doc__ = doc.format(name=name, tile_info="tile or scalar" if support_scalar else "tile")
        return fn

    return decorator


@bind_tile_method
@set_docstring("cosine")
def cos(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CosOp)


@bind_tile_method
@set_docstring("sine")
def sin(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinOp)


@bind_tile_method
@set_docstring("tangent")
def tan(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanOp)


@bind_tile_method
@set_docstring("hyperbolic sine")
def sinh(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SinhOp)


@bind_tile_method
@set_docstring("hyperbolic cosine")
def cosh(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CoshOp)


@bind_tile_method
@set_docstring("hyperbolic tangent")
def tanh(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_TanhOp)


@overload
def erf(input: Tile) -> Tile:
    ...


@overload
def erf(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tile_method
@set_docstring("exponential", support_scalar=True)
def exp(input: Union[Tile, RuntimeFloat]) -> Union[Tile, PlainValue]:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_ExpOp, support_scalar=True)


@bind_tile_method
@set_docstring("natural logarithm")
def log(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_LogOp)


@bind_tile_method
@set_docstring("logarithm (base 2)")
def log2(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_Log2Op)


@bind_tile_method
@set_docstring("floor rounding")
def floor(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_FloorOp)


@bind_tile_method
@set_docstring("ceil rounding")
def ceil(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_CeilOp)


@bind_tile_method
@set_docstring("absolute value")
def abs(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_AbsFOp)


@overload
def erf(input: Tile) -> Tile:
    ...


@overload
def erf(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tile_method
@set_docstring("error function", support_scalar=True)
def erf(input: Union[Tile, RuntimeFloat]) -> Union[Tile, PlainValue]:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_ErfOp, support_scalar=True)


@bind_tile_method
@set_docstring("exponential (base 2)")
def exp2(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_Exp2Op)


@bind_tile_method
@set_docstring("inverse square root")
def rsqrt(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_RsqrtOp)


@overload
def sqrt(input: Tile) -> Tile:
    ...


@overload
def sqrt(input: RuntimeFloat) -> PlainValue:
    ...


@bind_tile_method
@set_docstring("square root", support_scalar=True)
def sqrt(input: Union[Tile, RuntimeFloat]) -> Union[Tile, PlainValue]:
    builder = global_builder.get_ir_builder()
    return op_unary_impl(input, builder.create_math_SqrtOp, support_scalar=True)


@bind_tile_method
@set_docstring("ReLU value")
def relu(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    handle = builder.create_asctile_ReluOp(input.to_ir().get_type(), input.to_ir())
    return Tile(handle)


@bind_tile_method(name="__neg__")
@set_docstring("negation")
def negative(input: Tile) -> Tile:
    builder = global_builder.get_ir_builder()
    result_dtype = input.dtype
    if result_dtype.is_int():
        handle = input * (-1)
    else:
        handle = builder.create_arith_NegFOp(input.to_ir())
    return Tile(handle)


@bind_tile_method
@set_docstring("softmax")
def softmax(input: Tile) -> Tile:
    check_type("input", input, Tile)
    if input.dtype not in [KT.float32, KT.half]:
        raise RuntimeError("Only float and half types are supported.")
    if len(input.shape) > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported")
    handle = global_builder.get_ir_builder().create_asctile_SoftmaxOp(input.to_ir().get_type(), input.to_ir())
    return Tile(handle)


@set_docstring("RmsNorm function")
def rms_norm(input: Tile, gamma: Tile, epsilon: RuntimeFloat) -> Tile:
    check_type("input", input, Tile)
    check_type("gamma", gamma, Tile)
    check_type("epsilon", epsilon, RuntimeFloat)
    if input.dtype not in [KT.float32, KT.half]:
        raise RuntimeError("Only float and half types are supported.")
    if len(input.shape) > 2:
        raise RuntimeError("Tensor dimensionality greater than two is not supported.")
    handle = global_builder.get_ir_builder().create_asctile_RmsnormOp(input.to_ir().get_type(), input.to_ir(),
                                                                      gamma.to_ir(),
                                                                      _mat(epsilon, input.dtype).to_ir())
    return Tile(handle)
