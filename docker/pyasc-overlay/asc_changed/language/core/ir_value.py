# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import abc
from typing import Any, NoReturn, Optional, Union
from typing_extensions import Self, TypeAlias

from ..._C import ir
from .constexpr import ConstExpr
from .dtype import DataType, KnownTypes as KT
from .utils import require_jit, global_builder

IRHandle: TypeAlias = ir.Value


class IRValue(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def to_ir(self) -> IRHandle:
        raise NotImplementedError


class GlobalAddress(IRValue):

    def __init__(self, handle: IRHandle, dtype: Optional[DataType] = None):
        """This contructor should not be called by user"""
        self.handle = handle
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"GlobalAddress(dtype={self.dtype}, handle=...)"

    @require_jit
    def __add__(self, offset: "RuntimeInt") -> GlobalAddress:
        offset = materialize_ir_value(offset, KT.int_)
        builder = global_builder.get_ir_builder()
        offset_index = builder.create_arith_IndexCastOp(offset.to_ir(), builder.get_index_type())
        handle = builder.create_emitasc_PtrOffsetOp(self.to_ir(), offset_index)
        return GlobalAddress(handle, self.dtype)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return GlobalAddress(handle, DataType.from_ir(ir.get_element_type(handle.get_type())))

    def to_ir(self) -> IRHandle:
        return self.handle


class PlainValue(IRValue):

    def __init__(self, handle: IRHandle, dtype: Optional[DataType] = None):
        """This contructor should not be called by user"""
        self.handle = handle
        self.dtype = dtype or DataType.from_ir(handle.get_type())

    @require_jit
    def __rxor__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "XOrI", None)

    # Binary operations

    @require_jit
    def __add__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "AddI", "AddF")

    @require_jit
    def __sub__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "SubI", "SubF")

    @require_jit
    def __mul__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "MulI", "MulF")

    @require_jit
    def __truediv__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "DivSI", "DivF")

    @require_jit
    def __floordiv__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "DivSI", "DivF")

    @require_jit
    def __mod__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "RemSI", None)

    @require_jit
    def __pow__(self, other) -> NoReturn:
        raise NotImplementedError("Power operator is not implemented for PlainValue")

    @require_jit
    def __lshift__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "ShLI", None)

    @require_jit
    def __rshift__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "ShRSI", None)

    @require_jit
    def __and__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "AndI", None)

    @require_jit
    def __or__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "OrI", None)

    @require_jit
    def __xor__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "XOrI", None)

    def __repr__(self) -> str:
        return f"PlainValue(dtype={self.dtype}, handle=...)"

    # Binary operations (reversed)

    @require_jit
    def __radd__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "AddI", "AddF")

    @require_jit
    def __rsub__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "SubI", "SubF")

    @require_jit
    def __rmul__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "MulI", "MulF")

    @require_jit
    def __rtruediv__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "DivSI", "DivF")

    @require_jit
    def __rfloordiv__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "DivSI", "DivF")

    @require_jit
    def __rmod__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "RemSI", None)

    @require_jit
    def __rpow__(self, other) -> NoReturn:
        raise NotImplementedError("Power operator is not implemented for PlainValue")

    @require_jit
    def __rlshift__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "ShLI", None)

    @require_jit
    def __rrshift__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "ShRSI", None)

    @require_jit
    def __rand__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "AndI", None)

    @require_jit
    def __ror__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "OrI", None)

    # Comparison operations

    @require_jit
    def __eq__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, ir.CmpIPredicate.eq, ir.CmpFPredicate.OEQ)

    @require_jit
    def __ne__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, ir.CmpIPredicate.ne, ir.CmpFPredicate.ONE)

    @require_jit
    def __ge__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, ir.CmpIPredicate.sge, ir.CmpFPredicate.OGE)

    @require_jit
    def __gt__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, ir.CmpIPredicate.sgt, ir.CmpFPredicate.OGT)

    @require_jit
    def __le__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, ir.CmpIPredicate.sle, ir.CmpFPredicate.OLE)

    @require_jit
    def __lt__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, ir.CmpIPredicate.slt, ir.CmpFPredicate.OLT)

    # Comparison operations (reversed)

    @require_jit
    def __req__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, ir.CmpIPredicate.eq, ir.CmpFPredicate.OEQ)

    @require_jit
    def __rne__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, ir.CmpIPredicate.ne, ir.CmpFPredicate.ONE)

    @require_jit
    def __rge__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, ir.CmpIPredicate.sge, ir.CmpFPredicate.OGE)

    @require_jit
    def __rgt__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, ir.CmpIPredicate.sgt, ir.CmpFPredicate.OGT)

    @require_jit
    def __rle__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, ir.CmpIPredicate.sle, ir.CmpFPredicate.OLE)

    @require_jit
    def __rlt__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, ir.CmpIPredicate.slt, ir.CmpFPredicate.OLT)

    # Unary operations

    @require_jit
    def __neg__(self) -> PlainValue:
        if self.dtype.is_float():
            return global_builder.get_ir_builder().create_arith_NegFOp(self.to_ir())
        return self.__mul__(-1)

    @require_jit
    def __pos__(self) -> PlainValue:
        return self

    @require_jit
    def __not__(self) -> PlainValue:
        return self.__eq__(0)

    @require_jit
    def __invert__(self) -> PlainValue:
        raise NotImplementedError("Inversion operator is not implemented for PlainValue")

    @staticmethod
    def infer_common_type(lhs: Any, rhs: Any) -> DataType:
        result_type = None
        if isinstance(lhs, PlainValue):
            result_type = lhs.dtype
        elif isinstance(rhs, PlainValue):
            result_type = rhs.dtype
        else:
            raise ValueError("Either lhs or rhs must be PlainValue, "
                             f"got {lhs.__class__.__name__} and {rhs.__class__.__name__}")
        return result_type

    @classmethod
    def apply_binary_op(cls, lhs: Any, rhs: Any, build_int: str, build_float: str) -> PlainValue:
        result_type = cls.infer_common_type(lhs, rhs)
        lhs = materialize_ir_value(lhs, result_type)
        rhs = materialize_ir_value(rhs, result_type)
        builder_attr = build_int if result_type.is_int() else build_float
        if builder_attr is None:
            raise ValueError(f"Binary operation is not supported between {lhs} and {rhs}")
        handle = getattr(global_builder.get_ir_builder(), f"create_arith_{builder_attr}Op")(lhs.to_ir(), rhs.to_ir())
        return PlainValue(handle=handle, dtype=result_type)

    @classmethod
    def apply_bool_op(cls, lhs: Any, rhs: Any, builder_attr: str) -> PlainValue:
        lhs = materialize_ir_value(lhs, KT.bit)
        rhs = materialize_ir_value(rhs, KT.bit)
        handle = getattr(global_builder.get_ir_builder(), f"create_arith_{builder_attr}Op")(lhs.to_ir(), rhs.to_ir())
        return PlainValue(handle=handle, dtype=KT.bit)

    @classmethod
    def apply_compare_op(cls, lhs: Any, rhs: Any, pred_int: int, pred_float: int) -> PlainValue:
        common_type = cls.infer_common_type(lhs, rhs)
        lhs = materialize_ir_value(lhs, common_type)
        rhs = materialize_ir_value(rhs, common_type)
        builder = global_builder.get_ir_builder()
        method = builder.create_arith_CmpIOp if common_type.is_int() else builder.create_arith_CmpFOp
        pred = pred_int if common_type.is_int() else pred_float
        handle = method(pred, lhs.to_ir(), rhs.to_ir())
        return PlainValue(handle=handle, dtype=KT.int1)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return PlainValue(handle, DataType.from_ir(handle.get_type()))

    @require_jit
    def cast(self, dtype: DataType) -> PlainValue:
        if self.dtype == dtype:
            return self
        from_i = self.dtype.is_int()
        from_f = self.dtype.is_float()
        to_i = dtype.is_int()
        to_f = dtype.is_float()
        method = None
        builder = global_builder.get_ir_builder()
        if not self.dtype.is_numeric() or not dtype.is_numeric():
            pass
        elif self.dtype.bitwidth == dtype.bitwidth:
            if from_f and to_i:
                method = builder.create_arith_FPToSIOp
            elif from_i and to_f:
                method = builder.create_arith_SIToFPOp
            elif from_i and to_i and self.dtype.is_unsigned() != dtype.is_unsigned():
                method = builder.create_emitc_CastOp
        elif (from_i and to_i) or (from_f and to_f):
            ext = self.dtype.bitwidth < dtype.bitwidth
            if from_i:
                if self.dtype.is_unsigned() != dtype.is_unsigned():
                    method = builder.create_emitc_CastOp
                    return PlainValue(handle=method(self.to_ir(), dtype.to_ir()), dtype=dtype)
                method = builder.create_arith_ExtSIOp if ext else builder.create_arith_TruncIOp
            else:
                method = builder.create_arith_ExtFOp if ext else builder.create_arith_TruncFOp
        if method is None:
            raise NotImplementedError(f"Arithmetic cast from {self.dtype} to {dtype} is not supported")
        return PlainValue(handle=method(self.to_ir(), dtype.to_ir()), dtype=dtype)

    @require_jit
    def ceildiv(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "CeilDivSI", None)

    # Logical (bool) operations

    @require_jit
    def logical_and(self, other) -> PlainValue:
        return self.apply_bool_op(self, other, "AndI")

    @require_jit
    def logical_or(self, other) -> PlainValue:
        return self.apply_bool_op(self, other, "OrI")

    def to_ir(self) -> IRHandle:
        return self.handle


RuntimeBool: TypeAlias = Union[PlainValue, bool]
RuntimeInt: TypeAlias = Union[PlainValue, int]
RuntimeFloat: TypeAlias = Union[PlainValue, float]
RuntimeNumeric: TypeAlias = Union[RuntimeInt, RuntimeFloat]


def materialize_ir_value(value: RuntimeNumeric, required_type: Optional[DataType] = None) -> PlainValue:
    if isinstance(value, PlainValue):
        return value if required_type is None else value.cast(required_type)
    if isinstance(value, IRValue):
        if required_type is not None:
            raise ValueError(f"{required_type} cannot be specified for {value} which is not PlainValue")
        return value
    if isinstance(value, ConstExpr):
        return materialize_ir_value(value.value, required_type)
    if not isinstance(value, (int, float)):
        raise TypeError(f"Unsupported value type for materialization: {value.__class__.__name__}")
    if required_type is not None:
        if required_type == KT.bit:
            value = bool(value)
        if required_type.is_int():
            value = int(value)
        elif required_type.is_float():
            value = float(value)

    return convert_value(value, required_type)


def convert_value(value: Any, required_type: Optional[DataType] = None) -> PlainValue:
    builder = global_builder.get_ir_builder()

    type_to_builder = {
        bool: {"bit": builder.get_i1}, int: {
            "int1": builder.get_i1, "int8": builder.get_i8, "int16": builder.get_i16, "int32": builder.get_i32, "int64":
            builder.get_i64, "uint8": builder.get_ui8, "uint16": builder.get_ui16, "uint32": builder.get_ui32, "uint64":
            builder.get_ui64
        }, float: {"float16": builder.get_f16, "float32": builder.get_f32, "float64": builder.get_f64}
    }

    if isinstance(value, bool):
        if required_type is not None and required_type != KT.bit:
            raise ValueError("Required type must be None or KT.bit")
        return PlainValue(builder.get_i1(value))

    if isinstance(value, int):
        if required_type is None:
            required_type = KT.int_
        if str(required_type) not in type_to_builder[int]:
            raise ValueError(f"Unsupported DataType for materialization: {required_type}")
        factory = type_to_builder[int][str(required_type)]

    if isinstance(value, float):
        if required_type is None:
            required_type = KT.float_
        if str(required_type) not in type_to_builder[float]:
            raise ValueError(f"Unsupported DataType for materialization: {required_type}")
        factory = type_to_builder[float][str(required_type)]

    return PlainValue(factory(value), required_type)


def cast_to_index(value: Union[RuntimeNumeric, IRHandle]) -> IRHandle:
    builder = global_builder.get_ir_builder()
    if isinstance(value, int):
        return builder.get_index(value)
    if isinstance(value, PlainValue):
        return cast_to_index(value.to_ir())
    if isinstance(value, IRHandle):
        return builder.create_arith_IndexCastOp(value, builder.get_index_type())
    raise TypeError(f"Unsupported type for index materialization: {value.__class__.__name__}")
