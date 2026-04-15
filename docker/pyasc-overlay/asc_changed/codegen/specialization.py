# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import abc
from typing import Dict, Optional, Type

from .._C import ir
from ..language.core.constexpr import ConstExpr
from ..language.core.dtype import DataType
from ..language.core.ir_value import IRValue
from ..language.core.struct import Struct


class BaseArgType(abc.ABC):

    @abc.abstractmethod
    def to_ir(self) -> ir.Type:
        raise NotImplementedError


class PointerArgType(BaseArgType):

    def __init__(self, dtype: DataType):
        self.dtype = dtype

    def to_ir(self) -> ir.Type:
        return ir.get_unranked_memref_type(self.dtype.to_ir(), ir.AddressSpace.gm)


class PlainArgType(BaseArgType):

    def __init__(self, dtype: DataType):
        self.dtype = dtype

    def to_ir(self) -> ir.Type:
        return self.dtype.to_ir()


class IRArgType(BaseArgType):

    def __init__(self, py_type: Type[IRValue], ir_type: ir.Type):
        if not issubclass(py_type, IRValue):
            raise TypeError("Only IRValue can be passed between JIT functions")
        self.py_type = py_type
        self.ir_type = ir_type

    def to_ir(self) -> ir.Type:
        return self.ir_type


class StructArgType(BaseArgType):

    def __init__(self, py_type: Type[Struct]):
        if not issubclass(py_type, Struct):
            raise TypeError("Only Struct can be passed from host to device")
        self.py_type = py_type

    def to_ir(self):
        return ir.get_memref_type(self.py_type.get_ir_type(), ir.dynshape, ir.AddressSpace.gm)


class Specialization:

    def __init__(self, args: Dict[str, BaseArgType], constexprs: Optional[Dict[str, ConstExpr]] = None):
        self.args = args
        self.constexprs = {} if constexprs is None else constexprs
