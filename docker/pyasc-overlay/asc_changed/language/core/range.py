# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from ..._C import ir
from .ir_value import RuntimeInt


class BaseRange:

    @overload
    def __init__(self, stop: int, /):
        ...

    @overload
    def __init__(self, start: int, stop: int, /):
        ...

    @overload
    def __init__(self, start: int, stop: int, step: int, /):
        ...

    def __init__(self, *args):
        if len(args) < 1 or len(args) > 3:
            raise ValueError(f"range expects from 1 to 3 arguments, got {len(args)}")
        self.start: RuntimeInt = 0
        self.stop: RuntimeInt = 0
        self.step: RuntimeInt = 1
        if len(args) == 1:
            self.stop = args[0]
        elif len(args) >= 2:
            self.start = args[0]
            self.stop = args[1]
        if len(args) == 3:
            self.step = args[2]

    def __iter__(self):
        return self

    def __next__(self) -> int:
        raise NotImplementedError("This function must not be called")

    def handle_op(self, op: ir.ForOp) -> None:
        pass


class range(BaseRange):
    pass


class static_range:

    @overload
    def __init__(self, stop: int, /):
        ...

    @overload
    def __init__(self, start: int, stop: int, /):
        ...

    @overload
    def __init__(self, start: int, stop: int, step: int, /):
        ...

    def __init__(self, *args):
        if len(args) < 1 or len(args) > 3:
            raise ValueError(f"range expects from 1 to 3 arguments, got {len(args)}")
        self.start: int = 0
        self.stop: int = 0
        self.step: int = 1
        if len(args) == 1:
            self.stop = args[0]
        elif len(args) >= 2:
            self.start = args[0]
            self.stop = args[1]
        if len(args) == 3:
            self.step = args[2]

    def __iter__(self):
        return self

    def __next__(self) -> int:
        raise NotImplementedError("This function must not be called")
