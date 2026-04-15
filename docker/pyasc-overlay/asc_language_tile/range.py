# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, overload

from ..._C import ir
from ..core.range import BaseRange
from ..core.utils import check_type, global_builder


class range(BaseRange):

    @overload
    def __init__(self, start: int, stop: Optional[int] = None, step: int = 1, /, *, unroll_factor: int = 1,
                 parallel: bool = False):
        ...

    def __init__(self, *args, unroll_factor: int = 1, parallel: bool = False):
        check_type("unroll_factor", unroll_factor, int)
        check_type("parallel", parallel, bool)
        if unroll_factor < 1:
            raise ValueError(f"Loop unroll factor must be 1 or greater, got {unroll_factor}")
        super().__init__(*args)
        self.unroll_factor = unroll_factor
        self.parallel = parallel

    def handle_op(self, op: ir.ForOp) -> None:
        builder = global_builder.get_ir_builder()
        op.set_attr(ir.attr.unroll_factor, builder.get_index_attr(self.unroll_factor))
        if self.parallel:
            op.set_attr(ir.attr.parallel, builder.get_unit_attr())
