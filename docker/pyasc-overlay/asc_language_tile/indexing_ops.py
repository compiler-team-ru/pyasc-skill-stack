# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from contextlib import contextmanager
from typing import Any, Generator, Iterable, Optional, Union, overload

from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from ..core.utils import check_type, global_builder
from .tile import Tile
from .utils import create_tile, infer_common_dtype


def where(mask: Tile, src0: Union[Tile, RuntimeNumeric], src1: Union[Tile, RuntimeNumeric]) -> Tile:
    check_type("mask", mask, Tile)
    src_dtype = infer_common_dtype(src0, src1)
    src0 = create_tile(src0, src_dtype, mask.shape)
    src1 = create_tile(src1, src_dtype, mask.shape)
    handle = global_builder.get_ir_builder().create_asctile_SelectOp(src0.to_ir().get_type(), mask.to_ir(),
                                                                     src0.to_ir(), src1.to_ir())
    return Tile(handle)


@overload
def mask(*, count: RuntimeInt, other: Optional[RuntimeNumeric] = None) -> Generator[None, Any, None]:
    ...


@overload
def mask(*, bits: Iterable[RuntimeInt], other: Optional[RuntimeNumeric] = None) -> Generator[None, Any, None]:
    ...


@contextmanager
def mask(*, count: Optional[RuntimeInt] = None, bits: Optional[Iterable[RuntimeInt]] = None,
         other: Optional[RuntimeNumeric] = None) -> Generator[None, Any, None]:
    builder = global_builder.get_ir_builder()
    other = _mat(other).to_ir() if other is not None else None
    if count is not None:
        mask_op = builder.create_asctile_CountMaskOp(_mat(count, KT.int64).to_ir(), other)
    elif bits is not None:
        bits = tuple(bits)
        if len(bits) != 2:
            raise RuntimeError(f"Expected two integers in 'bits', got {len(bits)}")
        mask_op = builder.create_asctile_BitwiseMaskOp(
            _mat(bits[0], KT.int64).to_ir(),
            _mat(bits[1], KT.int64).to_ir(), other)
    else:
        raise ValueError("One of 'count', 'bits' must be provided")
    old_insertion_point = global_builder.get_ir_builder().save_insertion_point()
    mask_region = mask_op.get_region()
    new_block = builder.create_block(mask_region)
    builder.set_insertion_point_to_start(new_block)
    try:
        yield
    finally:
        builder.create_asctile_YieldOp()
        builder.restore_insertion_point(old_insertion_point)
