# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import inspect
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Generator, Iterable, List, Literal, NoReturn, Optional, Tuple, Type, TypeVar,
                    Union, overload)
from typing_extensions import ParamSpec, TypeAlias

from .._C import ir
from .errors import CodegenError, UnsupportedSyntaxError
from .function import Function, FunctionLocation
from .name_scope import NameScope
from .specialization import BaseArgType, IRArgType, PointerArgType, PlainArgType, Specialization, StructArgType
from ..common.compat import get_annotations, merge_dict
from ..language.core.constexpr import ConstExpr
from ..language.core.dtype import KnownTypes
from ..language.core.ir_value import GlobalAddress, IRHandle, IRValue, PlainValue, materialize_ir_value
from ..language.core.range import BaseRange, static_range
from ..language.core.struct import BaseField, Struct
from ..language.core.utils import static_assert, global_builder

T = TypeVar("T")
P = ParamSpec("P")


@dataclass
class CodegenOptions:
    """Code generation and AST traversal options"""

    capture_exceptions: bool = True
    """
    Capture all exceptions raised by functions called by code generator internally and show the corresponding location
    in the user function code being traversed that caused an exception instead.
    Usually, it must always be enabled, but may be disabled for the language debugging purposes.
    """

    ir_multithreading: bool = True
    """
    Enable parallel processing for the current IR context if possible.
    Usually, it must always be enabled, but may be disabled for the IR debugging purposes.
    """


@dataclass
class ReturnType:
    py_type: Type[IRValue]
    ir_type: ir.Type


ReturnTypesDict: TypeAlias = Dict[str, List[ReturnType]]


@dataclass
class VisitorState:
    discard_everything: bool = False
    inside_function: bool = False
    return_allowed: bool = True
    return_types: List[ReturnType] = field(default_factory=list)
    visited_return_types: ReturnTypesDict = field(default_factory=dict)


@dataclass
class BlockInOut:
    block: ir.Block
    init_handles: Dict[str, IRHandle]
    yield_values: Dict[str, IRValue]
    yield_handles: Dict[str, IRHandle]


class FunctionVisitor(ast.NodeVisitor):

    def __init__(
        self,
        source_lines: Optional[List[str]],
        spec: Specialization,
        global_vars: Dict[str, Any],
        location: FunctionLocation,
        options: CodegenOptions,
        visited_return_types: Optional[ReturnTypesDict] = None,
        is_kernel: bool = True,
    ):
        super().__init__()
        self.src = source_lines
        self.ir_function: Optional[ir.FuncOp] = None
        self.spec = spec
        self.scope = NameScope(merge_dict(global_vars, spec.constexprs))
        self.location = location
        global_builder.get_ir_builder().set_insertion_point_to_start(global_builder.get_ir_module().get_body())
        global_builder.get_ir_builder().set_loc(self.location.filename, self.location.line_offset, 0)
        self.state = VisitorState()
        self.options = options
        self.is_kernel = is_kernel
        if visited_return_types:
            self.state.visited_return_types = visited_return_types

    @staticmethod
    def get_binary_method_name(op_class: Type[ast.operator]) -> str:
        names: Dict[Type[ast.operator], str] = {
            ast.Add: '__add__',
            ast.Sub: '__sub__',
            ast.Mult: '__mul__',
            ast.Div: '__truediv__',
            ast.FloorDiv: '__floordiv__',
            ast.Mod: '__mod__',
            ast.Pow: '__pow__',
            ast.LShift: '__lshift__',
            ast.RShift: '__rshift__',
            ast.BitAnd: '__and__',
            ast.BitOr: '__or__',
            ast.BitXor: '__xor__',
            ast.MatMult: '__matmul__',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__class__.__name__} is not implemented")

    @staticmethod
    def get_bool_method_name(op_class: Type[ast.boolop]) -> str:
        names: Dict[Type[ast.boolop], str] = {
            ast.And: 'logical_and',
            ast.Or: 'logical_or',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__name__} is not implemented")

    @staticmethod
    def get_unary_method_name(op_class: Type[ast.unaryop]) -> str:
        names: Dict[Type[ast.unaryop], str] = {
            ast.USub: '__neg__',
            ast.UAdd: '__pos__',
            ast.Not: '__not__',
            ast.Invert: '__invert__',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__name__} is not implemented")

    @staticmethod
    def get_compare_method_name(op_class: Type[ast.cmpop]) -> str:
        names: Dict[Type[ast.cmpop], str] = {
            ast.Eq: '__eq__',
            ast.NotEq: '__ne__',
            ast.Gt: '__gt__',
            ast.GtE: '__ge__',
            ast.Lt: '__lt__',
            ast.LtE: '__le__',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__name__} is not implemented")

    @staticmethod
    def has_builder_support(value) -> bool:
        return isinstance(value, IRValue)

    def raise_unsupported(self, node: ast.AST, message: Optional[str] = None, context: bool = False) -> NoReturn:
        error = UnsupportedSyntaxError(node, self.src, message)
        if context:
            raise error
        raise error from None

    def apply_binary_method(self, method_name, lhs, rhs) -> Any:
        reverse_method_name = re.sub(r"__(.*)__", r"__r\1__", method_name)
        if not self.has_builder_support(lhs) and self.has_builder_support(rhs):
            return getattr(rhs, reverse_method_name)(lhs)
        result = getattr(lhs, method_name)(rhs)
        if result is NotImplemented:
            result = getattr(rhs, reverse_method_name)(lhs)
        return result

    def get_call_args(self, func: Callable, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.arguments

    def call_jit_function(self, fn: Function, args: Tuple[Any], kwargs: Dict[str, Any]) -> Optional[Any]:
        base_fn = fn.fn
        call_args = self.get_call_args(base_fn, *args, **kwargs)
        annotations = get_annotations(base_fn)
        runtime_args, constexprs = Function.split_args(call_args, annotations)
        arg_values: Dict[str, IRValue] = {}
        for name, value in runtime_args.items():
            if isinstance(value, IRValue):
                arg_values[name] = value
            else:
                arg_values[name] = materialize_ir_value(value)
        arg_types = {name: IRArgType(type(value), value.to_ir().get_type()) for name, value in arg_values.items()}
        fn_name = fn.node.name
        ret_types = []
        if global_builder.get_ir_module().has_function(fn_name):
            ret_types = self.state.visited_return_types[fn_name]
        else:
            spec = Specialization(arg_types, constexprs)
            with self.visit_region():
                visitor = FunctionVisitor(fn.src, spec, base_fn.__globals__, fn.location, self.options,
                                          self.state.visited_return_types, is_kernel=False)
                visitor.visit(fn.node)
                ret_types = visitor.state.return_types
                self.state.visited_return_types[fn_name] = ret_types
        ir_operands = [value.to_ir() for value in arg_values.values()]
        op = global_builder.get_ir_builder().create_func_CallOp(fn.node.name, ir_operands,
                                                                [ret_type.ir_type for ret_type in ret_types])
        if len(ret_types) == 0:
            return None
        return_values = [ret_type.py_type.from_ir(op.get_result(i)) for i, ret_type in enumerate(ret_types)]
        if len(ret_types) == 1:
            return return_values[0]
        return return_values

    def compute_inout(self, node: ast.AST, stmts: List[ast.stmt], ind_var: Optional[Tuple[str, ir.Type]] = None,
                      make_args: bool = False) -> BlockInOut:
        with self.visit_region() as (outer_scope, _):
            block = ir.Block()
            if ind_var is not None:
                name, ir_type = ind_var
                arg = block.add_argument(ir_type)
                self.scope.save(name, PlainValue(arg))
            global_builder.get_ir_builder().set_insertion_point_to_start(block)
            self.visit_statements(stmts)
            for name in self.scope.redefined:
                old_value = outer_scope.lookup(name)
                new_value = self.scope.lookup(name)
                if type(old_value) is not type(new_value):
                    self.raise_unsupported(
                        node, f"'{name}' was re-assigned to an object with different type: "
                        f"initial type is {old_value.__class__.__name__}, new type is {new_value.__class__.__name__}")
            _, init_handles = self.mat_ir_values(outer_scope.lookup(name) for name in self.scope.redefined)
            if make_args:
                for handle in init_handles:
                    arg = block.add_argument(handle.get_type())
                    handle.replace_uses_in_block(block, arg)
            yield_values, yield_handles = self.mat_ir_values(self.scope.lookup(name) for name in self.scope.redefined)
            return BlockInOut(
                block=block,
                init_handles=dict(zip(self.scope.redefined, init_handles)),
                yield_values=dict(zip(self.scope.redefined, yield_values)),
                yield_handles=dict(zip(self.scope.redefined, yield_handles)),
            )

    def dereference_name(self, name: str) -> Optional[Any]:
        return self.scope.lookup(name)

    @overload
    def ensure_bool_value(self, node: ast.expr, require_ir: Literal[True]) -> PlainValue:
        ...

    @overload
    def ensure_bool_value(self, node: ast.expr, require_ir: bool = False) -> Union[PlainValue, bool]:
        ...

    def ensure_bool_value(self, node: ast.expr, require_ir: bool = False):
        value = ConstExpr.unwrap(self.visit(node))
        if isinstance(value, (int, float, type(None))):
            if not require_ir:
                return bool(value)
            value = materialize_ir_value(bool(value))
        if not isinstance(value, PlainValue):
            self.raise_unsupported(
                node, f"Condition expression must be evaluated as PlainValue, got {value.__class__.__name__}")
        return value.cast(KnownTypes.bit)

    def get_arg_value(self, arg_type: BaseArgType, handle: IRHandle) -> IRValue:
        if isinstance(arg_type, PointerArgType):
            return GlobalAddress(handle=handle, dtype=arg_type.dtype)
        if isinstance(arg_type, PlainArgType):
            return PlainValue(handle=handle, dtype=arg_type.dtype)
        if isinstance(arg_type, IRArgType):
            return arg_type.py_type.from_ir(handle)
        if isinstance(arg_type, StructArgType):
            return arg_type.py_type.from_ir(handle).create_local()
        raise NotImplementedError(f"Not implemented for {arg_type.__class__.__name__}")

    def mat_ir_values(self, values: Iterable[Any]) -> Tuple[Tuple[PlainValue, ...], Tuple[IRHandle, ...]]:
        ir_values = tuple(materialize_ir_value(value) for value in values)
        ir_handles = tuple(value.to_ir() for value in ir_values)
        return ir_values, ir_handles

    @contextmanager
    def nest_scope(self) -> Generator[None, Any, None]:
        outer_scope = self.scope
        self.scope = outer_scope.inherit()
        try:
            yield
        finally:
            self.scope = outer_scope

    def generic_visit(self, node: ast.AST) -> NoReturn:
        self.raise_unsupported(node, f"{node.__class__.__name__} syntax is not supported in JIT function")

    def visit(self, node: Optional[ast.AST]) -> Optional[Any]:
        if node is None:
            return None
        if self.state.discard_everything:
            return None
        if self.state.inside_function and isinstance(node, ast.FunctionDef):
            self.raise_unsupported(node, "Nested functions are not supported")
        if not self.state.inside_function and not isinstance(node, ast.FunctionDef):
            raise RuntimeError(f"JIT compilation is applicable to functions only, got {node.__class__.__name__} node")
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            global_builder.get_ir_builder().set_loc(self.location.filename, self.location.line_offset + node.lineno,
                                                    node.col_offset)
        try:
            return super().visit(node)
        except CodegenError:
            raise
        except Exception as e:
            if self.options.capture_exceptions:
                raise CodegenError(node, self.src, f"{e.__class__.__name__}: {e}") from e
            raise

    def visit_arguments(self, node: ast.arguments) -> Tuple[List[str], str]:
        if node.defaults or node.kw_defaults:
            self.raise_unsupported(node, "Default values for function arguments are not supported")
        if node.posonlyargs:
            self.raise_unsupported(node, "Positional-only arguments are not supported")
        if node.kwonlyargs:
            self.raise_unsupported(node, "Keyword-only arguments are not supported")
        arg_names = [str(self.visit(arg)) for arg in node.args]
        kwarg_name = str(self.visit(node.kwarg))
        return arg_names, kwarg_name

    def visit_arg(self, node: ast.arg) -> str:
        return node.arg

    @contextmanager
    def visit_region(self) -> Generator[Tuple[NameScope, ir.InsertPoint], Any, None]:
        outer_scope = self.scope
        self.scope = outer_scope.inherit()
        insert_point = global_builder.get_ir_builder().save_insertion_point()
        return_allowed = self.state.return_allowed
        self.state.return_allowed = False
        try:
            yield outer_scope.inherit(), insert_point
        finally:
            self.scope = outer_scope
            global_builder.get_ir_builder().restore_insertion_point(insert_point)
            self.state.return_allowed = return_allowed

    def visit_statements(self, stmts: List[ast.stmt]) -> None:
        for stmt in stmts:
            self.visit(stmt)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        return self.visit_Assign(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        test = self.visit(node.test)
        try:
            test = ConstExpr(test)
            static_assert(test, self.visit(node.msg))
        except TypeError:
            self.raise_unsupported(
                node,
                f"An assertion turned out to test a runtime value {test!r}, only compile-time values are supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        if len(targets) != 1:
            self.raise_unsupported(node, "Assignment operator must have exactly one target")
        lhs = targets[0]
        rhs = self.visit(node.value)
        if isinstance(lhs, ast.Subscript) and isinstance(lhs.ctx, ast.Store):
            base = self.visit(lhs.value)
            subscript = self.visit(lhs.slice)
            base.__setitem__(subscript, rhs)
            return
        if isinstance(lhs, ast.Attribute) and isinstance(lhs.ctx, ast.Store):
            base = self.visit(lhs.value)
            if setter := getattr(base, "__setattrjit__", None):
                setter(lhs.attr, rhs)
            else:
                setattr(base, lhs.attr, rhs)
            return
        lhs_names = []
        if isinstance(lhs, ast.Name):
            lhs_names.append(self.visit(lhs))
        elif isinstance(lhs, ast.Tuple) and all(isinstance(elt, ast.Name) for elt in lhs.elts):
            lhs_names.extend(self.visit(lhs))
        else:
            self.raise_unsupported(node, "Assignment target must be name or tuple of names")
        rhs_values = []
        if isinstance(rhs, Iterable) and len(lhs_names) != 1:
            if len(rhs) != len(lhs_names):
                self.raise_unsupported(
                    node, "Assignment operator must have equal number of names and values, "
                    f"got {len(lhs_names)} names and {len(rhs)} values")
            rhs_values.extend(rhs)
        else:
            rhs_values.append(rhs)
        if len(lhs_names) != len(rhs_values):
            raise RuntimeError("Assignment operator must have equal number of names and values")
        for name, value in zip(lhs_names, rhs_values):
            self.scope.save(name, value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        lhs = node.target
        if isinstance(lhs, ast.Name):
            lhs = ast.Name(lhs.id, ctx=ast.Load())
        elif isinstance(lhs, ast.Subscript):
            lhs = ast.Subscript(lhs.value, lhs.slice, ctx=ast.Load())
        else:
            self.raise_unsupported(node, f"{lhs.__class__.__name__} is not supported as left operand of AugAssign")
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        self.visit(assign)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        lhs = self.visit(node.value)
        attr = str(node.attr)
        try:
            value = getattr(lhs, attr)
            if not isinstance(lhs, Struct) or not isinstance(value, BaseField):  # else: __getattrjit__ must be called
                return value
        except AttributeError:
            pass
        getter = getattr(lhs, "__getattrjit__", None)
        if getter:
            return getter(attr)
        raise AttributeError(f"'{lhs.__class__.__name__}' object has no attribute '{attr}'")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        method_name = self.get_binary_method_name(type(node.op))
        return self.apply_binary_method(method_name, lhs, rhs)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if len(node.values) != 2:
            self.raise_unsupported(node, "Chained boolean operators are not supported, group pairs with parentheses")
        lhs = self.visit(node.values[0])
        rhs = self.visit(node.values[1])
        method_name = self.get_bool_method_name(type(node.op))
        return self.apply_binary_method(method_name, lhs, rhs)

    def visit_Call(self, node: ast.Call) -> Optional[Any]:
        fn = self.visit(node.func)
        if not callable(fn):
            self.raise_unsupported(node, f"{fn.__class__.__name__} instance is not callable")
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))
        kwargs = dict(self.visit(keyword) for keyword in node.keywords)
        if isinstance(fn, Function):
            return self.call_jit_function(fn, args, kwargs)
        return fn(*args, **kwargs)

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.comparators) != 1 or len(node.ops) != 1:
            self.raise_unsupported(node, "Only simple comparison is supported (one operation, one comparator)")
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Is):
            return lhs is rhs
        if isinstance(op, ast.IsNot):
            return lhs is not rhs
        method_name = self.get_compare_method_name(type(node.ops[0]))
        return self.apply_binary_method(method_name, lhs, rhs)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Expr(self, node: ast.Expr) -> Optional[Any]:
        return self.visit(node.value)

    def parse_iterator(self, node: ast.For) -> tuple:
        func = self.visit(node.iter.func)
        args = [self.visit(arg) for arg in node.iter.args]
        kwargs = dict(self.visit(keyword) for keyword in node.iter.keywords)
        return func, args, kwargs

    def handle_static_range(self, node: ast.For, args, kwargs, target):
        range_obj = static_range(*args, **kwargs)
        for i in range(range_obj.start, range_obj.stop, range_obj.step):
            self.scope.save(target, i)
            self.visit_statements(node.body)

    def visit_For(self, node: ast.For) -> None:
        if len(node.orelse) != 0:
            self.raise_unsupported(node, "else statement is not allowed after for-loop")
        target = self.visit(node.target)
        if not isinstance(target, str):
            self.raise_unsupported(node, f"For-loop target must be an identifier, got {target.__class__.__name__}")
        func, args, kwargs = self.parse_iterator(node)
        iter_args = None, None, None
        if func is static_range:
            self.handle_static_range(node, args, kwargs, target)
            return
        elif inspect.isclass(func) and issubclass(func, (range, BaseRange)):
            cls = BaseRange if issubclass(func, range) else func
            range_obj = cls(*args, **kwargs)
            iter_args = range_obj.start, range_obj.stop, range_obj.step
        else:
            self.raise_unsupported(
                node,
                "Only for-loops with range or asc.language.range or asc.language.static_range are supported",
            )
        builder = global_builder.get_ir_builder()
        start, stop, step = map(lambda s: materialize_ir_value(s, KnownTypes.int_), iter_args)
        if start.dtype != stop.dtype or start.dtype != step.dtype:
            self.raise_unsupported(node, "Loop bounds must have the same DataType")
        yields = {}
        with self.visit_region():
            block_inout = self.compute_inout(node, node.body, (target, start.to_ir().get_type()), make_args=True)
            op = builder.create_scf_ForOp(start.to_ir(), stop.to_ir(), step.to_ir(),
                                          list(block_inout.init_handles.values()))
            range_obj.handle_op(op)
            self.scope.save(target, PlainValue(op.get_induction_var()))
            body = op.get_body()
            body.clear()
            ir.inline_block_at_end(block_inout.block, body, body.get_arguments())
            builder.set_insertion_point_to_end(body)
            builder.create_scf_YieldOp(list(block_inout.yield_handles.values()))
            for i, (name, value) in enumerate(block_inout.yield_values.items()):
                yields[name] = value.from_ir(op.get_result(i))
        for name, value in yields.items():
            self.scope.save(name, value)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> str:
        value = self.visit(node.value)
        template = "{"
        if node.conversion >= 0:
            template += f"!{chr(node.conversion)}"
        spec = self.visit(node.format_spec)
        if spec:
            template += f":{spec}"
        template += "}"
        return template.format(value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.state.inside_function = True
        arg_types = self.spec.args.values()
        builder = global_builder.get_ir_builder()
        input_ir_types = [arg_type.to_ir() for arg_type in arg_types]
        self.ir_function = builder.create_func_FuncOp(node.name, builder.get_function_type(input_ir_types))
        self.ir_function.make_aicore()
        if self.is_kernel:
            self.ir_function.make_global()
        entry = self.ir_function.add_entry_block()
        arg_names = list(self.spec.args.keys())
        self.ir_function.set_arg_names(arg_names)
        builder.set_insertion_point_to_start(entry)
        for i, (name, arg) in enumerate(self.spec.args.items()):
            value = self.get_arg_value(arg, self.ir_function.get_arg(i))
            self.scope.save(name, value)
        self.visit_statements(node.body)
        if not entry.has_terminator():
            builder.create_func_ReturnOp()
        if self.state.return_types:
            result_ir_types = [ret_type.ir_type for ret_type in self.state.return_types]
            self.ir_function.set_type(builder.get_function_type(input_ir_types, result_ir_types))
        self.state.inside_function = False
        self.state.discard_everything = False

    def visit_If(self, node: ast.If) -> None:
        cond = self.ensure_bool_value(node.test)
        if isinstance(cond, bool):  # condition is known at compile-time
            if cond:
                self.visit_statements(node.body)
            else:
                self.visit_statements(node.orelse)
            return
        yields = {}
        with self.visit_region():
            then_inout = self.compute_inout(node, node.body)
            else_inout = self.compute_inout(node, node.orelse)

            def merge_sorted(dict1: Dict[str, T], dict2: Dict[str, T]) -> Dict[str, T]:
                dicts = merge_dict(dict1, dict2)
                return {key: dicts[key] for key in sorted(dicts.keys())}

            yield_values = merge_sorted(then_inout.yield_values, else_inout.yield_values)
            ret_types = [value.to_ir().get_type() for value in yield_values.values()]
            builder = global_builder.get_ir_builder()
            op = builder.create_scf_IfOp(cond.to_ir(), ret_types, with_else=True)

            def select(keys: Iterable[str], main: Dict[str, T], fallback: Dict[str, T]) -> Dict[str, T]:
                return {key: main[key] if key in main else fallback[key] for key in keys}

            then_block = op.get_then_block()
            then_block.clear()
            then_inout.block.merge_block_before(then_block)
            builder.set_insertion_point_to_end(then_block)
            then_yields = select(yield_values.keys(), then_inout.yield_handles, else_inout.init_handles)
            builder.create_scf_YieldOp(list(then_yields.values()))

            else_block = op.get_else_block()
            else_block.clear()
            else_inout.block.merge_block_before(else_block)
            builder.set_insertion_point_to_end(else_block)
            else_yields = select(yield_values.keys(), else_inout.yield_handles, then_inout.init_handles)
            builder.create_scf_YieldOp(list(else_yields.values()))

            for i, (name, value) in enumerate(yield_values.items()):
                yields[name] = value.from_ir(op.get_result(i))
        for name, value in yields.items():
            self.scope.save(name, value)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        cond = self.ensure_bool_value(node.test, require_ir=True)
        with self.visit_region() as (_, insert_point):
            then_block = ir.Block()
            builder = global_builder.get_ir_builder()
            builder.set_insertion_point_to_start(then_block)
            then_value = materialize_ir_value(self.visit(node.body))
            else_block = ir.Block()
            builder.set_insertion_point_to_start(else_block)
            else_value = materialize_ir_value(self.visit(node.orelse))
            builder.restore_insertion_point(insert_point)
            if then_value.dtype != else_value.dtype:
                self.raise_unsupported(
                    node,
                    f"Conditional operator has inconsistent result types: {then_value.dtype} / {else_value.dtype}")
            ret_type = then_value.dtype
            if not ret_type.is_numeric():
                self.raise_unsupported(node, f"Conditional operator must have numeric result type, got {ret_type}")
            op = builder.create_scf_IfOp(cond.to_ir(), [ret_type.to_ir()], with_else=True)
            then_block.merge_block_before(op.get_then_block())
            builder.set_insertion_point_to_end(op.get_then_block())
            builder.create_scf_YieldOp([then_value.to_ir()])
            else_block.merge_block_before(op.get_else_block())
            builder.set_insertion_point_to_end(op.get_else_block())
            builder.create_scf_YieldOp([else_value.to_ir()])
            return PlainValue(op.get_result(0))

    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        values = (self.visit(value) for value in node.values)
        return "".join(values)

    def visit_keyword(self, node: ast.keyword) -> Tuple[str, Any]:
        return node.arg, self.visit(node.value)

    def visit_List(self, node: ast.List) -> List[Optional[Any]]:
        return [self.visit(elt) for elt in node.elts]

    def visit_Name(self, node: ast.Name) -> Union[str, Optional[Any]]:
        if isinstance(node.ctx, ast.Store):
            return node.id
        value = self.dereference_name(node.id)
        return ConstExpr.unwrap(value)

    def visit_Pass(self, node: ast.Pass) -> None:
        pass

    def visit_Return(self, node: ast.Return) -> None:
        if not self.state.return_allowed:
            self.raise_unsupported(node, "Return statement is not allowed in nested blocks")
        value = self.visit(node.value)
        self.state.discard_everything = True
        if value is None:
            return
        if self.is_kernel:
            self.raise_unsupported(node, "JIT kernel function cannot return objects")
        values = []
        if isinstance(value, Iterable):
            values.extend(value)
        else:
            values.append(value)
        ir_values = [materialize_ir_value(value) for value in values]
        self.state.return_types = [ReturnType(type(value), value.to_ir().get_type()) for value in ir_values]
        global_builder.get_ir_builder().create_func_ReturnOp([value.to_ir() for value in ir_values])

    def visit_Slice(self, node: ast.Slice) -> slice:
        return slice(self.visit(node.lower), self.visit(node.upper), self.visit(node.step))

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if not isinstance(node.ctx, ast.Load):
            self.raise_unsupported(node, "Subscript operation is not allowed (must be Load context or assignment)")
        value = self.visit(node.value)
        slices = self.visit(node.slice)
        return value.__getitem__(slices)

    def visit_Tuple(self, node: ast.Tuple) -> Tuple[Optional[Any], ...]:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        method_name = self.get_unary_method_name(type(node.op))
        return getattr(operand, method_name)()

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) != 1:
            self.raise_unsupported(node, "Only one item in with-statement is supported")
        item = node.items[0]
        context = self.visit(item.context_expr)
        with self.nest_scope():
            entered = context.__enter__()
            if isinstance(item.optional_vars, ast.Name):
                self.scope.save(self.visit(item.optional_vars), entered)
            self.visit_statements(node.body)
            context.__exit__(None, None, None)

    def visit_While(self, node: ast.While) -> None:
        if len(node.orelse) != 0:
            self.raise_unsupported(node, "else statement is not allowed after while-loop")
        builder = global_builder.get_ir_builder()
        after_inout = self.compute_inout(node, node.body, make_args=True)
        ret_types = [handle.get_type() for handle in after_inout.init_handles.values()]
        op = builder.create_scf_WhileOp(ret_types, list(after_inout.init_handles.values()))
        with self.visit_region():
            before_block = builder.create_block(op.get_before(), ret_types)
            builder.set_insertion_point_to_start(before_block)
            cond = self.ensure_bool_value(node.test, require_ir=True)
            builder.create_scf_ConditionOp(cond.to_ir(), before_block.get_arguments())
        with self.visit_region():
            after_block = builder.create_block(op.get_after(), ret_types)
            ir.inline_block_at_end(after_inout.block, after_block, after_block.get_arguments())
            builder.set_insertion_point_to_end(after_block)
            builder.create_scf_YieldOp(list(after_inout.yield_handles.values()))
        for i, (name, value) in enumerate(after_inout.yield_values.items()):
            self.scope.save(name, value.from_ir(op.get_result(i)))
