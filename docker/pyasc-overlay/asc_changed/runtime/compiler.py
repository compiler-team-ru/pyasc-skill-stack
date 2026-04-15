# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, final

from .._C import ir, passes, translation
from ..lib.runtime import CoreType, get_soc_version
from ..lib.utils import get_ascend_path
from .config import KernelType
from . import utils


@dataclass
class CompileOptions:
    """Binary compilation and IR transformation options"""

    debug: bool = False
    strip_loc: bool = False
    verify_sync: bool = False
    print_ir_before_all: bool = False
    run_passes: bool = True
    kernel_type: Optional[KernelType] = None
    auto_sync: Optional[bool] = True
    auto_sync_log: Optional[str] = ""
    matmul_cube_only: bool = False

    always_compile: bool = False
    """
    Always run full compilation pipeline instead of fetching a cached objects from the prevoius runs.
    This option may be useful for testing and parallel execution.
    """

    bisheng_options: Optional[Tuple[str]] = None
    """
    Append extra arguments to the :code:`bisheng` command line used to produce an object file for the kernel.
    Please, run :code:`bisheng --help` to get the list of options supported by the compiler.
    """

    opt_level: int = 3
    """
    Optimization level for the Bisheng compiler. Supported values are :code:`1`, :code:`2`, :code:`3`.
    Typically, this parameter affects the :code:`-O` argument of the command line for the compiler.
    """

    run_asc2_passes: bool = False
    """
    Enable PyAsc2 compilation pipeline.
    **This option is enabled automatically** when :code:`@asc2.jit` decorator is used.
    """

    densify_load_store: bool = False
    """
    Densify :py:obj:`asc2.load` and :py:obj:`asc2.store` statements by grouping them together.

    .. warning::
        This is an experimental feature. It might or might not cause functional or performance regressions.
    """

    insert_sync: Optional[bool] = None
    """
    Insert synchronization instructions automatically.
    **This feature is enabled by default**, which is usually a must, but may be disabled for the debugging purposes.
    """

    reuse_ub: bool = False
    """
    Try to reduce the UB memory usage by replacing the tiles with those allocated earlier but became unused.
    Having this feature enabled may help to avoid UB overflow but may introduce performance regressions.
    """

    reuse_ub_in_out: bool = False
    """
    Allow for input/output tiles to be reused alongside intermediate ones.
    This option only has effect if :code:`reuse_ub` feature is enabled and :code:`densify_load_store` feature is not.

    .. warning::
        This is an experimental feature. It might or might not cause functional or performance regressions.
    """

    static_alloc: bool = False
    """
    Perform static allocation for tiles instead of relying on Ascend C TPipe backend.
    The static allocation feature may help to reduce an overhead caused by scalar code.
    """

    vf_fusion: bool = False
    """
    Perform the merging of vector blocks. 
    The function can eliminate unnecessary memory accesses and improve data locality.
    """


class CompilePlatform(Enum):
    """get soc version"""
    Ascend910B = "Ascend910B"
    Ascend910_93 = "Ascend910_93"
    Ascend910_95 = "Ascend910_95"


@dataclass(frozen=True)
class CompilationTarget:
    common_arch: Optional[str] = None
    vec_arch: Optional[str] = None
    cube_arch: Optional[str] = None
    common_options: List[str] = field(default_factory=list)
    vec_options: List[str] = field(default_factory=list)
    cube_options: List[str] = field(default_factory=list)

    @staticmethod
    def get(kernel_type: KernelType, platform: CompilePlatform) -> CompilationTarget:
        if platform in [CompilePlatform.Ascend910B, CompilePlatform.Ascend910_93, CompilePlatform.Ascend910_95]:
            common_option = [
                "-std=c++17", "--cce-disable-kernel-global-attr-check", "-mllvm", "-cce-aicore-stack-size=0x8000",
                "-mllvm", "-cce-aicore-function-stack-size=0x8000", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false"
            ]
            if platform == CompilePlatform.Ascend910B or platform == CompilePlatform.Ascend910_93:
                arch = "c220"
            elif platform == CompilePlatform.Ascend910_95:
                arch = "c310"
            if kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
                return CompilationTarget(vec_arch="dav-%s-vec" % arch, cube_arch="dav-%s-cube" % arch,
                                         common_options=common_option)
            elif kernel_type in [KernelType.MIX_AIV_1_0, KernelType.MIX_AIV_HARD_SYNC, KernelType.AIV_ONLY]:
                return CompilationTarget(common_arch="dav-%s-vec" % arch, common_options=common_option)
            else:
                return CompilationTarget(common_arch="dav-%s-cube" % arch, common_options=common_option)
        raise RuntimeError(f"Compilation is not supported for {CompilePlatform.value} platform")


@dataclass(frozen=True)
class CompiledKernel:
    binary: Optional[bytes] = None
    core_type: CoreType = CoreType.VectorCore
    kernel_args: Optional[Tuple[ir.KernelArgument]] = None
    enable_debug: bool = False
    memory_consumed: Optional[Dict[str, int]] = None


class Compiler:

    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = CompileOptions() if options is None else options
        self.soc_version = get_soc_version()
        if not self._check_compile_options():
            raise RuntimeError("Please check input compile option")
        self.dump_dir: Optional[Path] = None
        self.platform = CompilePlatform.Ascend910B
        if self.soc_version.value.startswith("Ascend910_93"):
            self.platform = CompilePlatform.Ascend910_93
        elif self.soc_version.value.startswith("Ascend910_95") or self.soc_version.value.startswith("Ascend950PR_95"):
            self.platform = CompilePlatform.Ascend910_95

        dump_dir = os.environ.get("PYASC_DUMP_PATH", None)
        if dump_dir is not None:
            try:
                self.dump_dir = Path(dump_dir).resolve()
            except OSError as e:
                raise RuntimeError("Get {} realpath failed.".format(str(dump_dir))) from e
        utils.FileUtils.create_dir(self.dump_dir)
        self.enable_debug = False
        compiler = shutil.which(os.environ.get("PYASC_COMPILER", "bisheng"))
        if compiler is None:
            raise RuntimeError("Compiler executable is not found, check PYASC_COMPILER environment variable")
        self.compiler = compiler
        linker = shutil.which(os.environ.get("PYASC_LINKER", "ld.lld"))
        if linker is None:
            raise RuntimeError("Linker executable is not found, check PYASC_LINKER environment variable")
        self.linker = linker

    @staticmethod
    def run_translation(mod: ir.ModuleOp) -> str:
        return translation.ir_to_ascendc(mod)

    @staticmethod
    def add_unroll_loop(pm: passes.PassManager) -> None:
        passes.asctile.add_unroll_loop(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

    def _schedule_lowering(self, pm: passes.PassManager) -> None:
        platform_95 = self.platform == CompilePlatform.Ascend910_95
        reuse_ub_in_out = not self.options.densify_load_store and self.options.reuse_ub and self.options.reuse_ub_in_out
        passes.ascendc.add_privatize_func(pm)
        passes.common.add_inliner(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_reconcile_unrealized_casts(pm)
        if self.options.run_asc2_passes:
            passes.asctile.add_tag_unroll_groups(pm)
            if not reuse_ub_in_out:
                self.add_unroll_loop(pm)
            if self.options.densify_load_store:
                passes.asctile.add_promote_pure_operations(pm)
                passes.common.add_canonicalizer(pm)
                passes.asctile.add_densify_unroll_groups(pm)
            passes.asctile.add_legalize_matmul(pm)
            passes.common.add_canonicalizer(pm)
            passes.asctile.add_transform_math_ops(pm)
            if platform_95:
                passes.asctile.add_unscalarize_reduction(pm)
                passes.common.add_canonicalizer(pm)
                passes.common.add_cse(pm)
            passes.asclower.add_expand_math(pm)
            passes.asclower.add_redress_i1_tile(pm)
            passes.asclower.add_lower_arith(pm)
            passes.asclower.add_lower_arith_binary(pm)
            passes.asclower.add_lower_atomic(pm)
            passes.asclower.add_lower_asctile(pm)
            passes.asclower.add_lower_asctile_i1(pm)
            passes.asclower.add_lower_math(pm)
            passes.asclower.add_lower_scf(pm)
            passes.common.add_canonicalizer(pm)
            passes.asclower.add_realize_conversion_cast(pm)
            passes.asclower.add_expand_mask(pm)
            passes.ascendc.add_fill_asc_operands(pm)
        passes.ascendc.add_input_output_tensor(pm)
        if self.options.reuse_ub:
            if reuse_ub_in_out:
                passes.ascendc.add_reuse_ub_allocation(pm, reuse_in_out=True)
                self.add_unroll_loop(pm)
            passes.ascendc.add_reuse_ub_allocation(pm)
            passes.common.add_canonicalizer(pm)
        passes.ascendc.add_hoist_ub_allocation(pm, exclude_in_out=not platform_95)
        if self.options.vf_fusion:
            passes.ascendc.add_fuse_vf_block(pm)
        if self.options.static_alloc:
            passes.ascendc.add_allocate_tensor(pm)
        else:
            passes.ascendc.add_materialize_tensor(pm, always_buf=platform_95)
        passes.ascendc.add_unify_pipe(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

    def _schedule_optimizing(self, pm: passes.PassManager) -> None:
        passes.common.add_licm(pm)
        passes.common.add_sccp(pm)
        passes.common.add_canonicalizer(pm)
        if self.options.insert_sync:
            passes.ascendc.add_erase_sync(pm)
            passes.ascendc.add_hoist_que_bind(pm)
            if self.platform != CompilePlatform.Ascend910_95:
                passes.ascendc.add_insert_sync(pm)
            else:
                passes.ascendc.add_insert_bufid_sync(pm)
                passes.common.add_canonicalizer(pm)
                passes.ascendc.add_fuse_bufid_sync(pm)
            passes.ascendc.add_unify_pipe(pm)
            passes.common.add_canonicalizer(pm)

    @staticmethod
    def _run_cmd(cmd: List[str], cmd_type: str) -> None:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        is_compile_cmd = cmd_type == "compile"
        if proc.returncode != 0:
            if is_compile_cmd and "--cce-aicore-only" in cmd:
                cmd_idx_four = 4
                cmd_idx_five = 5
                cmd.insert(cmd_idx_four, "-mllvm")
                cmd.insert(cmd_idx_five, "-disable-machine-licm")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                out, ret = proc.communicate()
                if ret == 0:
                    return
            raise RuntimeError("{} failed!\nError message is {}\nPlease rerun {}".format(
                cmd_type, out.decode(), (" ".join(cmd))))

    @final
    def run(self, mod: ir.ModuleOp, func_name: str) -> CompiledKernel:
        utils.FileUtils.dump_file(self.dump_dir, "codegen.mlir", str(mod))
        mod.set_attr(ir.attr.soc_version, ir.Builder(mod.op).get_str_attr(self.soc_version.value))
        if self.options.run_passes:
            self.run_passes(mod)
        utils.FileUtils.dump_file(self.dump_dir, "ascir.mlir", str(mod))
        source = self.run_translation(mod)
        if self.enable_debug:
            source = self._gen_init_dump_code(source, func_name)
        utils.FileUtils.dump_file(self.dump_dir, "ascendc.cpp", source)
        kernel_args = ir.get_kernel_arg_attrs(mod)
        return self.run_compilation(source, kernel_args, enable_debug=self.enable_debug,
                                    memory_consumed=mod.op.get_dict_of_int_attr(ir.attr.memory_consumed))

    def run_passes(self, mod: ir.ModuleOp) -> None:
        pm = passes.PassManager(mod.get_context())
        pm.enable_verifier()
        if self.options.print_ir_before_all:
            pm.enable_printing()
        if self.options.insert_sync is None:
            self.options.insert_sync = mod.need_insert_sync()
        self._schedule_passes(pm)
        pm.run(mod)
        if self.options.kernel_type is None:
            kernel_type = mod.op.get_str_attr(ir.attr.kernel_type)
            if kernel_type == "mixed":
                self.options.kernel_type = (KernelType.AIC_ONLY
                                            if self.options.matmul_cube_only else KernelType.MIX_AIC_1_2)
            elif kernel_type == "vector":
                self.options.kernel_type = KernelType.AIV_ONLY
            elif kernel_type == "cube":
                self.options.kernel_type = KernelType.AIC_ONLY
        self.enable_debug = (mod.op.has_unit_attr("asc.enable_debug")
                             and str(os.environ.get("ASCENDC_DUMP", "True")).lower() == "true")

    def run_compilation(self, source: str, kernel_args: Optional[Tuple[ir.KernelArgument]] = None,
                        **compiled_kernel_args) -> CompiledKernel:
        with tempfile.TemporaryDirectory(prefix="pyasc_compiler_") as tmp_dir:
            src = Path(tmp_dir) / "input.cce"
            src.write_text(source)
            dst = Path(tmp_dir) / "output.o"
            self._gen_dst_kernel(tmp_dir, src, dst)
            if self.dump_dir is not None:
                shutil.copyfile(dst, self.dump_dir / "binary.o")
            if self.options.kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
                core_type = CoreType.AiCore
            elif self.options.kernel_type in [
                    KernelType.AIV_ONLY, KernelType.MIX_AIV_HARD_SYNC, KernelType.MIX_AIV_1_0
            ]:
                core_type = CoreType.VectorCore
            else:
                core_type = CoreType.CubeCore
            return CompiledKernel(dst.read_bytes(), core_type, kernel_args, **compiled_kernel_args)

    def _check_compile_options(self) -> bool:
        is_soc_version_valid = self.soc_version.value.startswith("Ascend910B") or \
            self.soc_version.value.startswith("Ascend910_93") or self.soc_version.value.startswith("Ascend910_95") or \
            self.soc_version.value.startswith("Ascend950PR_95")
        is_core_type_valid = self.options.kernel_type is None or (isinstance(self.options.kernel_type, KernelType) and \
            self.options.kernel_type.value <= 7 and self.options.kernel_type.value >= 0)
        is_opt_level_valid = self.options.opt_level in [1, 2, 3]
        return is_soc_version_valid and is_core_type_valid and is_opt_level_valid

    def _schedule_postprocessing(self, pm: passes.PassManager) -> None:
        passes.ascendc.add_declare_py_struct(pm)
        passes.ascendc.add_generate_boilerplate(pm)
        if self.options.matmul_cube_only:
            passes.ascendc.add_define_cube_only(pm)
        passes.ascendc.add_legalize_kernel_args(pm, set_ffts_addr=(self.platform != CompilePlatform.Ascend910_95))
        passes.ascendc.add_detect_kernel_type(pm)
        passes.ascendc.add_detect_enable_debug(pm)
        if self.options.verify_sync:
            passes.ascendc.add_verify_sync(pm)
        if self.options.strip_loc:
            passes.common.add_strip_debug_info(pm)
        if self.options.run_asc2_passes:
            passes.ascendc.add_compute_memory_consumption(pm)

    def _schedule_passes(self, pm: passes.PassManager) -> None:
        self._schedule_lowering(pm)
        self._schedule_optimizing(pm)
        self._schedule_postprocessing(pm)

    def _gen_init_dump_code(self, source: str, func_name: str) -> str:
        dump_code = ""
        dump_code += "  #if defined ASCENDC_DUMP\n"
        dump_code += "    constexpr uint32_t ascendc_one_core_dump_size = " + str(utils.ONE_CORE_DUMP_SIZE) + ";\n"
        if self.options.kernel_type in [
                KernelType.MIX_AIC_1_0, KernelType.MIX_AIV_1_0, KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2
        ]:
            dump_code += "    AscendC::InitDump(true, dump_addr, ascendc_one_core_dump_size);\n"
        else:
            dump_code += "    AscendC::InitDump(false, dump_addr, ascendc_one_core_dump_size);\n"
        if self.enable_debug:
            dump_code += "    uint64_t asc_timestamp = 0;\n"
            dump_code += "    uint64_t asc_version = 0;\n"
            dump_code += "    __gm__ char* asc_version_str = nullptr;\n"
            dump_code += "    GetCannVersion(asc_version_str, asc_version, asc_timestamp);\n"
            dump_code += "    if (asc_timestamp == 0) {\n"
            dump_code += "      AscendC::printf(\"[WARNING]: CANN TimeStamp is invalid, \
                CANN TimeStamp is %u\\n\", asc_timestamp);\n"

            dump_code += "    } else {\n"
            dump_code += "      AscendC::printf(\"CANN Version: %s, TimeStamp: %u\\n\", \
                (__gm__ const char*)(asc_version_str), asc_timestamp);\n"

            dump_code += "    }\n"
        dump_code += "  #endif\n"
        source_lines = source.split('\n')
        kernel_code_with_dump = ""
        for line in source_lines:
            if func_name in line and "__aicore__" in line:
                split_line = line.split(")")
                new_line = split_line[0] + ", __gm__ uint8_t* dump_addr)" + split_line[1]
                kernel_code_with_dump += new_line + "\n"
                kernel_code_with_dump += dump_code
            else:
                kernel_code_with_dump += line + "\n"
        return kernel_code_with_dump

    def _gen_dst_kernel(self, tmp_dir: str, src: Path, dst: Path) -> None:
        target = CompilationTarget.get(self.options.kernel_type, self.platform)
        common_options = []
        ascend_path = get_ascend_path()
        tikcpp_path = os.path.realpath(os.path.join(ascend_path, "compiler", "tikcpp"))
        cann_version_file_path = os.path.join(tikcpp_path, "..", "..", "include", "version", "cann_version.h")
        if os.path.exists(cann_version_file_path):
            common_options += ["-include", cann_version_file_path]
        common_options += [
            "-I",
            os.path.join(tikcpp_path, "tikcfw"), "-I",
            os.path.join(tikcpp_path, "tikcfw/impl"), "-I",
            os.path.join(tikcpp_path, "tikcfw/interface")
        ]
        if self.options.bisheng_options:
            common_options.extend(self.options.bisheng_options)
        common_options.extend(target.common_options)
        if self.options.kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
            dst_cube_file = Path(tmp_dir) / "output_cube.o"
            cmds = self._get_compiler_cmd(target.cube_arch, src, dst_cube_file, common_options)
            self._run_cmd(cmds, "compile")
            dst_vec_file = Path(tmp_dir) / "output_vec.o"
            cmds = self._get_compiler_cmd(target.vec_arch, src, dst_vec_file, common_options)
            self._run_cmd(cmds, "compile")
            link_cmd = [
                self.linker, "-m", "aicorelinux", "-Ttext=0",
                "%s" % str(dst_cube_file),
                "%s" % str(dst_vec_file), "-static", "-o",
                "%s" % str(dst)
            ]
            self._run_cmd(link_cmd, "link")
        elif self.options.kernel_type in [
                KernelType.MIX_AIC_1_0, KernelType.MIX_AIC_HARD_SYNC, KernelType.MIX_AIV_1_0,
                KernelType.MIX_AIV_HARD_SYNC
        ]:
            dst_mix_file = Path(tmp_dir) / "output_mix_aic.o"
            if self.options.kernel_type in [KernelType.MIX_AIV_1_0, KernelType.MIX_AIV_HARD_SYNC]:
                dst_mix_file = Path(tmp_dir) / "output_mix_aiv.o"
            cmds = self._get_compiler_cmd(target.common_arch, src, dst_mix_file, common_options)
            self._run_cmd(cmds, "compile")
            link_cmd = [
                self.linker, "-m", "aicorelinux", "-Ttext=0",
                "%s" % str(dst_mix_file), "-static", "-o",
                "%s" % str(dst)
            ]
            self._run_cmd(link_cmd, "link")
        else:
            cmds = self._get_compiler_cmd(target.common_arch, src, dst, common_options)
            self._run_cmd(cmds, "compile")
            link_cmd = [self.linker, "-m", "aicorelinux", "-Ttext=0", "%s" % str(dst), "-static", "-o", "%s" % str(dst)]
            self._run_cmd(link_cmd, "link")

    def _get_compiler_cmd(self, arch: str, src_path: Path, dst_path: Path, common_options: List[str]) -> List[str]:
        opt_level = "-O" + str(self.options.opt_level)
        compile_cmds = [self.compiler, "-c", "-x", "cce", opt_level]
        compile_cmds += [str(src_path), "--cce-aicore-arch=%s" % arch, "--cce-aicore-only", "-o", str(dst_path)]
        compile_cmds += common_options
        if not self.enable_debug:
            compile_cmds += ["-DASCENDC_DUMP=0"]
        else:
            compile_cmds += ["-DASCENDC_DUMP=1"]
        if self.options.debug:
            compile_cmds += ["-g", "-mllvm", "--cce-aicore-jump-expand=true"]
        if self.options.auto_sync:
            compile_cmds += ["--cce-auto-sync", "-mllvm", "-api-deps-filter"]
            if self.options.auto_sync_log != "":
                compile_cmds += ["-cce-auto-sync-log=%s" % self.options.auto_sync_log]
        if self.options.kernel_type in [KernelType.MIX_AIC_1_1, KernelType.MIX_AIC_1_2]:
            compile_cmds += ["-cce-enable-mix"]
            compile_cmds += ["-D__MIX_CORE_MACRO__=1"]
        if self.options.kernel_type == KernelType.MIX_AIC_1_1:
            compile_cmds += ["-D__MIX_CORE_AIC_RATION__=1"]
        compile_cmds += ["-D__NPU_TILING__", "-DTILING_KEY_VAR=0"]
        return compile_cmds
