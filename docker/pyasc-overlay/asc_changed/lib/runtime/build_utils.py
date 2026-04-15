# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import functools
import os
import platform
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pybind11

from asc.runtime import config


@functools.lru_cache(None)
def get_ascend_path() -> str:
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError("ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first")
    return Path(path)


def build_npu_ext(obj_name: str, is_model: bool, soc: config.Platform, src_path: str, src_dir: str) -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = os.path.join(src_dir, f"lib{obj_name}{suffix}")

    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = gxx if gxx is not None else clangxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    # find the python library
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use pyasc with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd = [cxx, src_path, "-w", f"-I{py_include_dir}"]

    arch = platform.machine()
    # find the ascend library
    asc_path = get_ascend_path()

    pkg_include_path = os.path.join(asc_path, f'{arch}-linux', 'pkg_inc')
    profiling_include_path = os.path.join(asc_path, f'{arch}-linux', 'pkg_inc/profiling')
    runtime_include_path = os.path.join(asc_path, f'{arch}-linux', 'pkg_inc/runtime')
    if Path(pkg_include_path).exists() and Path(profiling_include_path).exists()  \
        and Path(runtime_include_path).exists():
        include_path = [
            f"-I{pkg_include_path}", f"-I{profiling_include_path}", f"-I{runtime_include_path}",
            "-DSEPARATE_PKG_ARCH"]
    else:
        include_path = [
            f"-I{os.path.join(asc_path, 'include/experiment')}",
            f"-I{os.path.join(asc_path, 'include/experiment/msprof')}"]

    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{pybind11.get_include()}",
        f"-L{os.path.join(asc_path, 'lib64')}"]

    cc_cmd += include_path

    if is_model:
        cc_cmd += [f"-L{os.path.join(asc_path, f'tools/simulator/{soc.value}/lib')}", "-lruntime_camodel"]
    else:
        cc_cmd += ["-lruntime", "-lmsprofiler"]

    cc_cmd += ["-lascendcl", "-std=c++17", "-shared", "-fPIC", "-o", so_path]
    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so_path
    else:
        raise RuntimeError("Failed to compile " + src_path)


if __name__ == "__main__":
    build_npu_ext("npu_utils", True, os.path.join("./", "npu_utils.cpp"), "./")
    build_npu_ext("rts_wrapper", True, os.path.join("./", "rt_wrapper.cpp"), "./")
