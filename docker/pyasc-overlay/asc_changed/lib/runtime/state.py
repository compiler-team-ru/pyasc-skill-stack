# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import atexit
import ctypes
import os
import shutil
import tempfile
import hashlib
from typing import Dict, NoReturn, Optional
from pathlib import Path
import sysconfig

from asc.runtime.cache import get_cache_manager
from asc.runtime import config
from .build_utils import build_npu_ext
from ..utils import get_ascend_path


class RuntimeInterface:

    def __init__(self, is_model: bool, soc: config.Platform) -> None:
        dirname = os.path.dirname(os.path.realpath(__file__))
        wrapper_name = "rt_wrapper"
        src = Path(os.path.join(dirname, f"{wrapper_name}.cpp")).read_text()

        suffix_key = ""
        version_cfg = get_ascend_path() / "version.cfg"
        if version_cfg.exists():
            suffix_key += version_cfg.read_text()
        key = hashlib.sha256((src + suffix_key).encode("utf-8")).hexdigest()
        cache_manager = get_cache_manager(key)
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        rt_lib = cache_manager.get_file(f"lib{wrapper_name}{suffix}")

        if rt_lib is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, f"{wrapper_name}.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_npu_ext(wrapper_name, is_model, soc, src_path, tmpdir)
                with open(so, "rb") as f:
                    rt_lib = cache_manager.put(f.read(), f"lib{wrapper_name}{suffix}", binary=True)

        self.lib: ctypes.CDLL = ctypes.CDLL(rt_lib, ctypes.RTLD_GLOBAL)
        self.model_log_path: Optional[str] = None
        if is_model:
            log_path_str = "CAMODEL_LOG_PATH"
            model_log_path = os.environ.get(log_path_str, "").strip()
            if not model_log_path:
                model_log_path = tempfile.mkdtemp(prefix="pyasc_camodel_")
                atexit.register(shutil.rmtree, model_log_path)
            model_log_path = os.path.abspath(model_log_path)
            os.environ[log_path_str] = model_log_path
            os.makedirs(model_log_path, exist_ok=True)
            self.model_log_path = model_log_path

    @staticmethod
    def check_error(error: int, fn_name: str) -> None:
        if error != 0:
            raise RuntimeError(f"Function {fn_name} returned {error}")

    @classmethod
    def call_from(cls, rt_lib: ctypes.CDLL, fn_name: str, *args) -> None:
        fn = getattr(rt_lib, fn_name)
        fn.restype = ctypes.c_uint64
        cls.check_error(fn(*args), fn_name)

    def call(self, fn_name: str, *args) -> None:
        self.call_from(self.lib, fn_name, *args)


class NPUUtils:

    def __new__(cls, is_model: bool, soc: config.Platform):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self, is_model: bool, soc: config.Platform):
        if is_model:
            return
        dirname = os.path.dirname(os.path.realpath(__file__))
        utils_name = "npu_utils"
        src = Path(os.path.join(dirname, f"{utils_name}.cpp")).read_text()

        suffix_key = ""
        version_cfg = get_ascend_path() / "version.cfg"
        if version_cfg.exists():
            suffix_key += version_cfg.read_text()
        key = hashlib.sha256((src + suffix_key).encode("utf-8")).hexdigest()
        cache_manager = get_cache_manager(key)
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        utils_lib = cache_manager.get_file(f"lib{utils_name}{suffix}")

        if utils_lib is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, f"{utils_name}.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_npu_ext(utils_name, is_model, soc, src_path, tmpdir)
                with open(so, "rb") as f:
                    utils_lib = cache_manager.put(f.read(), f"lib{utils_name}{suffix}", binary=True)

        import importlib.util
        spec = importlib.util.spec_from_file_location(utils_name, utils_lib)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod

        self.acl_init = self.npu_utils_mod.acl_init
        self.acl_finalize = self.npu_utils_mod.acl_finalize
        self.msprof_sys_cycle_time = self.npu_utils_mod.msprof_sys_cycle_time
        self.msprof_report_api = self.npu_utils_mod.msprof_report_api
        self.msprof_report_compact_info = self.npu_utils_mod.msprof_report_compact_info
        self.msprof_report_additional_info = self.npu_utils_mod.msprof_report_additional_info

    def __getattr__(self, name: str) -> NoReturn:
        if model:
            raise RuntimeError(f"{self.__class__.__name__} properties are not available when Model backend is active")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


model: bool = False
soc_verison: config.Platform = config.Platform.Ascend910B1
custom_lib_prefix: Optional[str] = None
device_id: Optional[int] = None
streams: Dict[int, Optional[ctypes.c_void_p]] = {}
kernels: Dict[int, bytes] = {}
allocs: Dict[int, int] = {}
lib: Optional[RuntimeInterface] = None
npu_utils: Optional[NPUUtils] = None


def load_lib() -> None:
    global lib
    lib = RuntimeInterface(is_model=model, soc=soc_verison)
    global npu_utils
    npu_utils = NPUUtils(is_model=model, soc=soc_verison)
