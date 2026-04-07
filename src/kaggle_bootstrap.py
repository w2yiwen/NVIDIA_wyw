"""
Optional Kaggle-only setup: offline pip target, .pth path injection, Triton/ptxas hacks.

Import and call `bootstrap()` at the start of a Kaggle notebook if you use the same
layout as the official offline bundle and utility script paths.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _resolve_python_path(target_dir: str | Path) -> None:
    for pth_file in Path(target_dir).glob("*.pth"):
        relpath = pth_file.read_text().strip()
        rel_pack_path = pth_file.parent / relpath
        if rel_pack_path.exists():
            sys.path.append(str(rel_pack_path))


def install_offline_packages(
    offline_dir: str = "/kaggle/input/nvidia-nemotron-offline-packages/offline_packages",
    target_dir: str = "/kaggle/working/packages",
    utility_pth_dir: str = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/",
) -> None:
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    if Path(utility_pth_dir).exists():
        _resolve_python_path(utility_pth_dir)

    if not Path(offline_dir).exists():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--target",
                target_dir,
                "datasets",
                "trl",
            ]
        )
        print("Installed datasets, trl from PyPI (offline bundle missing).")
        return

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-index",
            "--find-links",
            offline_dir,
            "--target",
            target_dir,
            "datasets",
            "trl",
        ]
    )
    print("Installed from offline packages.")

    sys.path.append(target_dir)
    _resolve_python_path(target_dir)


def apply_triton_ptxas_hack(
    ptxas_src: str = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell",
) -> None:
    """Copy ptxas and repoint Triton NVIDIA bin — only needed on Kaggle Blackwell stack."""

    try:
        import triton.backends.nvidia as nv_backend
    except ImportError:
        print("Triton not available; skipping ptxas hack.")
        return

    def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=True, upcast=True):
        import torch
        import torch.nn.functional as F

        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * F.silu(z.float())
        return out.to(dtype)

    for name, mod in list(sys.modules.items()):
        if hasattr(mod, "rmsnorm_fn"):
            mod.rmsnorm_fn = _pure_rmsnorm_fn

    dst = "/tmp/ptxas-blackwell"
    if os.path.exists(ptxas_src):
        shutil.copy2(ptxas_src, dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
        dst_bin = "/tmp/triton_nvidia_bin"
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for f in os.listdir(dst_bin):
            fp = os.path.join(dst_bin, f)
            if os.path.isfile(fp):
                os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
        os.environ["TRITON_PTXAS_PATH"] = dst
        print("Triton ptxas fix applied.")


def patch_nemotron_fast_path_disabled() -> None:
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name:
            mod.is_fast_path_available = False
            print(f"Patched {name}: is_fast_path_available = False")


def patch_triton_compiler_version() -> None:
    try:
        import triton.backends.nvidia.compiler as nv_compiler
    except ImportError:
        return
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
    nv_compiler.get_ptxas_version = lambda arch: "12.0"


def bootstrap(
    offline: bool = True,
    triton_hack: bool = True,
) -> None:
    if offline:
        install_offline_packages()
    if triton_hack and os.path.exists(
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
    ):
        apply_triton_ptxas_hack()
