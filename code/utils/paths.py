"""路径解析：让发布版产物（相对路径）与本地开发版（绝对路径）都能跑。

发布到 GitHub 的 WorldStrat 清单**不含本机绝对路径**（避免泄漏用户名，也便于他人
复现），只存相对数据根的路径，例如：

    "Landcover-1109896/L2A/Landcover-1109896-1-L2A_data.tiff"

数据根由环境变量 `WORLDSTRAT_ROOT` 指定；未设置时退回 `~/worldstrat`。
本地开发用的清单本来就是绝对路径，`resolve()` 会原样放行，行为不变。
"""
from __future__ import annotations

import os

ENV_WORLDSTRAT = "WORLDSTRAT_ROOT"


def worldstrat_root(default: str | None = None) -> str:
    """数据根：显式入参 > 环境变量 > `~/worldstrat`。"""
    if default:
        return default
    env = os.environ.get(ENV_WORLDSTRAT)
    if env:
        return env
    return os.path.expanduser(os.path.join("~", "worldstrat"))


def resolve(path, root: str | None = None):
    """相对路径按数据根补全；绝对路径与 None 原样返回。"""
    if not isinstance(path, str):
        return path
    if os.path.isabs(path):
        return path
    sep_aware = path.replace("\\", os.sep).replace("/", os.sep)
    return os.path.join(worldstrat_root(root), sep_aware)


def rebase_manifest(manifest, root: str | None = None):
    """把清单里每条 `lr`/`hr` 的相对路径按数据根补全（原地语义，返回新列表）。"""
    if not manifest:
        return manifest
    out = []
    for e in manifest:
        if not isinstance(e, dict):
            out.append(e)
            continue
        e = dict(e)
        for k in ("lr", "hr"):
            v = e.get(k)
            if isinstance(v, (list, tuple)):
                e[k] = [resolve(p, root) for p in v]
            else:
                e[k] = resolve(v, root)
        out.append(e)
    return out
