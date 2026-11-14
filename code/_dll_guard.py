#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vcomp140.dll 守卫：torch CUDA 在本机必须隐藏该 DLL 才不 segfault（见项目 MEMORY）。

必须在 `import torch` 之前调用 hide()；atexit 自动还原。
仅在 MainProcess 生效（DataLoader worker 不得重复改名）。

⚠ 健壮性（2026-09-30 修复）：
   System32 下可能残留 0 字节的 `vcomp140.dll.hide` 占位（上一轮崩溃遗留，Windows 权限
   拒绝删除）。旧实现 `rename(.dll -> .hide)` 因目标已存在而失败 -> DLL 保持可见 ->
   torch CUDA 双 OpenMP segfault（上一轮 80-epoch 训练因此跑满 1 epoch 后被回收）。
   现改为隐藏到不冲突的 `.guard` 后缀，并 best-effort 清理 stale `.hide` / 已存在的
   `.guard`（失败则忽略，改用带时间戳后缀兜底）。
"""
import atexit
import multiprocessing as _mp
import os
import time

_DLL = r"C:\Windows\System32\vcomp140.dll"
_hidden_path = None          # 实际隐藏到的目标路径（供 restore 使用）
_hidden_by_us = False


def _best_effort_cleanup_stale():
    # 清理上一轮残留的 0 字节 .hide 占位（若可删）；删不掉也无害，我们本就隐藏到 .guard
    stale = _DLL + ".hide"
    try:
        if os.path.exists(stale) and os.path.getsize(stale) == 0:
            os.remove(stale)
    except Exception:
        pass


def hide():
    global _hidden_path, _hidden_by_us
    if _mp.current_process().name != "MainProcess":
        return False
    _best_effort_cleanup_stale()
    if not os.path.exists(_DLL):
        # 已被隐藏（或本就不存在）
        return True

    # 优先隐藏到 .guard；若已存在先尝试删除，删不掉则改用时间戳后缀兜底
    cand = _DLL + ".guard"
    if os.path.exists(cand):
        try:
            os.remove(cand)
        except Exception:
            cand = _DLL + (".guard_%d" % int(time.time() * 1000))
    try:
        os.rename(_DLL, cand)
        _hidden_path = cand
        _hidden_by_us = True
        atexit.register(restore)
        return True
    except Exception:
        print("[warn] 无法隐藏 vcomp140.dll，torch CUDA 可能 segfault")
        return False


def restore():
    global _hidden_path, _hidden_by_us
    try:
        if _hidden_by_us and _hidden_path and os.path.exists(_hidden_path) \
                and not os.path.exists(_DLL):
            os.rename(_hidden_path, _DLL)
            _hidden_path = None
            _hidden_by_us = False
    except Exception:
        pass
