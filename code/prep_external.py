#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解压并体检外部数据集（PROBA-V / SEN2VENµS）。

用法:
  python code/prep_external.py probav     # 解压 + 结构报告
  python code/prep_external.py sen2venus
  python code/prep_external.py all --check  # 只体检不解压
"""
from __future__ import annotations

import os
import sys
import zipfile
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(BASE, "data", "external")


def unzip(zpath: str, dest: str, members=None) -> int:
    os.makedirs(dest, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for m in (members or z.namelist()):
            z.extract(m, dest)
            n += 1
            if n % 2000 == 0:
                print(f"  ...{n}", flush=True)
    return n


def report_probav():
    root = os.path.join(EXT, "probav", "probav_data")
    if not os.path.isdir(root):
        root = os.path.join(EXT, "probav")
    print(f"[probav] root={root}")
    # 自动定位含 train/RED 的目录
    cand = None
    for dp, dns, fns in os.walk(root):
        if os.path.basename(dp) == "RED" and os.path.basename(os.path.dirname(dp)) == "train":
            cand = os.path.dirname(os.path.dirname(dp))
            break
    if cand is None:
        print("  [!] 未找到 train/RED 结构")
        return None
    print(f"  data root = {cand}")
    for split in ("train", "test"):
        for band in ("RED", "NIR"):
            d = os.path.join(cand, split, band)
            if os.path.isdir(d):
                print(f"  {split}/{band}: {len(os.listdir(d))} scenes")
    d0 = os.path.join(cand, "train", "RED", sorted(os.listdir(os.path.join(cand, "train", "RED")))[0])
    print("  sample scene files:", sorted(os.listdir(d0))[:8])
    ext = Counter(os.path.splitext(f)[1] for f in os.listdir(d0))
    print("  ext:", dict(ext))
    return cand


def report_sen2venus():
    root = os.path.join(EXT, "sen2venus")
    print(f"[sen2venus] root={root}")
    zs = [f for f in os.listdir(root) if f.endswith(".zip")]
    print(f"  site zips: {zs}")
    if not zs:
        return None
    z0 = os.path.join(root, zs[0])
    with zipfile.ZipFile(z0) as z:
        names = z.namelist()
        print(f"  {zs[0]}: {len(names)} entries")
        for n in names[:12]:
            print("   ", n)
    return z0


def main():
    args = sys.argv[1:] or ["all"]
    check = "--check" in args
    do = set(args) - {"--check"}
    if "all" in do or "probav" in do:
        zp = os.path.join(EXT, "probav", "probav_data.zip")
        if os.path.exists(zp) and not check:
            print(f"[unzip] {zp}")
            n = unzip(zp, os.path.join(EXT, "probav"))
            print(f"  extracted {n} entries")
        report_probav()
    if "all" in do or "sen2venus" in do:
        if not check:
            d = os.path.join(EXT, "sen2venus")
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if f.endswith(".zip"):
                    zp = os.path.join(d, f)
                    print(f"[unzip] {zp}")
                    # 站点 zip 内部第一层就是 {SITE}/，故解压到 d 而不是 d/{SITE}，
                    # 否则会多一层（d/{SITE}/{SITE}/）导致加载器找不到二级 zip。
                    n = unzip(zp, d)
                    print(f"  extracted {n} entries")
        report_sen2venus()


if __name__ == "__main__":
    main()
