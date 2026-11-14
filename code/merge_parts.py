#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线合并 fetch_external.py 下载的分段（不联网）。

用途：下载全部完成后合并被中断（如任务被 kill）时，不必重下，
直接把 <out>.seg{i} 按序拼回 <out> 并校验 size / md5。

用法:
  python code/merge_parts.py --out data/external/probav/probav_data.zip \
      --size 708416603 --n 10 --md5 7101101c7c91eeb5979d472861c1b784
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--md5", default="")
    ap.add_argument("--keep", action="store_true", help="合并后保留分段（默认改名 .done）")
    args = ap.parse_args()

    total, n = args.size, args.n
    step = -(-total // n)
    bounds = []
    for i in range(n):
        s = i * step
        if s >= total:
            break
        bounds.append((i, s, min(s + step - 1, total - 1)))

    # 1) 逐段校验长度
    ok = True
    for i, s, e in bounds:
        sp = f"{args.out}.seg{i}"
        exp = e - s + 1
        sz = os.path.getsize(sp) if os.path.exists(sp) else -1
        if sz != exp:
            ok = False
            print(f"[FAIL] seg{i}: {sz} != {exp}")
    if not ok:
        print("分段不完整，请用 fetch_external.py 续传后重试")
        sys.exit(1)
    print(f"[ok] {len(bounds)} 段长度全部符合，合计 {total/1e6:.1f} MB")

    # 2) 合并（旧的半成品用 replace 挪开，避免撞沙箱删除配额）
    if os.path.exists(args.out):
        os.replace(args.out, args.out + ".partial")
    with open(args.out, "wb") as fo:
        for i, _, _ in bounds:
            sp = f"{args.out}.seg{i}"
            with open(sp, "rb") as fi:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
            if not args.keep:
                try:
                    os.replace(sp, sp + ".done")
                except OSError:
                    pass
            print(f"  + seg{i}", flush=True)

    size = os.path.getsize(args.out)
    print(f"[done] {args.out}  {size/1e6:.1f} MB")
    if size != total:
        print(f"[WARN] 大小不符 expect={total} got={size}")
        sys.exit(1)
    if args.md5:
        m = md5_of(args.out)
        good = (m == args.md5)
        print(f"[md5] {m}  {'OK' if good else 'MISMATCH expect=' + args.md5}")
        sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
