#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量 CRC 扫描 zip 成员，把损坏成员的文件偏移映射回下载分段号。

用途：分段下载器在"服务端忽略 Range 返回 200 全量"时可能写入错位字节，
整体 md5 会对不上但 zip 结构仍完好。用每个成员的 CRC32 定位坏数据，
换算成 <out>.seg{i} 的分段号，只重下有问题的段，避免整包重下。

用法:
  python code/scan_zip_bad.py --zip data/external/probav/probav_data.zip \
      --size 708416603 --n 10
"""
from __future__ import annotations

import argparse
import os
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="最多校验多少个成员（0=全部）")
    args = ap.parse_args()

    step = -(-args.size // args.n)
    z = zipfile.ZipFile(args.zip)
    infos = [i for i in z.infolist() if not i.is_dir()]
    if args.limit:
        infos = infos[: args.limit]

    bad_members, bad_segs, offsets = [], set(), {}
    for k, info in enumerate(infos):
        try:
            with z.open(info) as f:
                while f.read(1 << 20):
                    pass
        except Exception as e:
            bad_members.append((info.filename, str(e)[:60]))
            s = info.header_offset // step
            bad_segs.add(int(min(s, args.n - 1)))
            offsets.setdefault(int(min(s, args.n - 1)), info.header_offset)
        if (k + 1) % 5000 == 0:
            print(f"  ...{k+1}/{len(infos)} 已检，坏段 {sorted(bad_segs)}", flush=True)

    print(f"\n[scan] 成员 {len(infos)}，损坏 {len(bad_members)}")
    for name, err in bad_members[:20]:
        print(f"   BAD {name}  ({err})")
    print(f"[result] 坏分段号 = {sorted(bad_segs)}")
    for s in sorted(bad_segs):
        print(f"   seg{s}: 首个坏偏移 {offsets[s]} (段内 {offsets[s]-s*step})")


if __name__ == "__main__":
    main()
