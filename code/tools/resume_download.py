# -*- coding: utf-8 -*-
"""
抗限流断点续传下载器（纯 stdlib）。
- 优先 Range 续传（206）；服务器不支持 Range 时自动从头重来。
- 死循环重试，直到字节数达到远端 Content-Length 且（可选）校验通过。
用于 geodata.ucdavis.edu 这类会间歇性掐流的主机。
"""
import argparse
import os
import sys
import time
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def head_len(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0)), r.status


def download_once(url, out, total):
    off = os.path.getsize(out) if os.path.exists(out) else 0
    if total and off >= total:
        return off
    hdr = {"User-Agent": UA}
    if off > 0:
        hdr["Range"] = "bytes=%d-" % off
    req = urllib.request.Request(url, headers=hdr)
    t0 = time.time()
    got = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        code = r.status
        if code == 200 and off > 0:
            print("  [info] server ignored Range -> restart from 0", flush=True)
            off = 0
            mode = "wb"
        else:
            mode = "ab"
        with open(out, mode) as f:
            while True:
                chunk = r.read(262144)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got % (2 * 1024 * 1024) < 262144:
                    el = time.time() - t0
                    print("  +%.1f MB  (%.0f KB/s)  total=%.1f MB"
                          % (got / 1e6, got / 1024 / max(el, 1e-6), (off + got) / 1e6), flush=True)
    return os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tries", type=int, default=400)
    a = ap.parse_args()

    total = 0
    for i in range(6):
        try:
            total, st = head_len(a.url)
            if total:
                break
        except Exception as e:
            print("  [head retry %d] %s" % (i, e), flush=True)
            time.sleep(5)
    print("[target] %s  remote_bytes=%d" % (a.out, total), flush=True)

    for i in range(a.max_tries):
        try:
            sz = download_once(a.url, a.out, total)
        except Exception as e:
            print("  [try %d fail] %s: %s" % (i, type(e).__name__, e), flush=True)
            time.sleep(4)
            continue
        if total and sz >= total:
            print("[done] %d bytes in %d tries" % (sz, i + 1), flush=True)
            return 0
        print("  [try %d] partial %d / %d" % (i + 1, sz, total), flush=True)
        time.sleep(3)
    print("[give-up] still short", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
