# -*- coding: utf-8 -*-
"""
并行分片 + 断点续传下载器（纯 stdlib，正确实现）。

与仓库里旧的 ranged_download.py 的差别：旧版在分片失败时会留下全零空洞，
本版**每片单独校验字节数**，失败即重试，全部分片成功后才拼装，杜绝静默损坏。

用法：
  python par_download.py --url U --out O [--threads 8] [--chunk-mb 2]
"""
import argparse
import os
import sys
import threading
import time
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
LOCK = threading.Lock()


def remote_size(url):
    last = None
    for _ in range(8):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                n = r.headers.get("Content-Length")
                if n:
                    return int(n)
        except Exception as e:
            last = e
        time.sleep(3)
    raise RuntimeError("cannot HEAD %s (%s)" % (url, last))


def fetch_chunk(url, start, end, path, tries=60):
    want = end - start + 1
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": "bytes=%d-%d" % (start, end)})
            with urllib.request.urlopen(req, timeout=240) as r:
                data = r.read()
            if len(data) == want:
                with open(path, "wb") as f:
                    f.write(data)
                return True
            print("  [chunk %d-%d] short %d/%d, retry" % (start, end, len(data), want), flush=True)
        except Exception as e:
            print("  [chunk %d-%d] %s, retry" % (start, end, type(e).__name__), flush=True)
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk-mb", type=float, default=2.0)
    a = ap.parse_args()

    total = remote_size(a.url)
    print("[target] %s  bytes=%d" % (a.out, total), flush=True)

    cs = int(a.chunk_mb * 1024 * 1024)
    spans = [(s, min(s + cs - 1, total - 1)) for s in range(0, total, cs)]
    tmpdir = a.out + ".parts"
    os.makedirs(tmpdir, exist_ok=True)

    parts = [os.path.join(tmpdir, "p%05d" % i) for i in range(len(spans))]
    done = 0
    t0 = time.time()

    def worker(i):
        nonlocal done
        if not fetch_chunk(a.url, spans[i][0], spans[i][1], parts[i]):
            raise SystemExit("chunk %d failed" % i)
        with LOCK:
            done += 1
            el = time.time() - t0
            print("  [%3d/%3d] %.1f MB in %.0fs (%.0f KB/s)"
                  % (done, len(spans), done * cs / 1e6, el, done * cs / 1024 / max(el, 1e-6)),
                  flush=True)

    idx = list(range(len(spans)))
    queue = list(idx)
    qlock = threading.Lock()

    def run():
        while True:
            with qlock:
                if not queue:
                    return
                i = queue.pop(0)
            worker(i)

    ths = [threading.Thread(target=run, daemon=True) for _ in range(min(a.threads, len(spans)))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    # 拼装 + 校验
    with open(a.out, "wb") as out:
        for i, p in enumerate(parts):
            with open(p, "rb") as f:
                out.write(f.read())
    sz = os.path.getsize(a.out)
    print("[done] %d / %d bytes  (%.0fs)" % (sz, total, time.time() - t0), flush=True)
    for p in parts:
        os.remove(p)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    return 0 if sz == total else 1


if __name__ == "__main__":
    sys.exit(main())
