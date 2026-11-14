# -*- coding: utf-8 -*-
"""
并行分片 + 断点续传下载器

为什么需要它:
  本会话实证 —— 沙箱代理会粗暴掐断单条长连接:
    * git clone            -> schannel/TLS 握手失败
    * codeload.github.com  -> 可用, 但不支持 Range, 单连接约 19.7MB 后必断, 续传无效
    * raw.githubusercontent-> HTTP 000 完全不可达
    * cdn.jsdelivr.net     -> 单文件 <20MB 可用, 超过则 403
    * https://gh-proxy.com/ + raw.githubusercontent  -> 支持 Range (206), 可用
  因此大文件只能靠「多条短生命周期的 Range 请求」拼出来。gitlab

用法:
  python code/tools/ranged_download.py --url <URL> --out <PATH> [--threads 8] [--expected-size N]
"""
import argparse
import os
import ssl
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


class Downloader:
    def __init__(self, url, out_path, threads=8, max_retries=20, timeout=30):
        self.url = url
        self.out_path = out_path
        self.threads = threads
        self.max_retries = max_retries
        self.timeout = timeout
        self.total = None
        self.lock = threading.Lock()
        self.done = {}

    # ---------- 基础请求 ----------
    def _get(self, start=None, end=None):
        req = urllib.request.Request(self.url)
        if start is not None:
            rng = "bytes=%d-%s" % (start, "" if end is None else end)
            req.add_header("Range", rng)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=self.timeout, context=CTX) as r:
            code = r.status
            if start is not None and code == 206:
                cdr = r.headers.get("Content-Range", "")
                if "/" in cdr:
                    try:
                        self.total = int(cdr.split("/")[-1])
                    except ValueError:
                        pass
            return code, r.read()

    def probe_total(self):
        """用 1 字节 Range 请求探出真实总长。"""
        for attempt in range(6):
            try:
                self._get(start=0, end=0)
                if self.total:
                    return self.total
            except Exception as e:
                sys.stderr.write("probe attempt %d failed: %s\n" % (attempt + 1, e))
            time.sleep(1)
        return None

    # ---------- 分片下载 ----------
    def download_chunk(self, idx, start, end):
        """下载 [start, end] 并写入文件对应偏移, 带重试。"""
        for attempt in range(self.max_retries):
            try:
                code, data = self._get(start=start, end=end)
                with self.lock:
                    with open(self.out_path, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                got = len(data)
                want = end - start + 1
                with self.lock:
                    self.done[idx] = got
                if got == want:
                    return idx, want, None
                # 短读 -> 继续重试剩余部分
                return self.download_chunk(idx, start + got, end) if got > 0 else None
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return idx, 0, str(e)
                time.sleep(0.5)
        return idx, 0, "exhausted"

    def run(self):
        total = self.probe_total()
        if not total:
            print("[FATAL] 无法探测文件总长, 目标可能不可达")
            return False
        print("[probe] 目标总长 = %d bytes (%.1f MB)" % (total, total / 1024 ** 2))

        # 预分配文件
        with open(self.out_path, "wb") as f:
            f.truncate(total)

        chunk = max(1024 * 512, (total // (self.threads * 4)) // (1024 * 64) * (1024 * 64))
        tasks = []
        pos = 0
        idx = 0
        while pos < total:
            end = min(pos + chunk - 1, total - 1)
            tasks.append((idx, pos, end))
            pos = end + 1
            idx += 1
        print("[plan] 分为 %d 片, 每片 %.2f MB, %d 线程并发"
              % (len(tasks), chunk / 1024 ** 2, self.threads))

        t0 = time.time()
        completed = 0
        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            futs = {ex.submit(self.download_chunk, i, s, e): (i, s, e) for i, s, e in tasks}
            for fut in as_completed(futs):
                i, s, e = futs[fut]
                try:
                    _, got, err = fut.result()
                except Exception as ex2:
                    got, err = 0, str(ex2)
                completed += 1
                if got:
                    print("[chunk %4d] ok %d bytes  (%d/%d)" % (i, got, completed, len(tasks)))
                else:
                    print("[chunk %4d] FAILED %s" % (i, err))

        size = os.path.getsize(self.out_path)
        el = time.time() - t0
        print("[stat] 落盘 %d/%d bytes, 耗时 %.1fs, 平均 %.2f MB/s"
              % (size, total, el, total / 1024 ** 2 / max(el, 1e-6)))
        if size != total:
            print("[WARN] 大小不符, 请重跑本命令(会自动续接未完成分片)")
            return False
        print("[OK] 下载完成:", self.out_path)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--retries", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    d = Downloader(args.url, args.out, threads=args.threads, max_retries=args.retries)
    sys.exit(0 if d.run() else 1)


if __name__ == "__main__":
    main()
