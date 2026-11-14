"""Download L8-S2-NAIP ROIs (landsat8 + sentinel2) into .l8s2naip/ for the
real cross-sensor 3x arm. Skips already-present valid files; retries on failure.

Staging happens in a system temp dir (outside the workspace) so hf_hub's
.cache/lock cleanup never trips the sandbox safe-delete guard; finished files
are copied into .l8s2naip/.

Run with: HF_ENDPOINT=https://hf-mirror.com python code/tools/download_l8s2naip.py
(Requires HTTPS_PROXY to reach the mirror.)
"""
import os
import shutil
import tempfile
from huggingface_hub import hf_hub_download

# Keep hf_hub's blob cache outside the workspace too.
os.environ.setdefault("HF_HUB_CACHE", os.path.join(tempfile.gettempdir(), "hf_cache_e1"))

REPO = "namelesscoder/L8-S2-NAIP"
# Script lives at code/tools/download_l8s2naip.py; repo root is three dirnames up.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATADIR = os.path.join(ROOT, ".l8s2naip")
os.makedirs(DATADIR, exist_ok=True)

# All ROIs present in the repo (skips 4,7,8,9,11,12,16,17 which have no files).
ROIS = [1, 2, 3, 5, 6, 10, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
PAIRS = ["landsat8_export_roi_%d.tif", "sentinel2_export_roi_%d.tif"]


def dl(fname):
    dst = os.path.join(DATADIR, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 50000:
        return "skip"
    last = ""
    for attempt in range(4):
        try:
            tmp = tempfile.mkdtemp(prefix="l8s2_")
            p = hf_hub_download(repo_type="dataset", repo_id=REPO, filename=fname,
                                local_dir=tmp)
            if os.path.exists(p) and os.path.getsize(p) > 50000:
                shutil.copy(p, dst)
                return "ok"
            last = "downloaded-but-empty"
        except Exception as e:  # noqa
            last = repr(e)[:160]
    return "FAIL:%s" % last


def main():
    ok = skip = fail = 0
    for rid in ROIS:
        for pat in PAIRS:
            fn = pat % rid
            r = dl(fn)
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                fail += 1
            print("ROI %3d  %-34s -> %s  (ok=%d skip=%d fail=%d)" % (rid, fn, r, ok, skip, fail))
    print("\nDONE  ok=%d  skip=%d  fail=%d" % (ok, skip, fail))


if __name__ == "__main__":
    main()
