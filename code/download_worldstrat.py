"""Download / prepare the real WorldStrat paired dataset and build its manifest.

WorldStrat (NeurIPS'22) provides, per coverage / AOI:
    high_res/<hr>.tif            SPOT HR  (~1.5 m, RGB)
    low_res/Sentinel2/<date>.tif Sentinel-2 LR (multi-date)
See https://zenodo.org/record/6810792 and https://www.kaggle.com/datasets/jucor1/worldstrat.

Acquisition (this environment):
  * Zenodo  -> 403 Forbidden from here (blocked).
  * Kaggle  -> reachable, but needs an API token (~/.kaggle/kaggle.json).
  * So this script prefers Kaggle when a token is present, else prints exact
    manual steps. After the data is unpacked under --root, run:
        python prep_worldstrat.py --root <root> --out data/processed/worldstrat_manifest.json

Usage:
    python download_worldstrat.py --root data/raw/worldstrat --out data/processed/worldstrat_manifest.json
"""
import os
import sys
import json
import shutil
import argparse
import urllib.request
import ssl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import build_manifest_worldstrat

KAGGLE_DATASET = "jucor1/worldstrat"   # official WorldStrat mirror on Kaggle
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def _has_kaggle_token():
    cand = [
        os.path.join(os.environ.get("USERPROFILE", ""), ".kaggle", "kaggle.json"),
        os.path.expanduser("~/.kaggle/kaggle.json"),
        os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", ""), "kaggle.json"),
    ]
    return any(os.path.exists(c) for c in cand if c)


def _kaggle_cli_available():
    return shutil.which("kaggle") is not None


def download_via_kaggle(root):
    import subprocess
    os.makedirs(root, exist_ok=True)
    print(f"[download] using Kaggle CLI to fetch '{KAGGLE_DATASET}' -> {root}")
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
           "-p", root, "--unzip"]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Kaggle may nest under an extra folder; flatten one level if needed
    _flatten_single_subdir(root)
    print("[download] done. Now build the manifest with prep_worldstrat.py.")


def _flatten_single_subdir(root):
    entries = [e for e in os.listdir(root)
               if os.path.isdir(os.path.join(root, e))]
    if len(entries) == 1:
        only = os.path.join(root, entries[0])
        # only flatten if no coverage folders directly under root
        inner = os.listdir(only)
        if not any(os.path.isdir(os.path.join(root, e)) for e in inner):
            for f in inner:
                shutil.move(os.path.join(only, f), os.path.join(root, f))
            os.rmdir(only)


def build_manifest(root, out):
    manifest = build_manifest_worldstrat(root)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prep] wrote {len(manifest)} LR-HR pairs -> {out}")
    if not manifest:
        print("[prep] WARNING: 0 pairs. Check that <root> contains "
              "<cov>/high_res/*.tif and <cov>/low_res/Sentinel2/*.tif.")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/worldstrat")
    ap.add_argument("--out", default="data/processed/worldstrat_manifest.json")
    ap.add_argument("--no-download", action="store_true",
                    help="skip download; only (re)build the manifest from --root")
    args = ap.parse_args()

    if not args.no_download:
        if _has_kaggle_token() and _kaggle_cli_available():
            try:
                download_via_kaggle(args.root)
            except Exception as e:
                print(f"[download] Kaggle path failed: {e}")
                print(_manual_steps())
        else:
            print(_manual_steps())
            if not os.path.isdir(args.root) or not any(
                    os.scandir(args.root)):
                print(f"[abort] no data at {args.root!r}; cannot build manifest. "
                      f"Provide data or a Kaggle token first.")
                return
    build_manifest(args.root, args.out)


def _manual_steps():
    return (
        "\n=== How to get WorldStrat real paired data ===\n"
        "Option A (Kaggle, recommended here):\n"
        "  1. Get a Kaggle API token: Kaggle.com -> Account -> 'Create New Token'\n"
        "     -> save the JSON as  ~/.kaggle/kaggle.json\n"
        "  2. Install CLI:  pip install kaggle\n"
        "  3. Re-run:  python download_worldstrat.py --root data/raw/worldstrat\n"
        "Option B (Zenodo, if reachable from your network):\n"
        "  Download https://zenodo.org/record/6810792 and unpack into data/raw/worldstrat/\n"
        "  so each coverage has  high_res/*.tif  and  low_res/Sentinel2/*.tif .\n"
        "Then: python prep_worldstrat.py --root data/raw/worldstrat "
        "--out data/processed/worldstrat_manifest.json\n"
    )


if __name__ == "__main__":
    main()
