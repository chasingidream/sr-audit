"""Build a WorldStrat manifest JSON from the real data layout.

WorldStrat ROIs each contain Sentinel-2 (LR, multi-date) and SPOT/HR images.
This pairs them by coverage/ROI id so WorldStratDataset can load them.

Two layouts are supported:
  --layout zenodo  (default): the official Zenodo packages
      <root>/<cov>/<cov>_ps.tiff                  # SPOT pansharpened HR
      <root>/<cov>/L2A/<cov>-<date>-L2A_data.tiff # Sentinel-2 L2A LR (multi-date)
  --layout generic: the reorganized high_res/ + low_res/Sentinel2/ layout.

Usage:
    python prep_worldstrat.py --root <unpacked_root> \
        --out data/processed/worldstrat_manifest.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import build_manifest_worldstrat, build_manifest_worldstrat_zenodo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/processed/worldstrat_manifest.json")
    ap.add_argument("--layout", default="zenodo", choices=("zenodo", "generic"))
    args = ap.parse_args()

    builder = (build_manifest_worldstrat_zenodo
               if args.layout == "zenodo" else build_manifest_worldstrat)
    manifest = builder(args.root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prep] wrote {len(manifest)} LR-HR pairs -> {args.out}")


if __name__ == "__main__":
    main()
