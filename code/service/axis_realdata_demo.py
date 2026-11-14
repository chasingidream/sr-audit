#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""axis_realdata_demo.py —— 用【真实数据】实测统一网关（把"三轴权重实测"落到服务路径）。

流程（对每个支持上传的轴）：
  1) 从该轴验证集取出第一个场景，把模型实际看到的 LR 张量反归一化回原始反射率
     （WorldStrat: x*sd+mu, scene_norm[i]；PROBA-V: x*sd+mu, global 统计量）
     -> 存成 PNG —— 这就是"前端上传的那张低分辨率图"；
  2) POST /sr?model=<axis>&compare=1   -> [bicubic | 模型SR] 并排 PNG（忠实反射率口径）；
  3) POST /sr_audit?model=<axis>       -> SR 图 base64 + 无 GT 预检徽章 P1–P4（JSON）；
  4) 三张真实审计卡（audit_axis_*.json）POST /assess -> 网关侧徽章/regime（JSON）。

说明：
  * SEN2VENµS 模型是 4 波段（B2/B3/B4/B8），RGB 上传无法表达 -> 只走评测路径（已有卡），
    不做 HTTP 出图演示（见 gateway /axes 的 http_upload=false）。
  * PROBA-V 是 RED 波段反射率（0.03–0.13），PNG 按忠实口径存会偏暗；
    另存 *_stretch.png（线性拉伸显示用，仅本地查看，非服务口径）。
  * 产物目录：experiments/service_axis_demo/（gitignore；影像不出库，符合 Tier-D 谨慎原则）。

用法：
    python code/service/axis_realdata_demo.py
"""
import base64
import io
import json
import os
import sys
import urllib.request
import uuid

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from data.dataset import get_dataset                    # noqa: E402
from eval_paired_floor import load_cfg                  # noqa: E402

BASE_URL = "http://127.0.0.1:8770"
OUT_DIR = os.path.join(REPO, "experiments", "service_axis_demo")
BOUNDARY = "----srdemoboundary" + uuid.uuid4().hex[:8]


# --------------------------------------------------------------------------- #
# HTTP 小工具
# --------------------------------------------------------------------------- #
def post_multipart(path, filename, payload):
    body = (("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
             "Content-Type: image/png\r\n\r\n" % (BOUNDARY, filename)).encode()
            + payload + ("\r\n--%s--\r\n" % BOUNDARY).encode())
    req = urllib.request.Request(BASE_URL + path, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=%s" % BOUNDARY)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.status, dict(r.headers), r.read()


def post_json(path, obj):
    req = urllib.request.Request(BASE_URL + path, method="POST",
                                 data=json.dumps(obj).encode("utf-8"))
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, dict(r.headers), r.read()


# --------------------------------------------------------------------------- #
# 取真实验证场景 -> 反归一化回原始反射率 PNG
# --------------------------------------------------------------------------- #
def first_val_lr_png(cfg_path, norm_stats, out_png):
    """cfg -> 验证集第一个场景的 frame0 LR -> 反归一化 -> uint8 PNG。

    norm_stats: {"mu": float, "sd": float}（global）或 {"scene_norm": path, "index_offset": i}
    返回 (raw 数组 [H,W,C] 0-1, 场景名)。
    """
    cfg = load_cfg(cfg_path)
    ds = get_dataset(cfg, project_root=REPO, deterministic=True)
    val_frac = float(cfg.get("val_frac", 0.1))
    n_train = max(1, min(len(ds) - 1, int((1.0 - val_frac) * len(ds))))
    item = ds[n_train]                                   # 验证集第一个场景
    lrs = item["lrs"].numpy()                            # [T,C,64,64] 归一化空间
    frame0 = lrs[0]
    name = item.get("name") or ("scene_%d" % n_train)

    if "mu" in norm_stats:                               # global 归一化
        raw = frame0 * norm_stats["sd"] + norm_stats["mu"]
    else:                                                # per_scene（预计算 scene_norm json）
        sn = json.load(open(norm_stats["scene_norm"]))
        mu, sd = sn[norm_stats.get("index_offset", n_train)]
        raw = frame0 * sd + mu
    raw = np.clip(raw, 0.0, 1.0)
    arr = np.transpose(raw, (1, 2, 0))                   # [H,W,C]
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    img = Image.fromarray((arr * 255.0).astype("uint8"),
                          mode="L" if arr.ndim == 2 else "RGB")
    img.save(out_png, format="PNG")
    print(f"  [export] {os.path.basename(out_png)}  ({arr.shape[1]}x{arr.shape[0]}x"
          f"{arr.shape[2] if arr.ndim == 3 else 1}, 反射率范围 "
          f"{float(arr.min()):.3f}–{float(arr.max()):.3f})  scene={name}")
    return out_png


def stretch_png(src_png, dst_png, gain=4.0):
    """线性增益显示副本（仅本地查看；服务口径保持忠实不拉伸）。"""
    img = Image.open(src_png)
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    arr = np.clip(arr * gain, 0, 255).astype("uint8")
    Image.fromarray(arr).save(dst_png, format="PNG")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("== 统一网关真实数据实测（Tier-B 末轮权重，经 HTTP 服务路径）==")

    # ---------- 1) WorldStrat 6.63× ----------
    print("\n[1] WorldStrat 6.63× (mtxa_aligned, per_scene)")
    ws_png = first_val_lr_png(
        os.path.join(REPO, "code", "configs", "external_worldstrat.py"),
        {"scene_norm": os.path.join(REPO, "data", "processed", "worldstrat_scene_norm.json")},
        os.path.join(OUT_DIR, "ws_input_lr.png"))
    st, hd, body = post_multipart("/sr?model=ws&compare=1", "ws_input_lr.png",
                                  open(ws_png, "rb").read())
    open(os.path.join(OUT_DIR, "ws_sr_compare.png"), "wb").write(body)
    print(f"  [/sr] {st}  {len(body)} bytes -> ws_sr_compare.png  meta={hd.get('X-SR-Meta', '')[:80]}")
    st, _, body = post_multipart("/sr_audit?model=ws&compare=1", "ws_input_lr.png",
                                 open(ws_png, "rb").read())
    obj = json.loads(body)
    json.dump(obj, open(os.path.join(OUT_DIR, "ws_sr_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  [/sr_audit] {st}  precheck: {obj['precheck']['headline']}")
    for k, v in obj["precheck"]["gates"].items():
        print(f"      {v['verdict']:<5} {k:<10} {v['detail']}")

    # ---------- 2) PROBA-V 3× ----------
    print("\n[2] PROBA-V 3× (mtxa_aligned n300_s42, global mu=0.08206 sd=0.07568)")
    pv_png = first_val_lr_png(
        os.path.join(REPO, "code", "configs", "external_probav_n300_s42.py"),
        {"mu": 0.08206, "sd": 0.07568},
        os.path.join(OUT_DIR, "probav_input_lr.png"))
    st, hd, body = post_multipart("/sr?model=probav&compare=1", "probav_input_lr.png",
                                  open(pv_png, "rb").read())
    open(os.path.join(OUT_DIR, "probav_sr_compare.png"), "wb").write(body)
    stretch_png(os.path.join(OUT_DIR, "probav_sr_compare.png"),
                os.path.join(OUT_DIR, "probav_sr_compare_stretch.png"), gain=4.0)
    print(f"  [/sr] {st}  {len(body)} bytes -> probav_sr_compare.png (+_stretch 显示用 ×4)")
    st, _, body = post_multipart("/sr_audit?model=probav&compare=1", "probav_input_lr.png",
                                 open(pv_png, "rb").read())
    obj = json.loads(body)
    json.dump(obj, open(os.path.join(OUT_DIR, "probav_sr_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  [/sr_audit] {st}  precheck: {obj['precheck']['headline']}")
    for k, v in obj["precheck"]["gates"].items():
        print(f"      {v['verdict']:<5} {k:<10} {v['detail']}")

    # ---------- 3) 三轴真实审计卡 -> /assess 徽章 ----------
    print("\n[3] 三轴审计卡 POST /assess（网关侧 derive 徽章）")
    cards = [
        ("experiments/audit_axis_probav_n300_s42.json", "probav"),
        ("experiments/audit_axis_s2v_sisr.json", "s2v"),
        ("experiments/audit_axis_ws_mtxa_aligned.json", "ws"),
    ]
    for rel, tag in cards:
        card = json.load(open(os.path.join(REPO, rel)))
        st, _, body = post_json("/assess", {"card": card})
        rec = json.loads(body)["cards"][0]
        json.dump(rec, open(os.path.join(OUT_DIR, "assess_%s.json" % tag), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  {tag:<7} {os.path.basename(rel)}\n          -> {rec['credibility']} | "
              f"{rec['regime']} | {rec['gate_verdict']}\n             {rec['headline'][:110]}")

    print("\n完成。产物目录：experiments/service_axis_demo/")


if __name__ == "__main__":
    main()
