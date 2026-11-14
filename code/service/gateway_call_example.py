#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一网关 Python 客户端示例（纯标准库 urllib，无需 requests）。

演示四个核心调用：
  1) GET  /health                      —— 网关与各轴模型状态
  2) POST /sr?model=ws&compare=1       —— 上传图片 -> 返回 SR 并排 PNG（二进制）
  3) POST /audit                       —— 提交逐场景证据 -> G1–G5 门控 + 徽章 + regime（JSON）
  4) POST /sr_audit?model=ws           —— 一键：上传图片 -> JSON{SR 图 base64 + 无 GT 预检徽章}

用法：
    python code/service/gateway_call_example.py [图片路径]
默认图片 = 本目录上一级的评测样例（若无则用随机噪声图示意）。
"""
import base64
import json
import os
import sys
import urllib.request
import uuid

BASE = "http://127.0.0.1:8770"
BOUNDARY = "----srclientboundary" + uuid.uuid4().hex[:8]


def multipart(field, filename, payload, ctype="image/png"):
    body = (("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
             "Content-Type: %s\r\n\r\n" % (BOUNDARY, field, filename, ctype))
            .encode() + payload +
            ("\r\n--%s--\r\n" % BOUNDARY).encode())
    return body, "multipart/form-data; boundary=%s" % BOUNDARY


def post(path, data=None, content_type="application/json"):
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    if content_type:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, dict(r.headers), r.read()


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path and os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        fname = os.path.basename(img_path)
    else:
        import io
        from PIL import Image
        import numpy as np
        arr = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        img_bytes = buf.getvalue()
        fname = "_random64.png"

    # ---- 1) 健康检查 ----
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        health = json.loads(r.read())
    print("[1] /health ->", json.dumps(health["axes"], ensure_ascii=False)[:200], "...")

    # ---- 2) 出图：上传 -> SR 并排 PNG（二进制响应）----
    body, ct = multipart("image", fname, img_bytes)
    status, headers, out = post("/sr?model=ws&compare=1", body, ct)
    out_png = "_gateway_sr_out.png"
    with open(out_png, "wb") as f:
        f.write(out)
    meta = json.loads(base64.b64decode(headers.get("X-SR-Meta", "e30=")))
    print("[2] /sr?model=ws&compare=1 -> %s %d bytes -> %s | meta=%s"
          % (status, len(out), out_png, json.dumps(meta, ensure_ascii=False)))

    # ---- 3) 验真：逐场景证据 -> 徽章 + regime ----
    ex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_request.json")
    with open(ex_path, "rb") as f:
        payload = f.read()
    status, _, resp = post("/audit", payload)
    card = json.loads(resp)["audit"]
    print("[3] /audit -> %s | credibility=%s regime=%s gate=%s"
          % (status, card["credibility"], card["regime"], card["gate_verdict"]))
    print("    headline:", card["headline"][:110])

    # ---- 4) 一键：上传 -> JSON{SR 图 base64 + 无 GT 预检徽章} ----
    body, ct = multipart("image", fname, img_bytes)
    status, _, resp = post("/sr_audit?model=ws&compare=1", body, ct)
    obj = json.loads(resp)
    sr_png = base64.b64decode(obj["sr_image_base64"])
    with open("_gateway_sraudit_out.png", "wb") as f:
        f.write(sr_png)
    print("[4] /sr_audit?model=ws -> %s | SR 图 %d bytes -> _gateway_sraudit_out.png"
          % (status, len(sr_png)))
    print("    precheck:", obj["precheck"]["headline"])
    for k, v in obj["precheck"]["gates"].items():
        print("      %-8s %-14s %s" % (v["verdict"], k, v["detail"]))
    print("    note:", obj["precheck"]["note"][:80], "...")


if __name__ == "__main__":
    main()
