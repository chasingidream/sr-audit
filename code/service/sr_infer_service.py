#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SR 图像超分推理服务 —— 把训练好的超分模型封装成「上传图片 → 返回超分后图片」的前端可调用 HTTP 服务。

与 code/service/audit_service.py（审计协议服务）互为补充：
  * audit_service.py ：输入【评测证据】，输出【可信徽章 + regime】（协议层）。
  * sr_infer_service.py：输入【一张低分辨率图片】，输出【超分后的图片】（模型层）。

设计
----
* 复用项目模型 `models/sr_model.build_model` 与任一训练好的 checkpoint（默认 experiments/ckpts/ckpt_ep29.pt，
  WorldStrat RGB / scale=4 / norm=None 的早期权重；真实三轴权重属 Tier-B Zenodo 产物，本地缺，可用 --ckpt 指定）。
* 预处理严格复刻 data/dataset.py 的约定：
    - in_ch=3 → 上传 RGB 图 /255 进 [0,1]；in_ch=1 → 取亮度。
    - norm="per_scene"：用【该图自身】的 (mu,sd) 对 LR、HR 做同样标准化（与论文一致：LR 与 HR 共用 LR 统计），
      故任意单图部署可用输入自身统计，推理后按同一 (mu,sd) 反归一化（无需 GT）。
    - 多时相 T 帧：把单图 repeat 到 T 帧（对齐锚点=frame0 时各帧对齐到自身，等价单图）。
* 输出：模型 SR 图（PNG）+ bicubic 上采样基线（用于并排对比）。
* 设备：CUDA 可用则用 GPU，否则 CPU（单图推理均很快）。
* vcomp140.dll 守卫：torch CUDA 在本机需隐藏该 DLL 防 segfault（复用 code/_dll_guard.py，在 import torch 前调用）。

运行
----
    python code/service/sr_infer_service.py --port 8766
    # 打开 http://127.0.0.1:8766/ 即前端上传示例
    # 或：python code/service/sr_call_example.py
"""
import argparse
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# 路径与 DLL 守卫：必须在 import torch 之前把 vcomp140.dll 隐藏
# --------------------------------------------------------------------------- #
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
try:
    import _dll_guard
    _dll_guard.hide()
except Exception as exc:  # pragma: no cover
    print("[warn] _dll_guard 调用失败: %s（若 CUDA segfault 请手动隐藏 vcomp140.dll）" % exc)

import torch                                   # noqa: E402  (已在上方隐藏 vcomp)
import torch.nn.functional as F                 # noqa: E402


# =========================================================================== #
# 模型加载（惰性，首次请求或启动时）
# =========================================================================== #
_MODEL = None
_CFG = None
_DEVICE = None


def load_model(ckpt_path, device=None, preprocess=None):
    """加载 checkpoint 为一个模型档案 bundle。

    bundle = {model, cfg, device, ckpt, preprocess}；preprocess 是推理预处理档案：
      * None                    -> 按 cfg.dataset.norm 走（per_scene / none）
      * {"norm": "per_scene"}   -> 用输入图自身 (mu,sd)（与训练 scene_norm 语义等价）
      * {"norm": "global", "mu": m, "sd": s} -> 用训练时数据集级 LR 统计量
      * {"norm": "none"}        -> 不归一化
    返回 bundle，同时写入模块级全局（保持独立单模型运行的向后兼容）。
    """
    global _MODEL, _CFG, _DEVICE
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("cfg", {})
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 默认 out_size=None -> 输出 = scale * 输入尺寸（动态），更利于任意尺寸图片演示
    cfg = dict(cfg)
    cfg["out_size"] = None
    from models.sr_model import build_model
    model = build_model(cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    bundle = {"model": model, "cfg": cfg, "device": device,
              "ckpt": os.path.abspath(ckpt_path),
              "preprocess": dict(preprocess) if preprocess else None}
    _MODEL, _CFG, _DEVICE = model, cfg, device
    print("[sr] loaded %s | scale=%s T=%s in_ch=%s | norm=%s | device=%s"
          % (os.path.basename(ckpt_path), cfg.get("scale"), cfg.get("T"),
             cfg.get("in_ch"),
             (bundle["preprocess"] or {}).get("norm",
              (cfg.get("dataset") or {}).get("norm")), device))
    return bundle


def _ensure():
    if _MODEL is None:
        load_model(_CKPT_PATH, _DEVICE_PREF)


# =========================================================================== #
# 预处理 / 推理 / 后处理
# =========================================================================== #
def prepare_input(img, cfg, device=None, preprocess=None):
    """PIL.Image -> (lrs [1,T,in_ch,lr,lr] 张量, stats)。

    - RGB /255 → [0,1]；in_ch=1 → 亮度。
    - resize 到 lr_size（模型训练用 64）。
    - 归一化（优先 preprocess 档案，其次 cfg.dataset.norm）：
        per_scene -> 用本图自身 (mu,sd)（与训练 scene_norm 语义等价，任意单图可部署）；
        global    -> 用档案提供的训练期数据集级 LR 统计量 (mu,sd)；
        none      -> 不归一化。
    - repeat 到 T 帧。
    """
    device = device or _DEVICE
    pre = dict(preprocess or {})
    norm = pre.get("norm") or (cfg.get("dataset") or {}).get("norm")
    in_ch = int(cfg.get("in_ch", 3))
    lr_size = int(cfg.get("lr_size", 64))

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0   # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1)).copy()                       # [3,H,W]
    if in_ch == 1:
        lum = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
        arr = lum[None]                                              # [1,H,W]
    t = torch.from_numpy(arr).float()
    t = F.interpolate(t.unsqueeze(0), size=(lr_size, lr_size),
                      mode="bicubic", align_corners=True).squeeze(0)  # [in_ch,lr,lr]

    mu = sd = None
    if norm == "per_scene":
        mu = float(t.mean())
        sd = float(t.std()) + 1e-6
        t = (t - mu) / sd
    elif norm == "global":
        if "mu" not in pre or "sd" not in pre:
            raise ValueError("norm=global 需要模型档案提供训练时的 (mu, sd)")
        mu, sd = float(pre["mu"]), float(pre["sd"])
        t = (t - mu) / sd

    T = int(cfg.get("T", 5))
    lrs = t.unsqueeze(0).repeat(T, 1, 1, 1).unsqueeze(0).to(device)  # [1,T,in_ch,lr,lr]
    return lrs, (mu, sd)


def _resolve(bundle):
    """bundle 为空时回退到模块级全局（独立单模型运行模式）。"""
    if bundle is None:
        _ensure()
        bundle = {"model": _MODEL, "cfg": _CFG, "device": _DEVICE,
                  "ckpt": os.path.abspath(_CKPT_PATH), "preprocess": None}
    return bundle


def run_sr(img, bundle=None):
    """输入 PIL.Image（低分辨率），返回 (sr_png_bytes, bic_png_bytes, meta)。

    bundle 可选（多模型网关用）；缺省用模块级全局模型。
    """
    b = _resolve(bundle)
    model, cfg, device = b["model"], b["cfg"], b["device"]
    scale = int(cfg.get("scale", 4))
    in_ch = int(cfg.get("in_ch", 3))

    lrs, (mu, sd) = prepare_input(img, cfg, device, b.get("preprocess"))
    align_mode = cfg.get("align_mode", None)
    use_rsa = align_mode is not None
    with torch.no_grad():
        pred = model(lrs, hr_ref=None, use_rsa=use_rsa, use_ref=False)   # [1,in_ch,H,W]
        bic = F.interpolate(lrs[:, 0], scale_factor=scale, mode="bicubic",
                            align_corners=False)                         # 基线

    def to_uint8(x):
        x = x.detach().cpu().float()
        if mu is not None:
            x = x * sd + mu
        x = x.clamp(0.0, 1.0)
        arr = x[0].numpy()                                           # [in_ch,H,W]
        arr = np.transpose(arr, (1, 2, 0))                            # [H,W,in_ch]
        if in_ch == 1:
            arr = arr[:, :, 0]
        return np.clip(arr * 255.0, 0, 255).astype("uint8")

    sr_np = to_uint8(pred)
    bic_np = to_uint8(bic)
    sr_png = _png_bytes(sr_np)
    bic_png = _png_bytes(bic_np)
    meta = {
        "scale": scale, "in_ch": in_ch, "T": int(cfg.get("T", 5)),
        "device": str(device), "out_size": list(pred.shape[-2:]),
        "norm": (b.get("preprocess") or {}).get("norm") or (cfg.get("dataset") or {}).get("norm"),
        "ckpt": os.path.basename(b["ckpt"]) if b.get("ckpt") else None,
    }
    return sr_png, bic_png, meta, sr_np, bic_np


def _png_bytes(arr):
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_sr_side_by_side(img, bundle=None):
    """返回 ([bicubic | model SR] 并排 PNG, meta, sr_np, bic_np)（前端对比用）。"""
    sr_png, bic_png, meta, sr_np, bic_np = run_sr(img, bundle)
    bic = np.array(Image.open(io.BytesIO(bic_png))
                   .convert("RGB" if meta["in_ch"] == 3 else "L"))
    sr = np.array(Image.open(io.BytesIO(sr_png))
                 .convert("RGB" if meta["in_ch"] == 3 else "L"))
    if bic.ndim == 2:
        bic = np.stack([bic] * 3, -1)
    if sr.ndim == 2:
        sr = np.stack([sr] * 3, -1)
    h = max(bic.shape[0], sr.shape[0])
    if bic.shape[0] < h:
        pad = np.zeros((h - bic.shape[0], bic.shape[1], 3), dtype=np.uint8)
        bic = np.concatenate([bic, pad], 0)
    if sr.shape[0] < h:
        pad = np.zeros((h - sr.shape[0], sr.shape[1], 3), dtype=np.uint8)
        sr = np.concatenate([sr, pad], 0)
    combined = np.concatenate([bic, sr], axis=1)
    buf = io.BytesIO()
    Image.fromarray(combined, mode="RGB").save(buf, format="PNG")
    return buf.getvalue(), meta, sr_np, bic_np


# =========================================================================== #
# HTTP 服务
# =========================================================================== #
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, body, code=200, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(load_frontend(), ctype="text/html; charset=utf-8")
        elif path == "/health":
            self._send(json.dumps({"status": "ok", "service": "sr-infer",
                                    "loaded": _MODEL is not None,
                                    "device": str(_DEVICE)}, ensure_ascii=False))
        else:
            self._send(json.dumps({"error": "not found",
                                    "available": ["/", "/health", "/sr"]}), 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path != "/sr":
            self._send(json.dumps({"error": "not found", "available": ["/sr"]}), 404)
            return
        qs = parse_qs(parsed.query)
        side_by_side = qs.get("compare", ["0"])[0] in ("1", "true", "True")
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            ctype = self.headers.get("Content-Type", "")

            img = None
            if "multipart/form-data" in ctype:
                img = self._read_multipart(raw, ctype)
            else:
                obj = json.loads(raw.decode("utf-8"))
                b64 = obj.get("image")
                if not b64:
                    raise ValueError("JSON 体需含 image(base64)")
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                side_by_side = bool(obj.get("compare", side_by_side))

            if img is None:
                raise ValueError("未解析到图片")

            if side_by_side:
                out, meta, _, _ = run_sr_side_by_side(img)
            else:
                sr_png, bic_png, meta, _, _ = run_sr(img)
                out = sr_png
            # 默认返回模型 SR 图（image/png）；compare=1 时返回并排对比图
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self._cors()
            self.send_header("Content-Length", str(len(out)))
            self.send_header("X-SR-Meta", base64.b64encode(
                json.dumps(meta, ensure_ascii=False).encode("utf-8")).decode("ascii"))
            self.end_headers()
            self.wfile.write(out)
        except Exception as exc:
            self._send(json.dumps({"error": "sr failed", "detail": str(exc)},
                                  ensure_ascii=False), 400)

    @staticmethod
    def _read_multipart(raw, ctype):
        import email
        msg = email.message_from_bytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
        for part in msg.walk():
            if part.get_content_disposition() != "form-data":
                continue
            name = part.get_param("name", header="Content-Disposition")
            ctype_p = part.get_content_type() or ""
            if name == "image" or ctype_p.startswith("image/"):
                payload = part.get_payload(decode=True)
                if payload:
                    return Image.open(io.BytesIO(payload))
        return None


def load_frontend():
    here = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(here, "sr_demo.html")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SR Infer Service</h1><p>sr_demo.html 未找到。</p>"


def main():
    ap = argparse.ArgumentParser(description="SR 图像超分推理 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--ckpt", default=os.path.join(_CODE_DIR, "..", "experiments", "ckpts", "ckpt_ep29.pt"))
    ap.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    args = ap.parse_args()
    global _CKPT_PATH, _DEVICE_PREF
    _CKPT_PATH = os.path.abspath(args.ckpt)
    _DEVICE_PREF = args.device
    # 启动时加载，便于尽早报错
    try:
        load_model(_CKPT_PATH, _DEVICE_PREF)
    except Exception as exc:
        print("[warn] 启动时加载 checkpoint 失败: %s（将在首次请求时重试）" % exc)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("SR 推理服务已启动: http://%s:%d/" % (args.host, args.port))
    print("  POST /sr  上传图片(表单 image 或 JSON {image:base64}) → 返回 SR 图(PNG)")
    print("  POST /sr?compare 或 JSON {compare:true} → 返回 [bicubic|SR] 并排对比图")
    print("  GET  /    前端上传示例页    Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] 服务已停止")
    finally:
        httpd.server_close()


_CKPT_PATH = os.path.join(_CODE_DIR, "..", "experiments", "ckpts", "ckpt_ep29.pt")
_DEVICE_PREF = None

if __name__ == "__main__":
    main()
