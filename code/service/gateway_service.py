#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一网关 —— 一个端口同时提供「出图 + 验真 + 一键出图验真」。

组合两套互补服务（audit_service.py 协议层 / sr_infer_service.py 模型层）：

路由总表
--------
  GET  /                     统一演示页 gateway_demo.html
  GET  /health               网关 + 各轴模型装载状态
  GET  /axes                 三轴模型档案（ckpt / 预处理档案 / 是否支持任意图上传）
  GET  /schema               全部端点 I/O 说明
  GET  /example/audit        审计示例请求（example_request.json）
  POST /audit                验真：逐场景证据 -> G1–G5 门控 + 可信徽章 + regime（audit_service）
  POST /assess               验真：对已有审计卡片跑 derive（audit_service）
  POST /sr?model=&compare=   出图：上传图 -> SR PNG（X-SR-Meta 带元信息）
  POST /sr_audit?model=      一键：上传图 -> JSON{SR图base64 + 无GT预检徽章卡}
  POST /load  {"axis":...}   预加载某轴模型

三轴模型档案（Tier-B 末轮权重；SHA256 与 experiments/RELEASE_MANIFEST.md 逐位对账一致）
--------
  ws      WorldStrat 6.63×  mtxa_aligned  per_scene 归一化   支持任意 RGB 上传
  probav  PROBA-V 3×        mtxa_aligned  global(mu,sd)      支持任意图上传（亮度近似 RED 波段）
  s2v     SEN2VENµS 2×      sisr          global(mu,sd)      仅评测路径（4 波段，RGB 上传无法表达）

一键预检（/sr_audit）口径说明
--------
无 GT 单图只能做**降级预检** P1–P4（阈值沿用论文门控的同类量）：
  P1 结构一致性  CC(SR, bicubic)   ≥0.85 PASS / ≥0.6 WARN / <0.6 FAIL
  P2 动态范围    std(SR)/std(bic)  ∈[0.5,1.8]（同 G3 阈值）
  P3 周期伪影    FFT 谐波比 <2.5（同 G4 阈值，复用 sr_audit.periodicity_scan）
  P4 高频能量比  SR/bicubic（信息项，不判定）
它**不是**论文的 G1–G5 全审计：G1/G2/G3/G5 需要 GT 配对评测。徽章仅表示
"这张输出图本身未见结构发散/统计异常/周期伪影"，不代表增益真实。

运行
--------
    python code/service/gateway_service.py --port 8770
    # 打开 http://127.0.0.1:8770/ 统一演示页
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
# 路径：SERVICE_DIR = code/service，CODE_DIR = code，REPO = 仓库根
# --------------------------------------------------------------------------- #
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SERVICE_DIR)
REPO = os.path.dirname(CODE_DIR)
for p in (SERVICE_DIR, CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# sr_infer_service 内部先隐藏 vcomp140.dll 再 import torch（网关继承该顺序）
import torch                            # noqa: E402  (经由 sr_infer_service 的守卫加载)
import sr_infer_service as sri          # noqa: E402
import audit_service as aud             # noqa: E402
from sr_audit import periodicity_scan   # noqa: E402  (G4 同口径 FFT 扫描)


# =========================================================================== #
# 三轴模型档案 + 惰性装载
# =========================================================================== #
AXES = {
    "ws": {
        "label": "WorldStrat 6.63x (mtxa_aligned)",
        "ckpt": os.path.join("experiments", "ckpts_worldstrat", "mtxa_aligned", "ckpt_ep29.pt"),
        "preprocess": {"norm": "per_scene"},
        "http_upload": True,
        "note": "RGB 3 通道；per_scene 归一化=用上传图自身 (mu,sd)，与训练 scene_norm 语义等价",
    },
    "probav": {
        "label": "PROBA-V 3x (mtxa_aligned, n300_s42)",
        "ckpt": os.path.join("experiments", "ckpts_probav_n300_s42", "mtxa_aligned", "ckpt_ep29.pt"),
        # global 统计量 = 训练期数据集级 LR 侧估计（audit_axis 运行日志实测值）
        "preprocess": {"norm": "global", "mu": 0.08206, "sd": 0.07568},
        "http_upload": True,
        "note": "单波段 RED；任意上传图按亮度近似，global 归一化 (mu=0.08206, sd=0.07568)",
    },
    "s2v": {
        "label": "SEN2VENµS 2x (sisr)",
        "ckpt": os.path.join("experiments", "ckpts_s2v", "sisr", "ckpt_ep29.pt"),
        "preprocess": {"norm": "global", "mu": 0.07142, "sd": 0.11740},
        "http_upload": False,
        "note": "4 波段 (B2/B3/B4/B8)：RGB 上传无法表达，仅评测数据路径可用",
    },
}
_BUNDLES = {}


def get_bundle(axis):
    if axis not in AXES:
        raise KeyError("未知轴 %r（可选：%s）" % (axis, ", ".join(AXES)))
    if axis not in _BUNDLES:
        ckpt = os.path.join(REPO, AXES[axis]["ckpt"])
        _BUNDLES[axis] = sri.load_model(ckpt, preprocess=AXES[axis]["preprocess"])
    return _BUNDLES[axis]


# =========================================================================== #
# 一键预检（无 GT 单图，P1–P4）
# =========================================================================== #
NO_GT_NOTE = ("无 GT 单图预检（P1–P4），非论文 G1–G5 全审计：G1/G2/G3/G5 需要 GT 配对评测。"
              "本徽章仅表示这张输出图未见结构发散/统计异常/周期伪影，不代表增益真实；"
              "全审计请 POST /audit 提交逐场景证据。")


def _precheck(sr_np, bic_np):
    sr = sr_np.astype(np.float64) / 255.0
    bic = bic_np.astype(np.float64) / 255.0
    if sr.ndim == 2:
        sr, bic = sr[:, :, None], bic[:, :, None]

    # P1 结构一致性：CC(SR, bicubic)（逐通道 Pearson 取均值）
    ccs = []
    for c in range(sr.shape[2]):
        a, b = sr[:, :, c].ravel(), bic[:, :, c].ravel()
        if a.std() > 1e-12 and b.std() > 1e-12:
            ccs.append(float(np.corrcoef(a, b)[0, 1]))
    cc = float(np.mean(ccs)) if ccs else float("nan")
    p1 = "PASS" if cc >= 0.85 else ("WARN" if cc >= 0.6 else "FAIL")

    # P2 动态范围：std(SR)/std(bicubic)（G3 同阈值）
    ratios = [float(sr[:, :, c].std() / (bic[:, :, c].std() + 1e-12))
              for c in range(sr.shape[2])]
    srat = float(np.mean(ratios))
    p2 = "PASS" if 0.5 <= srat <= 1.8 else "FAIL"

    # P3 周期伪影：G4 同口径（sr_audit.periodicity_scan，阈值 <2.5）
    chw = torch.from_numpy(np.ascontiguousarray(np.transpose(sr, (2, 0, 1)))).float()
    ratio, period = periodicity_scan(chw)
    p3 = "PASS" if ratio < 2.5 else "WARN"

    # P4 信息项：高频（一阶差分）能量比
    def _hf(x):
        return float(np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean())
    hf_ratio = _hf(sr.mean(2)) / (_hf(bic.mean(2)) + 1e-12)

    verdict = "FAIL" if "FAIL" in (p1, p2, p3) else ("WARN" if "WARN" in (p1, p2, p3) else "PASS")
    gates = {
        "p1_结构一致性": {"verdict": p1,
                          "detail": "CC(SR,bicubic)=%.3f (≥0.85 PASS / ≥0.6 WARN)" % cc},
        "p2_动态范围": {"verdict": p2,
                        "detail": "std(SR)/std(bicubic)=%.2f (阈值 [0.5,1.8])" % srat},
        "p3_周期伪影": {"verdict": p3,
                        "detail": "FFT ratio=%.2f @ period=%dpx (阈值<2.5)" % (ratio, period)},
    }
    info = {"hf_energy_ratio_sr_over_bicubic": round(hf_ratio, 3),
            "note": "≈1 表示 SR 未比 bicubic 多出高频能量（可能无真实细节增益）；过大=噪声放大/幻觉。仅信息项，不判定。"}
    headline = ("PRECHECK | P1=%s(CC=%.2f) P2=%s(ratio=%.2f) P3=%s(ratio=%.2f@%dpx) -> %s"
                % (p1, cc, p2, srat, p3, ratio, period, verdict))
    return {"verdict": verdict, "gates": gates, "info": info,
            "headline": headline, "note": NO_GT_NOTE}


def _b64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")


# =========================================================================== #
# HTTP Handler
# =========================================================================== #
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, body, code=200, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self._cors()
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200, extra=None):
        self._send(json.dumps(obj, ensure_ascii=False), code=code, extra=extra)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ---------------- GET ----------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(self._load_page(), ctype="text/html; charset=utf-8")
        elif path == "/health":
            self._send_json({
                "status": "ok", "service": "sr-gateway",
                "audit_layer": "audit_service (G1-G5 门控 + 徽章 + regime)",
                "infer_layer": "sr_infer_service (SpatioTemporalSR)",
                "axes": {a: {"loaded": a in _BUNDLES,
                             "device": str(_BUNDLES[a]["device"]) if a in _BUNDLES else None,
                             "http_upload": spec["http_upload"],
                             "ckpt": spec["ckpt"],
                             "label": spec["label"]}
                         for a, spec in AXES.items()},
            })
        elif path == "/axes":
            self._send_json({"axes": AXES})
        elif path == "/schema":
            self._send_json(self._schema())
        elif path == "/example/audit":
            fp = os.path.join(SERVICE_DIR, "example_request.json")
            if os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    self._send(f.read(), ctype="application/json")
            else:
                self._send_json({"error": "example_request.json 未找到"}, 404)
        else:
            self._send_json({"error": "not found",
                             "available": ["/", "/health", "/axes", "/schema",
                                           "/example/audit", "/audit", "/assess",
                                           "/sr", "/sr_audit", "/load"]}, 404)

    # ---------------- POST ----------------
    def do_POST(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        try:
            if path in ("/audit", "/assess"):
                payload = self._read_json_body()
                fn = aud.run_audit if path == "/audit" else aud.run_assess
                self._send_json(fn(payload))
                return
            if path == "/load":
                payload = self._read_json_body()
                axis = payload.get("axis")
                b = get_bundle(axis)
                self._send_json({"loaded": True, "axis": axis,
                                 "device": str(b["device"]),
                                 "ckpt": os.path.basename(b["ckpt"])})
                return
            if path in ("/sr", "/sr_audit"):
                self._handle_infer(path, qs)
                return
            self._send_json({"error": "not found", "available":
                             ["/audit", "/assess", "/sr", "/sr_audit", "/load"]}, 404)
        except Exception as exc:
            self._send_json({"error": "gateway failed", "detail": str(exc)}, 400)

    # ---------------- 出图 / 一键 ----------------
    def _handle_infer(self, path, qs):
        axis = (qs.get("model") or qs.get("axis") or ["ws"])[0]
        spec = AXES.get(axis)
        if spec is None:
            raise ValueError("未知轴 %r（可选：%s）" % (axis, ", ".join(AXES)))
        if path == "/sr_audit" and not spec["http_upload"]:
            raise ValueError("轴 %s 不支持任意图上传（%s）" % (axis, spec["note"]))

        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")
        compare_json = False
        if "multipart/form-data" in ctype:
            img = sri.Handler._read_multipart(raw, ctype)
        else:
            obj = json.loads(raw.decode("utf-8"))
            b64img = obj.get("image")
            if not b64img:
                raise ValueError("JSON 体需含 image(base64)")
            img = Image.open(io.BytesIO(base64.b64decode(b64img)))
            compare_json = bool(obj.get("compare", False))
        if img is None:
            raise ValueError("未解析到图片")

        compare = (qs.get("compare", ["0"])[0] in ("1", "true", "True")) or compare_json
        bundle = get_bundle(axis)

        if path == "/sr":
            if compare:
                out, meta, _, _ = sri.run_sr_side_by_side(img, bundle)
            else:
                sr_png, _, meta, _, _ = sri.run_sr(img, bundle)
                out = sr_png
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self._cors()
            self.send_header("Content-Length", str(len(out)))
            self.send_header("X-SR-Meta", base64.b64encode(
                json.dumps(meta, ensure_ascii=False).encode("utf-8")).decode("ascii"))
            self.end_headers()
            self.wfile.write(out)
            return

        # /sr_audit：出图 + 无 GT 预检徽章，一次返回
        sr_png, bic_png, meta, sr_np, bic_np = sri.run_sr(img, bundle)
        combined = None
        if compare:
            combined, _, _, _ = sri.run_sr_side_by_side(img, bundle)
        pre = _precheck(sr_np, bic_np)
        self._send_json({
            "service": "sr-gateway", "axis": axis, "label": spec["label"],
            "meta": meta, "precheck": pre,
            "sr_image_base64": _b64(sr_png),
            "compare_image_base64": _b64(combined) if combined else None,
        })

    # ---------------- 工具 ----------------
    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _load_page(self):
        fp = os.path.join(SERVICE_DIR, "gateway_demo.html")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>SR Gateway</h1><p>gateway_demo.html 未找到。</p>"

    @staticmethod
    def _schema():
        return {
            "service": "sr-gateway",
            "layers": {
                "出图（模型层）": "POST /sr?model=ws|probav&compare=1 — 上传图(multipart image 或 JSON {image:base64}) -> SR PNG / 并排 PNG",
                "验真（协议层）": "POST /audit — {method,task,scale,val_dr,scenes:[{psnr_model,psnr_bicubic,psnr_mean,cc_bicubic,cc_model,std_model,std_gt,bias_abs,grid_ratio_model,grid_period_model,leak_stage}]} -> G1-G5 门控+徽章+regime",
                "验真（已有卡片）": "POST /assess — 卡片对象 / {card} / {cards:[...]} -> derive 徽章",
                "一键": "POST /sr_audit?model= — 上传图 -> JSON{sr_image_base64, precheck{P1-P4 徽章}, meta}",
            },
            "axes": {a: {"ckpt": s["ckpt"], "http_upload": s["http_upload"], "note": s["note"]}
                     for a, s in AXES.items()},
            "precheck_note": NO_GT_NOTE,
        }


def main():
    ap = argparse.ArgumentParser(description="SR 统一网关（出图 + 验真 + 一键）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--preload", default="ws,probav",
                    help="启动即加载的轴（逗号分隔）；留空则惰性加载")
    args = ap.parse_args()

    for axis in [a for a in args.preload.split(",") if a]:
        try:
            get_bundle(axis)
        except Exception as exc:
            print("[warn] 预加载 %s 失败: %s" % (axis, exc))

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("SR 统一网关已启动: http://%s:%d/" % (args.host, args.port))
    print("  POST /sr?model=ws        出图（PNG）        model=ws|probav")
    print("  POST /audit              验真（G1-G5 徽章）")
    print("  POST /sr_audit?model=ws  一键（SR 图 + 预检徽章 JSON）")
    print("  GET  /                   统一演示页    Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] 网关已停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
