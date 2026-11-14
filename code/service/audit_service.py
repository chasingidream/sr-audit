#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SR-Audit Service —— 把论文的核心交付物「SR-Audit 评测审计协议」封装成一个前端可直接调用的 HTTP 服务。

设计原则
--------
* 纯 numpy（唯一硬依赖）+ 标准库 http.server，**不依赖 torch / 权重 / GPU**。
  项目里 `sr_audit.py` 的 `audit()` 需要训练好的模型 + 数据集 + CUDA，无法塞进一个轻量前端服务；
  而论文真正可复用、可机器核查的"成果"是 **审计协议本身**（G1–G5 门控 + 可信徽章 + 科学 regime 判定）。
  因此本服务把协议计算独立出来：调用方提供某个超分方法的**评测证据**（逐场景指标），
  服务返回门控判定、可信徽章（VERIFIED/QUALIFIED/VOID/UNVERIFIED）、科学 regime（positive/saturated/floor-failure），
  以及配套的 Δ_floor、配对 t 检验（CI / p / Cohen's d）等金标准统计量。
* 徽章与 regime 判定**复用** `code/sr_audit_report.py` 的 `derive()`（纯标准库），保证与论文报告标准单一来源、口径一致。
* 提供两种输入模式：
    - `POST /audit`  ：传入逐场景证据（scenes），服务从零算出全部门控 + 统计量 + 徽章 + regime。
    - `POST /assess` ：直接传入一个已存在的审计卡片（含 g1..g5 / delta_floor_db / audit_gain_db / audit_p 等），
                       服务只跑 `derive()` 得到徽章 + regime（与 `sr_audit_report.py` 等价）。
* `GET /` 返回一个自带示例、可实时调用 `/audit` 的前端演示页（即"前端调用示例"）。

运行
----
    python code/service/audit_service.py --port 8765
    # 另开一个页面打开 http://127.0.0.1:8765/ 即可看到前端调用示例
    # 或用 code/service/call_example.py 做 Python 客户端调用

协议版本：SR-Audit Report Standard v1（与 code/sr_audit_report.py 对齐）。
"""
import argparse
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

# --------------------------------------------------------------------------- #
# 复用论文报告标准（纯标准库，无 torch 依赖）。放在 code/ 下，import 之。
# --------------------------------------------------------------------------- #
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

try:
    from sr_audit_report import derive as _derive_card  # 单一来源：徽章 + regime 逻辑
    _HAS_DERIVE = True
except Exception:  # pragma: no cover —— 极端情况下退化为内联最小实现
    _HAS_DERIVE = False

# 门控阈值（与 code/sr_audit.py / sr_audit_report.py 保持一致）
G1_CC_MIN = 0.30          # G1 配对完整性：CC(bicubic(LR), GT) >= 0.30
G2_CC_PASS = 0.90         # G2 结构保真：model CC / bicubic CC
G2_CC_WARN = 0.70
G3_STD_LO, G3_STD_HI = 0.5, 1.8   # G3 输出统计：std_ratio in [0.5, 1.8]
G4_GRID_MAX = 2.5         # G4 周期伪影：FFT 最强谐波比值 < 2.5
G5_PSNR_MAX = 45.0        # G5 泄漏风险：PSNR > 45 dB 视为不合理高
SIGNIFICANCE = 0.05       # 配对 t 检验显著性水平


# =========================================================================== #
# 纯 numpy 统计工具（无 scipy 依赖，保证任意 CPython+numpy 可跑）
# =========================================================================== #
def _betacf(a, b, x, maxit=300, eps=3e-12, fpmin=1e-300):
    """正则化不完全 Beta 函数的连分式（Numerical Recipes 6.4）。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta + math.log(_betacf(a, b, x))) / a
    return 1.0 - math.exp(lbeta + math.log(_betacf(b, a, 1.0 - x))) / b


def _t_two_sided_p(t, df):
    """双尾 t 检验 p 值：p = I_{df/(df+t^2)}(df/2, 1/2)。"""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betainc(0.5 * df, 0.5, x)


def _t_crit(df, alpha=0.05):
    """双尾置信水平 1-alpha 的 t 临界值（二分法）。"""
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = _t_two_sided_p(mid, df)
        if p > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_stats(diff):
    """逐样本差值 -> 均值 / 95% CI / 双尾 p / Cohen's d / t / df（纯 numpy）。"""
    diff = np.asarray(diff, dtype=np.float64)
    n = diff.size
    if n < 2:
        return {"n": int(n), "mean_diff": None, "ci": None,
                "p": None, "cohen_d": None, "t_stat": None, "df": int(n - 1)}
    dbar = float(diff.mean())
    s = float(diff.std(ddof=1))
    se = s / math.sqrt(n)
    t = dbar / se if se > 0 else 0.0
    df = n - 1
    p = _t_two_sided_p(t, df)
    tc = _t_crit(df, 0.05)
    ci = [dbar - tc * se, dbar + tc * se]
    cohen_d = dbar / s if s > 0 else 0.0
    return {"n": int(n), "mean_diff": round(dbar, 4),
            "ci": [round(ci[0], 4), round(ci[1], 4)],
            "p": float(p), "cohen_d": round(cohen_d, 4),
            "t_stat": round(float(t), 4), "df": int(df)}


# =========================================================================== #
# 核心：从逐场景证据计算门控 + 徽章 + regime
# =========================================================================== #
def _gate_from_evidence(scenes):
    """依据逐场景证据计算 G1–G5 的原始判定（verdict + detail）。

    任一证据缺失 -> 该门判为 n/a（不触发 FAIL/WARN），便于优雅降级。
    """
    n = len(scenes)
    # G1 配对完整性
    cc_b = [s.get("cc_bicubic") for s in scenes if s.get("cc_bicubic") is not None]
    if cc_b:
        pair_cc = float(np.mean(cc_b))
        g1 = ("PASS" if pair_cc >= G1_CC_MIN else "FAIL",
              "CC(bic,GT)=%.3f (阈值>=%.2f)" % (pair_cc, G1_CC_MIN))
    else:
        g1 = ("n/a", "缺少 cc_bicubic 证据")

    # G2 结构保真
    cc_m = [s.get("cc_model") for s in scenes if s.get("cc_model") is not None]
    if cc_b and cc_m:
        r_cc = float(np.mean(cc_m)) / max(float(np.mean(cc_b)), 1e-6)
        v = ("PASS" if r_cc >= G2_CC_PASS else
             ("WARN" if r_cc >= G2_CC_WARN else "FAIL"))
        g2 = (v, "model CC / bicubic CC = %.2f" % r_cc)
    else:
        g2 = ("n/a", "缺少 cc_model / cc_bicubic 证据")

    # G3 输出统计（std_ratio = std_model / std_gt）
    ratios = []
    for s in scenes:
        if s.get("std_ratio") is not None:
            ratios.append(float(s["std_ratio"]))
        elif s.get("std_model") is not None and s.get("std_gt") is not None:
            sg = float(s["std_gt"])
            if sg > 0:
                ratios.append(float(s["std_model"]) / sg)
    if ratios:
        std_ratio = float(np.mean(ratios))
        g3 = ("PASS" if G3_STD_LO <= std_ratio <= G3_STD_HI else "FAIL",
              "std_ratio=%.2f, |bias|=%.4f" % (
                  std_ratio,
                  float(np.mean([s.get("bias_abs", 0.0) for s in scenes]))))
    else:
        g3 = ("n/a", "缺少 std_model/std_gt 或 std_ratio 证据")

    # G4 周期伪影
    gr = [s.get("grid_ratio_model") for s in scenes if s.get("grid_ratio_model") is not None]
    if gr:
        grm = float(np.mean(gr))
        gp = [s.get("grid_period_model") for s in scenes if s.get("grid_period_model") is not None]
        period = int(np.median(gp)) if gp else -1
        g4 = ("PASS" if grm < G4_GRID_MAX else "WARN",
              "ratio=%.2f @ period=%spx" % (grm, period))
    else:
        g4 = ("n/a", "缺少 grid_ratio_model 证据")

    # G5 泄漏风险
    leak_stage = any(bool(s.get("leak_stage", False)) for s in scenes)
    over_psnr = any((s.get("psnr_model") or 0.0) > G5_PSNR_MAX for s in scenes)
    if leak_stage or over_psnr:
        why = "HR 进网络" if leak_stage else "PSNR>%.0f dB" % G5_PSNR_MAX
        g5 = ("FAIL", why)
    else:
        g5 = ("PASS", "未见泄漏")

    gates = {
        "g1": {"verdict": g1[0], "detail": g1[1]},
        "g2": {"verdict": g2[0], "detail": g2[1]},
        "g3": {"verdict": g3[0], "detail": g3[1]},
        "g4": {"verdict": g4[0], "detail": g4[1]},
        "g5": {"verdict": g5[0], "detail": g5[1]},
    }
    return gates, {"pair_cc": float(np.mean(cc_b)) if cc_b else None,
                   "std_ratio": float(np.mean(ratios)) if ratios else None}


def run_audit(payload):
    """`POST /audit` 主逻辑：逐场景证据 -> 完整审计结果。

    输入 payload 形如：
        {
          "method": "MySRMethod",
          "task": "WorldStrat 6.63x cross-sensor",
          "scale": 7,
          "val_dr": 15.06,
          "split": "val",
          "scenes": [
            {"psnr_model":29.38,"psnr_bicubic":30.82,"psnr_mean":33.08,
             "cc_bicubic":0.303,"cc_model":0.231,
             "std_model":0.18,"std_gt":0.24,"bias_abs":0.889,
             "grid_ratio_model":1.05,"grid_period_model":6},
            ...
          ]
        }
    返回完整审计对象（stats + gates + derive 后的徽章/regime）。
    """
    if not isinstance(payload, dict):
        raise ValueError("payload 必须是 JSON 对象")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) == 0:
        raise ValueError("缺少 scenes（逐场景评测证据）列表，或为空")
    method = payload.get("method") or payload.get("name") or payload.get("stage") or "unnamed-method"
    task = payload.get("task")
    scale = payload.get("scale")
    val_dr = payload.get("val_dr")
    split = payload.get("split")

    # 逐场景 PSNR（三类预测器）
    pm = np.array([float(s["psnr_model"]) for s in scenes], dtype=np.float64)
    pb = np.array([float(s["psnr_bicubic"]) for s in scenes], dtype=np.float64)
    pf = np.array([float(s["psnr_mean"]) for s in scenes], dtype=np.float64)

    mean_psnr_model = float(pm.mean())
    mean_psnr_bic = float(pb.mean())
    mean_psnr_mean = float(pf.mean())

    gain_db = mean_psnr_model - mean_psnr_bic              # model - bicubic
    delta_floor_db = mean_psnr_bic - mean_psnr_mean       # bicubic - mean 地板（零训练量）
    floor_gap_db = mean_psnr_model - mean_psnr_mean       # model - mean 地板
    passes_floor = bool(floor_gap_db > 0)

    pst = paired_stats(pm - pb)  # 配对 t 检验（金标准口径：逐样本差值）

    gates, aux = _gate_from_evidence(scenes)

    # 组装成 sr_audit_report.derive 的卡片 schema
    card = {
        "method": method,
        "task": task,
        "n_test": len(scenes),
        "split": split,
        "g1": gates["g1"]["verdict"],
        "g2": gates["g2"]["verdict"],
        "g3": gates["g3"]["verdict"],
        "g4": gates["g4"]["verdict"],
        "g5": gates["g5"]["verdict"],
        "delta_floor_db": round(delta_floor_db, 4),
        "audit_gain_db": round(gain_db, 4),
        "audit_p": pst.get("p"),
        "audit_d": pst.get("cohen_d"),
        "audit_ci": pst.get("ci"),
        "passes_floor": passes_floor,
        "floor_gap_db": round(floor_gap_db, 4),
    }
    rec = _derive(card)

    return {
        "standard": "SR-Audit Report Standard v1",
        "input": {
            "method": method, "task": task, "scale": scale,
            "val_dr": val_dr, "split": split, "n_test": len(scenes),
        },
        "stats": {
            "n_test": len(scenes),
            "psnr": {
                "mean_floor": round(mean_psnr_mean, 4),
                "bicubic": round(mean_psnr_bic, 4),
                "model": round(mean_psnr_model, 4),
            },
            "delta_floor_db": round(delta_floor_db, 4),
            "floor_gap_db": round(floor_gap_db, 4),
            "gain_db": round(gain_db, 4),
            "paired": pst,
            "aux": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in aux.items()},
        },
        "gates": gates,
        "audit": rec,
    }


def run_assess(payload):
    """`POST /assess`：直接对已有的审计卡片跑 derive（等价于 sr_audit_report.py）。

    payload 可为一个卡片对象，或 {card: {...}}，或 {cards: [...]}.
    """
    if isinstance(payload, dict):
        if "card" in payload:
            cards = [payload["card"]]
        elif "cards" in payload and isinstance(payload["cards"], list):
            cards = payload["cards"]
        else:
            cards = [payload]
    elif isinstance(payload, list):
        cards = payload
    else:
        raise ValueError("payload 必须是卡片对象 / {card} / {cards:[...]}")
    recs = [_derive(c) for c in cards]
    return {"standard": "SR-Audit Report Standard v1", "n_cards": len(recs), "cards": recs}


def _derive(card):
    """徽章 + regime 判定：优先复用 sr_audit_report.derive，否则内联等价实现。"""
    if _HAS_DERIVE:
        return _derive_card(card)
    # ---- 内联兜底（与 sr_audit_report.derive 等价）----
    GATES = ("g1", "g2", "g3", "g4", "g5")

    def _norm(v, gate=None):
        if v is None:
            return "n/a"
        if isinstance(v, (int, float)):
            if gate == "g3":
                return "PASS" if 0.5 <= float(v) <= 1.8 else "FAIL"
            if gate == "g4":
                return "PASS" if float(v) < 2.5 else "FAIL"
            return "n/a"
        s = str(v).strip().upper()
        for k in ("PASS", "WARN", "FAIL"):
            if s.startswith(k):
                return k
        return "n/a"

    norm = [_norm(card.get(k), k) for k in GATES]
    if "FAIL" in norm:
        gv = "FAIL"
    elif "WARN" in norm:
        gv = "WARN"
    elif "PASS" in norm:
        gv = "PASS"
    else:
        gv = "n/a"
    credibility = {"FAIL": "VOID", "WARN": "QUALIFIED", "PASS": "VERIFIED"}.get(gv, "UNVERIFIED")

    gain = card.get("audit_gain_db")
    p = card.get("audit_p")
    df = card.get("delta_floor_db")
    pf = card.get("passes_floor")
    if pf is None and card.get("floor_gap_db") is not None:
        pf = float(card["floor_gap_db"]) > 0
    if gain is None:
        regime = "n/a"
    elif df is not None and df <= 0:
        regime = "floor-failure"
    elif pf is False:
        regime = "floor-failure"
    elif p is not None and float(p) >= SIGNIFICANCE:
        regime = "saturated"
    else:
        regime = "positive"

    reasons = []
    if gv == "FAIL":
        reasons.append("gate FAIL -> numbers void")
    elif gv == "WARN":
        reasons.append("gate WARN -> qualified")
    if df is not None and df <= 0:
        reasons.append("baseline collapsed: Δ_floor=%.2f <= 0" % df)
    if pf is False:
        reasons.append("model does not clear the mean-predictor skill floor")
    if regime == "saturated":
        reasons.append("gain over bicubic not significant (p=%.3g)" % (p if p is not None else float("nan")))
    if regime == "positive":
        reasons.append("gain significant and clears the skill floor -> genuine skill")
    return {
        "method": card.get("method") or card.get("name") or card.get("stage") or "<unnamed>",
        "task": card.get("task"),
        "n_test": card.get("n_test"),
        "split": card.get("split"),
        "gate_verdict": gv,
        "credibility": credibility,
        "regime": regime,
        "audit_gain_db": gain,
        "audit_ci": card.get("audit_ci"),
        "audit_p": p,
        "audit_d": card.get("audit_d"),
        "passes_floor": pf,
        "floor_gap_db": card.get("floor_gap_db"),
        "delta_floor_db": df,
        "gates": {k: _norm(card.get(k), k) for k in GATES},
        "reasons": reasons,
        "headline": "%s | regime=%s | badge=%s" % (
            card.get("method") or "<unnamed>", regime, credibility),
    }


# =========================================================================== #
# HTTP 服务
# =========================================================================== #
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, code=200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
            self._send_html(load_frontend())
        elif path == "/health":
            self._send_json({"status": "ok",
                             "standard": "SR-Audit Report Standard v1",
                             "derive_source": "sr_audit_report.derive" if _HAS_DERIVE else "inline"})
        elif path == "/schema":
            self._send_json(SCHEMA)
        else:
            self._send_json({"error": "not found", "available": ["/", "/health", "/schema", "/audit", "/assess"]}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except Exception as exc:
            self._send_json({"error": "invalid JSON body", "detail": str(exc)}, 400)
            return

        if path == "/audit":
            try:
                self._send_json(run_audit(payload))
            except Exception as exc:
                self._send_json({"error": "audit failed", "detail": str(exc)}, 400)
        elif path == "/assess":
            try:
                self._send_json(run_assess(payload))
            except Exception as exc:
                self._send_json({"error": "assess failed", "detail": str(exc)}, 400)
        else:
            self._send_json({"error": "not found", "available": ["/audit", "/assess"]}, 404)


# --------------------------------------------------------------------------- #
# 前端示例页（GET / 返回，亦作为 code/service/frontend_example.html 单独交付）
# --------------------------------------------------------------------------- #
def load_frontend():
    here = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(here, "frontend_example.html")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SR-Audit Service</h1><p>前端示例页 frontend_example.html 未找到。</p>"


SCHEMA = {
    "standard": "SR-Audit Report Standard v1",
    "endpoint": "POST /audit",
    "request": {
        "method": "string (方法名)",
        "task": "string (可选, 任务描述)",
        "scale": "number (可选, 超分倍率)",
        "val_dr": "number (可选, 数据动态范围/归一化系数)",
        "split": "string (可选, 如 'val')",
        "scenes": [
            {
                "psnr_model": "number 必填 (模型 PSNR dB)",
                "psnr_bicubic": "number 必填 (双三次基线 PSNR dB)",
                "psnr_mean": "number 必填 (均值预测器地板 PSNR dB)",
                "cc_bicubic": "number 可选 (CC(bicubic,GT)) -> G1",
                "cc_model": "number 可选 (CC(model,GT)) -> G2",
                "std_model": "number 可选 (模型像素标准差) -> G3",
                "std_gt": "number 可选 (GT 像素标准差) -> G3",
                "std_ratio": "number 可选 (预计算的 std_model/std_gt) -> G3",
                "bias_abs": "number 可选 (|mean(pred)-mean(gt)|) -> G3 细节",
                "grid_ratio_model": "number 可选 (FFT 最强谐波比值) -> G4",
                "grid_period_model": "number 可选 (伪影周期 px) -> G4",
                "leak_stage": "bool 可选 (HR 是否进网络) -> G5",
            }
        ],
    },
    "response": {
        "stats": "PSNR 三类预测器均值 / Δ_floor / floor_gap / gain / 配对 t 检验(ci,p,cohen_d)",
        "gates": "g1..g5 的 verdict(PASS/WARN/FAIL/n-a)+detail",
        "audit": "derive() 结果: credibility 徽章 + regime + reasons + headline",
    },
    "assess_endpoint": "POST /assess （直接传审计卡片，等价 sr_audit_report.py）",
}


def main():
    ap = argparse.ArgumentParser(description="SR-Audit 协议 HTTP 服务（前端可调用）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("SR-Audit Service 已启动: http://%s:%d/" % (args.host, args.port))
    print("  POST /audit  从逐场景证据计算门控+徽章+regime")
    print("  POST /assess 对已存在卡片跑 derive（等价 sr_audit_report.py）")
    print("  GET  /       前端调用示例页")
    print("  Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] 服务已停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
