#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SR-Audit 服务 · Python 客户端调用示例
====================================

演示如何把某个超分方法的逐场景评测证据发送给 SR-Audit 服务，
并读取返回的可信徽章（credibility）与科学 regime。

运行前请先启动服务：
    python code/service/audit_service.py --port 8765

本示例只用标准库 urllib（无需额外安装 requests）。
若已安装 requests，可改用文件末尾注释里的等价写法。
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8765"


def audit(payload):
    """调用 POST /audit 并返回解析后的 JSON。"""
    req = urllib.request.Request(
        BASE + "/audit",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    # 1) 构造评测证据（这里用一组会触发 floor-failure 的示例数据，
    #    实际使用时替换为你的模型在验证集上的逐场景指标）。
    payload = {
        "method": "CrossAttnAlign (mtxa_mis_xa)",
        "task": "WorldStrat 6.63x cross-sensor, misaligned, 5-frame",
        "scale": 7,
        "val_dr": 15.06,
        "split": "val",
        "scenes": [
            {"psnr_model": 29.38, "psnr_bicubic": 30.82, "psnr_mean": 33.08,
             "cc_bicubic": 0.303, "cc_model": 0.231,
             "std_model": 0.18, "std_gt": 0.24, "bias_abs": 0.889,
             "grid_ratio_model": 1.05, "grid_period_model": 6},
            {"psnr_model": 28.91, "psnr_bicubic": 30.41, "psnr_mean": 32.77,
             "cc_bicubic": 0.291, "cc_model": 0.220,
             "std_model": 0.17, "std_gt": 0.23, "bias_abs": 0.912,
             "grid_ratio_model": 1.11, "grid_period_model": 6},
            {"psnr_model": 30.02, "psnr_bicubic": 31.10, "psnr_mean": 33.45,
             "cc_bicubic": 0.312, "cc_model": 0.244,
             "std_model": 0.19, "std_gt": 0.25, "bias_abs": 0.851,
             "grid_ratio_model": 0.98, "grid_period_model": 6},
        ],
    }

    # 2) 调用服务
    result = audit(payload)

    # 3) 读取关键结论
    a = result["audit"]
    s = result["stats"]
    print("方法        :", a["method"])
    print("可信徽章    :", a["credibility"])      # VERIFIED / QUALIFIED / VOID / UNVERIFIED
    print("科学 regime :", a["regime"])           # positive / saturated / floor-failure
    print("门控总判定  :", a["gate_verdict"])
    print("gain(model−bicubic):", s["gain_db"], "dB")
    print("Δ_floor(bicubic−mean):", s["delta_floor_db"], "dB")
    print("配对 t 检验 : p =", s["paired"]["p"], "  Cohen's d =", s["paired"]["cohen_d"])
    print("门控        :", {k: v["verdict"] for k, v in result["gates"].items()})
    print("理由        :", a["reasons"])
    print("headline    :", a["headline"])


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# 若已安装 requests，以上 audit() 可替换为：
#
#   import requests
#   r = requests.post(BASE + "/audit", json=payload)
#   result = r.json()
#
# 也可用 curl 直接试：
#   curl -X POST http://127.0.0.1:8765/audit \
#        -H "Content-Type: application/json" \
#        -d @code/service/example_request.json
# --------------------------------------------------------------------------- #
