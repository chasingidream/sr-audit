#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""audit_attack_surface.py —— 攻击面逐项确认（覆盖率核查）。

对 `docs/CEE_审稿人视角复评_2026-10-07.md` §七 列出的每条攻击面（R1–R16），
检查当前 CEE 主文中是否存在对应的**防守条款**，并确认已同步到 ISPRSJ 镜像。

三类结果：
  PASS   —— 该攻击面的全部防守短语都出现在 CEE 主文中（ISPRSJ 亦同步）
  CEE-ONLY —— CEE 主文有，但 ISPRSJ 镜像缺（传播缺口）
  MISSING —— CEE 主文缺该防守短语（需补写）

用法（项目解释器）：
    python code/tools/audit_attack_surface.py
退出码：全 PASS 为 0，否则 1。
"""
from __future__ import annotations
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CEE = os.path.join(BASE, "docs", "主文_CEE_submission.md")
ISPRSJ = os.path.join(BASE, "docs", "主文_ISPRSJ_submission.md")


def norm(text: str) -> str:
    """去 Markdown 强调标记 + 统一破折号，便于跨格式子串匹配。"""
    t = text.replace("**", "").replace("*", "").replace("`", "").replace("_", "")
    for d in ("–", "—", "‑", "−"):
        t = t.replace(d, "-")
    t = re.sub(r"\s+", " ", t)
    return t


# (id, 攻击描述, 严重度, [防守短语...])
SURFACES = [
    ("R15", "图集映射 Δ_floor（决策输入）而非 Δ_model（判决）", "中",
     ["it maps the recoverable budget", "the decision's input", "not the regime verdict"]),
    ("R12", "同区域两站点判决相反 ⇒ 判决是站点噪声", "中",
     ["property of the acquisition setting", "not of the dataset or region",
      "opposite verdicts", "65 km apart"]),
    ("R2", "轴太少 / 单数据集承载跨站点泛化（已补第二数据集站点面板）", "中-低",
     ["The trained exemplars are few even though the data-layer sample is global",
      "the positive axis is one dataset",
      "tested on a second dataset without retraining",
      "returns the opposite composition",
      "five held-out WorldStrat stations are floor-failure"]),
    ("R1", "本质是 NSE 换皮", "中",
     ["Relation to classical skill scores", "We inherit that criterion deliberately"]),
    ("R3", "只认证 MSE，无感知指标", "中",
     ["does not directly certify perceptual gains"]),
    ("R13", "positive 类横跨 7.7%–67%，不紧", "低-中",
     ["The label positive therefore asserts only", "span 7.7%-67%"]),
    ("R11", "唯一同倍率真实跨传感器臂不显著", "低",
     ["return two null results and three significant gains"]),
    ("R8/R9", "saturated 是 null / β 阈值是调出来的", "低",
     ["No arm lands between 1.8% and 7.7%", "saturation states a magnitude, not a loss of power"]),
    ("R5", "G1 = 0.303 贴门槛", "低",
     ["fix G1 at 0.30 and do not tune it"]),
    ("R7", "SEN2SR 两行在挑分布", "低",
     ["ties the bicubic baseline in its own distribution"]),
    ("R10", "SEN2NAIPv2 单国分布", "低",
     ["single-country domain"]),
    ("R14", "新站点来自打包镜像非原始发布", "低-中",
     ["TACO-packaged mirror of Zenodo record 14603764"]),
    ("R4", "图集该进正文（已闭合）", "已闭合",
     ["Fig. 5"]),
]


def main() -> int:
    cee = norm(open(CEE, encoding="utf-8").read())
    isp = norm(open(ISPRSJ, encoding="utf-8").read()) if os.path.exists(ISPRSJ) else ""

    rows = []
    n_pass = n_cee_only = n_missing = 0
    for sid, desc, sev, phrases in SURFACES:
        miss = [p for p in phrases if p not in cee]
        notisp = [p for p in phrases if p not in isp]
        if miss:
            status = "MISSING"
            n_missing += 1
        elif notisp:
            status = "CEE-ONLY"     # CEE 有、ISPRSJ 缺
            n_cee_only += 1
        else:
            status = "PASS"
            n_pass += 1
        rows.append((sid, desc, sev, status, miss, notisp))

    w = max(len(r[0]) for r in rows) + 1
    print("=" * 104)
    print(f"攻击面逐项确认（{CEE} → 覆盖率）")
    print("=" * 104)
    for sid, desc, sev, status, miss, notisp in rows:
        tag = {"PASS": "[PASS    ]", "CEE-ONLY": "[CEE-ONLY]", "MISSING": "[MISSING ]"}[status]
        print(f"  {tag} {sid:<{w}} sev={sev:<6} {desc[:46]}")
        if miss:
            print(f"             CEE 缺: {miss}")
        if notisp:
            print(f"             ISPRSJ 缺: {notisp}")
    print("=" * 104)
    print(f"PASS={n_pass}  CEE-ONLY={n_cee_only}  MISSING={n_missing}  合计={len(rows)}")
    print(f"R16（核查覆盖边界）为流程项，见 verify_manuscript_numbers.py B12–B16；")
    print(f"   其余 {len(rows)} 条为主文文本项。")

    # ---------------- R16 补充：主图/表引用完整性 ----------------
    raw = open(CEE, encoding="utf-8").read()
    cap_fig = set(re.findall(r"\*\*Fig\.\s*(\d+)\s*\|", raw))
    cap_tab = set(re.findall(r"\*\*Table\s*(\d+)\s*\|", raw))
    ref_fig = set(re.findall(r"Fig\.\s(\d)\b", raw))       # 正文引用（单数字 = 主图）
    ref_tab = set(re.findall(r"Table\s(\d)\b", raw))
    main_figs, main_tabs = {"1", "2", "3", "4", "5"}, {"1", "2"}
    prob = []
    for f in sorted(main_figs):
        if f not in cap_fig:
            prob.append(f"Fig. {f} 无图注")
        if f not in ref_fig:
            prob.append(f"Fig. {f} 正文未引用")
    for t in sorted(main_tabs):
        if t not in cap_tab:
            prob.append(f"Table {t} 无表题")
        if t not in ref_tab:
            prob.append(f"Table {t} 正文未引用")
    print("-" * 104)
    print(f"  R16-图/表引用完整性：主图 {sorted(main_figs)} 图注={sorted(cap_fig)}   "
          f"主表 {sorted(main_tabs)} 表题={sorted(cap_tab)}")
    if prob:
        print(f"  [INTEGRITY-WARN] " + "；".join(prob))
    else:
        print(f"  [INTEGRITY-PASS] 5 主图 + 2 表：图注/表题与正文引用一一对应")
    return 0 if (n_missing == 0 and not prob) else 1


if __name__ == "__main__":
    sys.exit(main())
