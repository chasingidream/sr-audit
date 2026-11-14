#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOTA 双协议复审注册表与对照表引擎 (Task #24, Nature 计划书 P2 阶段)
================================================================

仅依赖标准库。用途:
  1. 持有已验证的 SOTA 候选元数据 (SOTA_REGISTRY)
  2. render_registry()  -> 打印候选注册表 Markdown
  3. DualProtocolReport -> 摄入每个方法
       原协议指标 (orig_*)
       审计协议 (G1-G5 + 逐样本 PSNR 配对 t 检验 + 技能地板)
     输出 "增益缩水对照表"
  4. main:
       python code/sota_review_registry.py
           打印注册表 + 空模板
       python code/sota_review_registry.py --fill experiments/sota_review
           若 <dir>/<method>.json 存在则填充真实结果 (否则留空)

双协议定义见 docs/Task24_SOTA复审.md:
  - 原协议 (as-published): 方法论文报告的增益 (通常合成 bicubic + 组均值, 无技能地板/配对检验)
  - 审计协议 (ours): SR-Audit G1-G5 + 逐样本 PSNR 平均 + 配对 t 检验 (CI/p/Cohen's d) + 技能地板
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# ----------------------------------------------------------------------------
# 1. SOTA 候选注册表 (已联网核实的元数据; A2WNet 标为待核实替换)
# ----------------------------------------------------------------------------
@dataclass
class SotaEntry:
    name: str
    venue: str
    year: int
    task: str                       # MFSR / SISR / 跨卫星对齐 / 评测基准 ...
    data_scale_cross: str           # 数据·尺度·是否跨传感器
    code_url: str
    weights_url: str
    axis_map: str                   # 与本项目三轴 (PROBA-V/SEN2VENµS/WorldStrat) 的映射
    priority: str                   # P0 必跑 / P1 参考 / P2 / ⚠ 待核实
    notes: str


SOTA_REGISTRY: list[SotaEntry] = [
    SotaEntry(
        name="HighRes-net",
        venue="WorldStrat, NeurIPS 2022 (Datasets & Benchmarks)",
        year=2022,
        task="MFSR (多帧超分)",
        data_scale_cross="S2 10m -> SPOT 1.5m; 6.63x; 跨传感器; 多帧(16 revisits)",
        code_url="https://github.com/worldstrat/worldstrat",
        weights_url="https://github.com/worldstrat/worldstrat (BSD, 预训练发布)",
        axis_map="WorldStrat 轴直接基线 (本项目已用同数据)",
        priority="P0",
        notes="官方即 WorldStrat 基线, 最易公平重跑; 预期受 6.63x 尺度信息缺口拖累",
    ),
    SotaEntry(
        name="EDSR",
        venue="CVPR 2017 (Lim et al., Enhanced Deep Residual Networks)",
        year=2017,
        task="单图 SISR (SISR)",
        data_scale_cross="Set5/Set14/Urban100 合成 bicubic 退化 (2-4x); 通用 SISR",
        code_url="https://github.com/LimBee/NTIRE2017",
        weights_url="https://github.com/LimBee/NTIRE2017",
        axis_map="WorldStrat 轴单图 SOTA 对照 (本项目已在流水线独立复现, frame0)",
        priority="P0 (已独立复现)",
        notes="单图强 SOTA 基线; 本项目从规范独立实现 (16 blocks/64 feats, PixelShuffle x7->resize 424), 在 WorldStrat 流水线 frame0 单帧训练, 与 bicubic(frame0) 正交对比; 用于检验'即便强单图 SOTA 在 6.63x 跨传感器下是否救不回可恢复信息缺口'. 原协议增益(合成 bicubic 2-4x)与本真实配对不可直接比, 故 orig_gain_db 留空, 仅报审计协议配对增益.",
    ),
    SotaEntry(
        name="WDSR",
        venue="arXiv 2018 / ICLR 2019 (Yu et al., Wide Activation for Efficient and Accurate Image SR)",
        year=2018,
        task="单图 SISR (SISR)",
        data_scale_cross="Set5/Set14/Urban100/B100 合成 bicubic 退化 (2-4x); 通用 SISR",
        code_url="https://github.com/JiahuiYu/wdsr_ntire2018",
        weights_url="https://github.com/JiahuiYu/wdsr_ntire2018",
        axis_map="WorldStrat 轴单图 SOTA 对照 (本项目已在流水线独立复现, frame0)",
        priority="P0 (已独立复现)",
        notes="单图强 SOTA 基线 (wide activation 路线, 与 EDSR 深残差路线并行); 本项目从规范独立实现 WDSR-a (16 blocks/32 feats/expand4/res_scale0.1), 在 WorldStrat 流水线 frame0 单帧训练, 与 bicubic(frame0) 正交对比; 与 EDSR 共同检验'强单图 SOTA 在 6.63x 跨传感器下是否救不回可恢复信息缺口'(跨架构稳健性). 原协议增益(合成 bicubic 2-4x)与本真实配对不可直接比, 故 orig_gain_db 留空.",
    ),
    SotaEntry(
        name="RAMS",
        venue="Remote Sensing 2020, 12(14):2207 (Salvetti, Mazzia, Khaliq, Chiaberge)",
        year=2020,
        task="MFSR (3D conv + 残差注意力 RFAB/RTAB)",
        data_scale_cross="PROBA-V 原生: 单波段(NIR/RED) 300m->100m, 3x, T=9 多帧; 官方权重仅 NIR/RED 两套",
        code_url="https://github.com/EscVM/RAMS",
        weights_url="https://github.com/EscVM/RAMS (TF2 ckpt/NIR_RAMS, ckpt/RED_RAMS)",
        axis_map="PROBA-V 轴多帧对照 (本项目已独立复现, 审计协议正向对照)",
        priority="P0 (已独立复现)",
        notes="官方为 TensorFlow2 + tensorflow_addons(tfa 已停止维护, 本机无 TF); 且 utils/network.py 全局支路写死 RTAB(...,9,...)、输出 1 通道 -> 结构上与 PROBA-V 单波段 3x/T=9 绑定, 无法在不偷改结构前提下搬到 WorldStrat 6.63x RGB。故按 Task24 3.2 '无权重则严格按官方配置短训' 路径, 从 network.py 规范**独立 PyTorch 复现**(32f/3D+RFAB x12+时相降维+2D RTAB 全局支路, weight norm, T=9), 在【本项目 PROBA-V 轴】(同传感器+多帧+3x) 审计。该轴为本项目'真有增益'轴(model +2.33 dB) -> RAMS 条目构成审计协议的**正向对照**(证明 SR-Audit 非'一切皆杀'的锤子)。原协议增益(PROBA-V 自报 PSNR, 单波段+官方归一化)与本审计口径不可直接比, 故 orig_gain_db 留空。",
    ),
    SotaEntry(
        name="STANet",
        venue="ISPRS J Photogramm Remote Sens 2026, 232 (doi 10.1016/j.isprsjprs.2025.12.019)",
        year=2026,
        task="弱监督跨卫星时空对齐 SR (SAM+TAM)",
        data_scale_cross="跨卫星 Landsat-8; 时空未对齐",
        code_url="无 (2026-09-17 复核: Elsevier DOI 页 + 武大学者页 均无代码链接)",
        weights_url="无",
        axis_map="最直接竞品: 跨卫星未对齐自对齐, 与本项目 cross-attn 对齐同一战场",
        priority="NO-GO -> 降级 P1 参考 (定性引用, 不重跑)",
        notes=(
            "【NO-GO, 2026-09-17 复核】无公开代码/权重 -> 无法按审计协议重跑。"
            "证据: (1) 论文 (Guo, Liu, Fang, Gao, Chen, Tian; 武汉大学; ISPRS J 2026, 232:524-541) "
            "在 Elsevier DOI 与 scholars.whu.edu.cn 出版记录中均无 code/data 链接; "
            "(2) ⚠ 严重同名坑: GitHub 上的 'STANet' (justchenhao/STANet, renmada/STANET-Paddle-lite) "
            "是**遥感变化检测**的 Spatial-Temporal Attention Network (Chen & Shi, 另一篇论文), "
            "与本 SR STANet 无关; 检索者极易误取。"
            "处理: 按 Task24 3.2 '无权重则用其报告数字 + 我们的审计框架概念套用' -> "
            "降级为 P1 定性参考: 引用其 RS 2025/26 竞品定位与其自述真实失准设定, "
            "但不纳入增益缩水表的量化行 (与 EDiffSR 'not reproducible under audit' 同处理)。"
        ),
    ),
    SotaEntry(
        name="EDiffSR",
        venue="IEEE TGRS 2024, 62 (doi 10.1109/TGRS.2023.3341437)",
        year=2024,
        task="扩散 SISR (EANet + CPEM)",
        data_scale_cross="AID/DOTA/DIOR/NWPU-RESISC45 (合成 bicubic 退化)",
        code_url="https://github.com/CANIBBER/EDiffSR",
        weights_url="https://github.com/CANIBBER/EDiffSR",
        axis_map="synthetic-vs-real 对照锚点: 原协议在合成退化上跑",
        priority="P0",
        notes="292 引用影响力大; 预期真实跨传感器下增益大幅缩水",
    ),
    SotaEntry(
        name="SEN2SR",
        venue="Remote Sensing of Environment 2026, 334:115222 (doi 10.1016/j.rse.2025.115222)",
        year=2026,
        task="S2 -> 2.5m 单图 (低频硬约束 + Wald)",
        data_scale_cross="S2 10/20m -> 2.5m; 单图; SEN2NAIP 合成",
        code_url="https://github.com/ESAOpenSR/SEN2SR",
        weights_url="https://github.com/ESAOpenSR/SEN2SR",
        axis_map="已量化(2026-10-05 分布外 / 2026-10-06 原生分布): 两处都无真技能——分布外『输』, 原生分布『平』",
        priority="P1 (已复审并量化; 含原生分布复核)",
        notes="发布权重直跑两种分布, 未训练/未微调。① 分布外 WorldStrat 原生 x4 (128 RGBN -> 512), n=60: Δ_model=-0.051 dB (p=1.7e-06, 97% 为负), Δ_floor=-2.53 (该站点本身 floor-failure), G4 FAIL (周期比 5.50, 主导周期=4=PixelShuffle 倍数)。② 原生分布 SEN2NAIPv2 crosssensor x4, n=120 (两个独立子样本): Δ_model=+0.034 dB (p=0.068, CI 跨 0), Δ_floor=+14.64 ⇒ 只与 bicubic 打平(saturated), 仅吃到 0.23% 的可恢复余量; 全闸 PASS, G4=1.39@4px。⇒ 同一权重从 Δ_floor 轴两端到达同一终点。硬约束消融跨域同号(裸网络 -0.80/-0.65 dB) ⇒ 建设性约束修的是辐射保真(G3/G5), 非技能; G4 主导周期两处都=4 但幅度被域漂移放大 ~4 倍 ⇒ 周期闸兼作域漂移探测器。定性→定量完成; 见 experiments/sota_review/sen2sr.json + sen2sr_native.json, docs/SEN2SR审计_2026-10-05.md + docs/SEN2NAIPv2原生复核_2026-10-06.md",
    ),
    SotaEntry(
        name="OpenSR-Test / SEN2NAIP",
        venue="Nature Scientific Data 2024, 11:1389 (SEN2NAIP); ESA Phi-lab",
        year=2024,
        task="评测基准 (Consistency/Synthesis/Correctness 三性质)",
        data_scale_cross="S2 评测基准",
        code_url="https://github.com/ESAOpenSR",
        weights_url="-",
        axis_map="评测协议参照 (Wald + 三性质)",
        priority="P1",
        notes="审计协议应显式引用并对比其三性质",
    ),
    SotaEntry(
        name="RealFusion",
        venue="Remote Sensing of Environment 2025",
        year=2025,
        task="Landsat+MODIS 时空融合 (抗剧变/云)",
        data_scale_cross="跨传感器时序融合",
        code_url="待确认",
        weights_url="待确认",
        axis_map="跨传感器时序融合参照",
        priority="P1",
        notes="偏融合非纯 SR, 作背景引用",
    ),
    SotaEntry(
        name="DiffFuSR",
        venue="arXiv 2506.11764 (2025.06)",
        year=2025,
        task="扩散全波段 S2 -> 2.5m",
        data_scale_cross="S2 -> 2.5m; 扩散",
        code_url="https://arxiv.org/abs/2506.11764",
        weights_url="待确认",
        axis_map="2025 扩散锚点",
        priority="P1",
        notes="视算力补跑",
    ),
    SotaEntry(
        name="RareFlow",
        venue="arXiv 2510.23816 (2025.10)",
        year=2025,
        task="跨传感器 flow matching",
        data_scale_cross="跨传感器",
        code_url="https://arxiv.org/abs/2510.23816",
        weights_url="待确认",
        axis_map="跨传感器对照",
        priority="P1",
        notes="视算力补跑",
    ),
    SotaEntry(
        name="4KAgent",
        venue="2025.07",
        year=2025,
        task="agent-based SR",
        data_scale_cross="待补",
        code_url="-",
        weights_url="-",
        axis_map="2025 代表作",
        priority="P2",
        notes="信息待补",
    ),
    SotaEntry(
        name="RWSRGAN",
        venue="2025 Book Chapter",
        year=2025,
        task="GAN SR",
        data_scale_cross="待补",
        code_url="-",
        weights_url="-",
        axis_map="2025 GAN 锚点",
        priority="P2",
        notes="信息待补",
    ),
    SotaEntry(
        name="A2WNet (⚠ 待核实替换)",
        venue="检索显示 'Adaptive Attention Weather Network' (ICDAR 2026) 为天气图像识别, 非 RS-SR",
        year=2026,
        task="⚠ 非 RS-SR 方法",
        data_scale_cross="⚠ 不适用",
        code_url="-",
        weights_url="-",
        axis_map="⚠ 计划书原文疑似误标",
        priority="⚠ 建议替换",
        notes="建议替换为 STANet(已含) 或 RareFlow/DiffFuSR 之一, 以补齐跨传感器 2025-26 覆盖",
    ),
]


# ----------------------------------------------------------------------------
# 2. 双协议复审对照表引擎
# ----------------------------------------------------------------------------
@dataclass
class ProtocolResult:
    """单个方法在双协议下的复审结果。所有增益单位 dB。"""
    method: str
    orig_gain_db: Optional[float] = None          # 原协议报告增益 (model - bicubic, 组均值)
    audit_gain_db: Optional[float] = None          # 审计协议 逐样本配对增益 (model - bicubic)
    audit_ci: tuple = (None, None)                 # 95% CI (lo, hi)
    audit_p: Optional[float] = None                # 配对 t 检验 p 值
    audit_d: Optional[float] = None                # Cohen's d
    passes_floor: Optional[bool] = None            # model 是否显著越过均值地板
    g1: str = "-"; g2: str = "-"; g3: str = "-"; g4: str = "-"; g5: str = "-"
    note: str = ""


class DualProtocolReport:
    """汇总多方法的双协议复审，输出增益缩水对照表。"""

    def __init__(self) -> None:
        self.rows: list[ProtocolResult] = []

    def add(self, r: ProtocolResult) -> None:
        self.rows.append(r)

    @staticmethod
    def _shrinkage(r: ProtocolResult) -> str:
        """量化增益缩水。"""
        if r.orig_gain_db is None:
            return "—"
        if r.audit_gain_db is None:
            return "未测"
        if r.audit_p is not None and r.audit_p >= 0.05:
            return "100% 消失 (不显著)"
        delta = r.orig_gain_db - r.audit_gain_db
        return f"{delta:+.2f} dB"

    @staticmethod
    def _fmt(x, nd=2, dash="—") -> str:
        return dash if x is None else f"{x:.{nd}f}"

    @staticmethod
    def _fmt_ci(ci) -> str:
        # 防御: None / 非序列 / 长度!=2 / 含 None 一律降级为占位符 (避免 unpack 崩溃)
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            return "—"
        lo, hi = ci
        if lo is None or hi is None:
            return "—"
        return f"[{lo:+.2f},{hi:+.2f}]"

    def render_markdown(self) -> str:
        head = ("| 方法 | 原协议增益(dB) | 审计协议配对增益(dB) | 95%CI | p | d | "
                "过地板? | G1 | G2 | G3 | G4 | G5 | 缩水 |\n")
        sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        lines = [head, sep]
        for r in self.rows:
            floor = ("—" if r.passes_floor is None
                     else ("✓" if r.passes_floor else "✗"))
            lines.append(
                f"| {r.method} "
                f"| {self._fmt(r.orig_gain_db)} "
                f"| {self._fmt(r.audit_gain_db)} "
                f"| {self._fmt_ci(r.audit_ci)} "
                f"| {self._fmt(r.audit_p, 4)} "
                f"| {self._fmt(r.audit_d)} "
                f"| {floor} "
                f"| {r.g1} | {r.g2} | {r.g3} | {r.g4} | {r.g5} "
                f"| {self._shrinkage(r)} "
                f"|"
            )
            if r.note:
                lines.append(f"| | | | | | | | | | | | | | *{r.note}* |")
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.rows], f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------
# 3. 渲染候选注册表
# ----------------------------------------------------------------------------
def render_registry() -> str:
    head = ("| # | 方法 | 出处 | 任务 / 数据 / 尺度 / 跨传感器 | 开源 | "
            "三轴映射 | 优先级 | 备注 |\n")
    sep = "|---|---|---|---|---|---|---|---|\n"
    lines = [head, sep]
    for i, e in enumerate(SOTA_REGISTRY, 1):
        lines.append(
            f"| {i} | **{e.name}** | {e.venue} | {e.task}; {e.data_scale_cross} | "
            f"[code]({e.code_url}) | {e.axis_map} | {e.priority} | {e.notes} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 4. 从 experiments/sota_review/<method>.json 填充 (每个文件即 ProtocolResult 字段)
# ----------------------------------------------------------------------------
EXPECTED_FIELDS = ["method", "orig_gain_db", "audit_gain_db", "audit_ci",
                  "audit_p", "audit_d", "passes_floor",
                  "g1", "g2", "g3", "g4", "g5", "note"]


def load_results(folder: str) -> DualProtocolReport:
    rep = DualProtocolReport()
    if not os.path.isdir(folder):
        print(f"[warn] 结果目录不存在: {folder} -> 输出空模板")
        return rep
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(folder, fn), "r", encoding="utf-8") as f:
            d = json.load(f)
        # 只接受 ProtocolResult 形状的 json (必须含非空 "method" 键); 其余(原始指标快照
        # 如 ediffsr_raw.json / 嵌套 dict / 无 method 的文件)一律跳过, 否则会产生
        # method=None 的空行并使 audit_ci=None 在渲染时崩溃。
        if not isinstance(d, dict) or not d.get("method"):
            print(f"[skip] {fn}: 非 ProtocolResult 形状 (缺 method), 已跳过")
            continue
        rep.add(ProtocolResult(**{k: d.get(k) for k in EXPECTED_FIELDS}))
    return rep


# ----------------------------------------------------------------------------
# 5. main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="SOTA 双协议复审注册表/对照表引擎 (Task #24)")
    ap.add_argument("--fill", default=None,
                    help="experiments/sota_review 目录, 若存在 <method>.json 则填充真实结果")
    args = ap.parse_args()

    print("# SOTA 候选注册表 (Task #24)\n")
    print(render_registry())
    print()

    print("# 增益缩水对照表 (双协议复审)\n")
    if args.fill:
        rep = load_results(args.fill)
        if rep.rows:
            print(rep.render_markdown())
            out = os.path.join(args.fill, "_shrinkage_report.md")
            with open(out, "w", encoding="utf-8") as f:
                f.write("# 增益缩水对照表\n\n" + rep.render_markdown() + "\n")
            print(f"\n[ok] 已写入 {out}")
        else:
            print(rep.render_markdown())
            print("\n[warn] 未找到任何 <method>.json, 输出空模板。")
    else:
        # 空模板: 列出 P0 方法占位
        rep = DualProtocolReport()
        for e in SOTA_REGISTRY:
            if e.priority == "P0":
                rep.add(ProtocolResult(method=e.name,
                                       note="待实跑; 填 experiments/sota_review/<method>.json"))
        print(rep.render_markdown())
        print("\n[info] 实跑后用: "
              "python code/sota_review_registry.py --fill experiments/sota_review")


if __name__ == "__main__":
    main()
