# -*- coding: utf-8 -*-
"""
生成发布清单（RELEASE_MANIFEST）：把 experiments/ 的产物按发布层级分类，
输出体积、SHA256 与生成命令，供 GitHub 入库 / Zenodo 归档决策使用。

纯 stdlib（os/hashlib/json/re/glob），无需 torch/numpy。

层级定义（与 docs/experiments_产物发布策略.md 一致）：
  A 入库 GitHub   : 结果 JSON + 图件（小体积、文本可 diff）
  B Zenodo 归档   : 只取每个 stage 的**末轮**权重（单文件 ~19.5 MiB）
  C 不发布        : 逐 epoch 权重、被取代的旧 run
  D 禁止发布      : 原始影像（许可限制）

用法:
  python code/tools/release_manifest.py            # 写 experiments/RELEASE_MANIFEST.md + .json
  python code/tools/release_manifest.py --no-hash  # 跳过 SHA256（更快）
"""
import argparse
import glob
import hashlib
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")

# 参与发布的权重根目录（其下自动发现含 ckpt_ep*.pt 的 stage 目录）
# 说明：stage 目录层级不统一（如 ckpts_worldstrat/mtxa_aligned 与
# ckpts_probav_n300_s42/mtxa_aligned），故不硬编码，改为自动发现。
RELEASE_CKPT_ROOTS = [
    "ckpts_worldstrat",
    "ckpts_probav",
    "ckpts_probav_n300_s42",
    "ckpts_probav_n300_s7",
    "ckpts_probav_n300_s123",
    "ckpts_s2v",
]
# 明确排除（被取代的旧 run / 逐 epoch 平铺目录）
EXCLUDE_DIRS = {"ckpts"}

EP_RE = re.compile(r"ckpt_ep(\d+)\.pt$")


def discover_ckpt_dirs():
    """扫描 RELEASE_CKPT_ROOTS，返回 [(相对 ROOT 的 stage 目录, 该目录下的 ckpt 数)]。"""
    found = []
    for root in RELEASE_CKPT_ROOTS:
        base = os.path.join(EXP, root)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
            n = sum(1 for fn in fns if EP_RE.search(fn))
            if n:
                found.append((_rel(dp), n))
    # 兜底：若上面一个都没找到，全局扫一遍（仍排除 ckpts/）
    if not found:
        for dp, dns, fns in os.walk(EXP):
            dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
            n = sum(1 for fn in fns if EP_RE.search(fn))
            if n:
                found.append((_rel(dp), n))
    return sorted(set(found))


def _rel(p):
    """统一用正斜杠输出（跨平台、便于粘贴到 README）。"""
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return "%.1f %s" % (n, u) if u != "B" else "%d B" % n
        n /= 1024.0


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_size(path):
    t = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if os.path.exists(fp):
                t += os.path.getsize(fp)
    return t


def last_epoch_ckpt(d):
    """返回 (末轮文件路径, epoch 号)，无 ckpt 则 (None, None)。"""
    best, best_ep = None, -1
    for fn in os.listdir(d):
        m = EP_RE.search(fn)
        if m:
            ep = int(m.group(1))
            if ep > best_ep:
                best, best_ep = os.path.join(d, fn), ep
    return best, (best_ep if best_ep >= 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true", help="跳过 SHA256 计算")
    ap.add_argument("--out", default=os.path.join(EXP, "RELEASE_MANIFEST.md"))
    args = ap.parse_args()

    # ---- Tier A: 结果 JSON ----
    tier_a = []
    for pat in ("*.json", "sota_review/*.json", "sen2sr_audit/*.json",
                "figs_final/*.pdf", "figs_final/*.png"):
        for p in sorted(glob.glob(os.path.join(EXP, pat))):
            tier_a.append((_rel(p), os.path.getsize(p)))
    # 排除明显属于中间态的 raw 产物
    tier_a = [x for x in tier_a if not x[0].endswith(("_raw.json", "release_manifest.json"))]

    # ---- Tier B: 发布权重（每 stage 只取末轮）----
    tier_b, total_b = [], 0
    for rel, n_ep in discover_ckpt_dirs():
        d = os.path.join(ROOT, rel)
        f, ep = last_epoch_ckpt(d)
        if f is None:
            tier_b.append((rel, None, None, None, "无 ckpt"))
            continue
        size = os.path.getsize(f)
        total_b += size
        tier_b.append((rel, ep, size, f, "ok"))

    # ---- Tier C: 其余 ckpt（不发布）----
    ckpt_roots = [p for p in glob.glob(os.path.join(EXP, "ckpts*")) if os.path.isdir(p)]
    tier_c = [(_rel(p), dir_size(p)) for p in ckpt_roots]
    total_c = sum(s for _, s in tier_c)

    # ---- SHA256 ----
    hashes = {}
    if not args.no_hash:
        for rel, size in tier_a:
            hashes[rel] = sha256(os.path.join(ROOT, rel))
        for rel, ep, size, f, st in tier_b:
            if f:
                hashes[rel] = sha256(f)

    # ---- 输出 ----
    lines = []
    lines.append("# RELEASE_MANIFEST — experiments/ 产物发布清单")
    lines.append("")
    lines.append("> 自动生成：`python code/tools/release_manifest.py`。层级定义见 `docs/experiments_产物发布策略.md`。")
    lines.append("> Tier A = 入库 GitHub；Tier B = Zenodo 归档（仅末轮权重）；Tier C = 不发布；Tier D = 禁止发布（原始影像，许可限制）。")
    lines.append("")
    lines.append("## Tier A — 结果记录与图件（入库 GitHub，可 diff）")
    lines.append("")
    lines.append("| 文件 | 体积 | SHA256 |")
    lines.append("|---|---|---|")
    for rel, size in tier_a:
        lines.append("| `%s` | %s | `%s` |" % (rel, human(size), hashes.get(rel, "-")[:16] + ("…" if rel in hashes else "")))
    lines.append("")
    lines.append("**Tier A 合计：%s（%d 个文件）**" % (human(sum(s for _, s in tier_a)), len(tier_a)))
    lines.append("")
    lines.append("## Tier B — 发布权重（仅每 stage 末轮，Zenodo）")
    lines.append("")
    lines.append("| stage 目录 | 末轮 epoch | 体积 | SHA256 |")
    lines.append("|---|---|---|---|")
    for rel, ep, size, f, st in tier_b:
        if st != "ok":
            lines.append("| `%s` | - | - | %s |" % (rel, st))
        else:
            lines.append("| `%s` | %d | %s | `%s` |" % (rel, ep, human(size), hashes.get(rel, "-")[:16] + ("…" if rel in hashes else "")))
    lines.append("")
    lines.append("**Tier B 合计：%s（发布 %d 个权重文件；原始目录含全部 epoch 共 %s）**"
                 % (human(total_b), sum(1 for r in tier_b if r[4] == "ok"), human(total_c)))
    lines.append("")
    lines.append("> ⚠ 逐 epoch 权重（`ckpt_ep0..N`）合计数十 GB，**不发布**；只发布末轮。发布前用 "
                 "`train.py` 的评测口径核对该末轮即为论文所用口径（本项目主口径 = 末轮、不做验证集择优）。")
    lines.append("")
    lines.append("## Tier C — 本地保留，不发布")
    lines.append("")
    lines.append("| 目录 | 体积 |")
    lines.append("|---|---|")
    for rel, size in sorted(tier_c, key=lambda x: -x[1]):
        lines.append("| `%s` | %s |" % (rel, human(size)))
    lines.append("")
    lines.append("## Tier D — 禁止发布（许可）")
    lines.append("")
    lines.append("| 内容 | 原因 |")
    lines.append("|---|---|")
    lines.append("| WorldStrat HR（SPOT 6/7 1.5 m） | Airbus 影像为 **CC BY-NC**，不得再分发 |")
    lines.append("| SEN2VENµS HR（VENµS 5 m，Theia） | **CC BY-NC 4.0**，不得再分发 |")
    lines.append("| 由上述 HR 裁出的任何裁剪图/patch | 派生自受限影像，同样不得再分发 |")
    lines.append("| 逐样本 PSNR 记录（Tier A 的 `paired_*.json`） | ✅ **可发布**：纯数值统计，不含像素 |")
    lines.append("")
    lines.append("## 发布前自检")
    lines.append("")
    lines.append("- [ ] Tier A 全部文件已 `git add -f`（`experiments/` 在 .gitignore 内）")
    lines.append("- [ ] Tier B 权重已确认与论文口径一致（末轮），并写入 README 的下载说明")
    lines.append("- [ ] Zenodo 归档包已计算整体 SHA256 并回填 `docs/数据与代码可用性声明.md` 的占位 DOI")
    lines.append("- [ ] 包内不含任何 Tier D 影像或由其裁出的图件")
    lines.append("- [ ] README_audit.md 的零训练复现命令在干净环境实测通过")
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    summary = {
        "tier_a": {"n": len(tier_a), "bytes": sum(s for _, s in tier_a)},
        "tier_b": {"n": sum(1 for r in tier_b if r[4] == "ok"), "bytes": total_b,
                   "stages": [{"dir": r[0], "epoch": r[1], "bytes": r[2], "status": r[4]} for r in tier_b]},
        "tier_c": {"dirs": tier_c, "bytes": total_c},
        "hashes": hashes,
    }
    with open(os.path.join(EXP, "release_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("[ok] Tier A: %d files, %s" % (len(tier_a), human(sum(s for _, s in tier_a))))
    print("[ok] Tier B: %d weights, %s" % (summary["tier_b"]["n"], human(total_b)))
    print("[ok] Tier C: %d dirs, %s" % (len(tier_c), human(total_c)))
    print("[ok] -> %s" % args.out)


if __name__ == "__main__":
    main()
