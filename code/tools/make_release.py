"""把仓库打包成一个可直接推到 GitHub 的干净发布树（Tier A only）。

为什么要这个工具
----------------
本仓库是**开发仓库**：`experiments/` 里躺着 42.7 GiB 的检查点（Tier C）、
`/data/external/` 里有 CC BY-NC 的原始影像（Tier D）、`docs/` 里混着审稿意见与
内部复审。直接在开发仓库上 `git add -f` 是高风险操作（一旦手滑就是灾难，且
`docs/` 里的东西不能公开）。

因此发布走「另起干净仓库」：本工具只按 `experiments/RELEASE_MANIFEST.md` 的
**Tier A 白名单**复制文件，并顺手做两件脱敏：

1. 文本里的本机工程路径（`...`）→ 去掉前缀，变仓库内相对路径；
2. WorldStrat 清单里的本机数据路径（`<用户主目录>/worldstrat/...`）
   → 去掉数据根，变相对路径。配套改动：`code/data/dataset.py` 的
   `_rebase_worldstrat_manifest()` 会按环境变量 `WORLDSTRAT_ROOT` 把相对路径补全，
   所以本地绝对路径清单与发布版相对路径清单都能跑。

用法
----
    python code/tools/make_release.py --dest sr-audit
    python code/tools/make_release.py --dest sr-audit --clean

`--clean` 会先清空 dest 下的 `code/ results/ data/`（只删这三个，不动 dest 根）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(BASE, "experiments")
MANIFEST_MD = os.path.join(EXP, "RELEASE_MANIFEST.md")

# 需要从文本产物中剥掉的工程内前缀（含 JSON 里被转义的双反斜杠形态）。
PROJ_TOKENS = [
    "E:\\\\WorkBuddy\\\\时空超分\\\\",
    "",
    "",
    "",
    "",
    "",
    "E:\\WorkBuddy\\时空超分",
    "",
    "",
    "",
]

# 这些扩展名才做文本脱敏；其余（png/pdf/pth/npy）按字节原样复制。
TEXT_EXT = {".json", ".jsonl", ".md", ".txt", ".py", ".yaml", ".yml", ".cfg", ".cff"}
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}

RELEASE_SRC = os.path.join(BASE, "release")   # 仓库根级文件的源（README/LICENSE/…）
ROOT_FILES = ["README.md", "README_audit.md", "ENVIRONMENT.md", "LICENSE", "CITATION.cff"]

ABS_PATH_RE = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:[\\/](?:[A-Za-z0-9_.\- ]+[\\/])+[A-Za-z0-9_.\-]*")
# 系统路径不算泄漏：vcomp140.dll 的陷阱与中文字体都是要写进 ENVIRONMENT.md 的正当内容。
SYS_PREFIX = re.compile(r"(?i)^[a-z]:[\\/](windows|program files|program files \(x86\))")


def _is_sys_path(m: str) -> bool:
    """源码里可能是 `C:\\\\Windows\\\\...`（转义形态），先归一化再判前缀。"""
    return bool(SYS_PREFIX.match(m.replace("\\\\", "\\")))


def human(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TiB"


def tier_a_files() -> list[str]:
    """从 RELEASE_MANIFEST.md 读 Tier A 白名单（自动生成的表就是权威来源）。

    表行形如 `| \\`path\\` | 1.0 KiB | \\`sha256…\\` |`，只取**第一个**反引号单元格，
    否则会把 SHA256 列也当成路径。
    """
    if not os.path.exists(MANIFEST_MD):
        raise SystemExit(f"缺少 {MANIFEST_MD}；先跑 code/tools/release_manifest.py")
    txt = open(MANIFEST_MD, encoding="utf-8").read()
    sec = txt.split("## Tier A")[1].split("## Tier B")[0]
    paths = []
    for line in sec.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        m = re.search(r"`([^`]+)`", line)
        if not m:
            continue
        p = m.group(1)
        if "/" in p or "\\" in p:      # 排除表头与 SHA256（无分隔符）
            paths.append(p)
    missing = [p for p in paths if not os.path.exists(os.path.join(BASE, p))]
    if missing:
        print(f"[warn] 清单中 {len(missing)} 个文件不存在，已跳过：{missing[:3]} …")
    return [p for p in paths if os.path.exists(os.path.join(BASE, p))]


def sanitize_text(s: str) -> tuple[str, int]:
    n = 0
    for tok in PROJ_TOKENS:
        if tok in s:
            n += s.count(tok)
            s = s.replace(tok, "")
    return s, n


def copy_file(src: str, dst: str) -> tuple[int, int]:
    """复制单文件；文本类做脱敏。返回 (字节数, 替换次数)。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = os.path.getsize(src)
    if os.path.splitext(src)[1].lower() in TEXT_EXT:
        raw = open(src, "rb").read()
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
            return size, 0
        s2, n = sanitize_text(s)
        if n:
            open(dst, "w", encoding="utf-8", newline="").write(s2)
            return size, n
    shutil.copy2(src, dst)
    return size, 0


def sanitize_worldstrat_manifest(obj) -> tuple[object, int]:
    """把清单里的 lr/hr 绝对路径改成相对于数据根的相对路径（正斜杠）。"""
    n = 0

    def rel(p):
        nonlocal n
        if not isinstance(p, str):
            return p
        s = p.replace("\\", "/")
        m = re.search(r"(?i)/(worldstrat|worldstrat-zenodo)/", s)
        if m:
            n += 1
            return s[m.end():]
        if re.match(r"^[A-Za-z]:/", s):
            n += 1
            return s.split("/", 3)[-1] if s.count("/") >= 3 else s
        return s

    out = []
    for e in obj:
        if not isinstance(e, dict):
            out.append(e)
            continue
        e = dict(e)
        for k in ("lr", "hr"):
            v = e.get(k)
            e[k] = [rel(x) for x in v] if isinstance(v, list) else rel(v)
        out.append(e)
    return out, n


def copy_processed(dest: str) -> list[tuple[str, int]]:
    """复制 data/processed 下的清单与归一化参数；两份清单做结构化脱敏。"""
    src_dir = os.path.join(BASE, "data", "processed")
    dst_dir = os.path.join(dest, "data", "processed")
    os.makedirs(dst_dir, exist_ok=True)
    report = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".json"):
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if name.startswith("worldstrat_manifest"):
            obj = json.load(open(src, encoding="utf-8"))
            obj, n = sanitize_worldstrat_manifest(obj)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            report.append((f"data/processed/{name}", n))
        else:
            shutil.copy2(src, dst)
            report.append((f"data/processed/{name}", 0))
    return report


def copy_code(dest: str) -> int:
    """复制 code/ 全集，剔除 __pycache__ 与数据软链目录。"""
    src = os.path.join(BASE, "code")
    dst = os.path.join(dest, "code")
    total = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, src).replace("\\", "/")
        if rel.startswith("data/raw"):
            continue
        for fn in files:
            if fn.endswith((".pyc", ".pyo")):
                continue
            s = os.path.join(root, fn)
            d = os.path.join(dst, rel, fn) if rel != "." else os.path.join(dst, fn)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            _, n = copy_file(s, d)
            total += n
    return total


def clean(dest: str) -> None:
    for sub in ("code", "results", "data"):
        p = os.path.join(dest, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"[clean] 删除 {p}")


def sha256(path: str, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_root_files(dest: str) -> list[str]:
    """把 release/ 下的仓库根级文件（README/LICENSE/CITATION/…）拷到发布树根。"""
    if not os.path.isdir(RELEASE_SRC):
        print(f"[warn] 缺少 {RELEASE_SRC}，跳过根级文件")
        return []
    got = []
    for fn in sorted(os.listdir(RELEASE_SRC)):
        src = os.path.join(RELEASE_SRC, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, fn))
            got.append(fn)
    missing = [f for f in ROOT_FILES if f not in got]
    if missing:
        print(f"[warn] release/ 中缺：{missing}")
    return got


def rewrite_manifest(dest: str, files: list[str]) -> None:
    """在发布树内重算 RELEASE_MANIFEST.md 的 Tier A 表。

    开发清单里的路径是 `experiments/…`、SHA 是**脱敏前**的，直接拷过去会全表对不上。
    这里按发布后的实际文件重写：路径改成 `results/…`，体积与 SHA256 取脱敏后的值。
    """
    txt = open(MANIFEST_MD, encoding="utf-8").read()
    head, rest = txt.split("## Tier A", 1)
    tail = "## Tier B" + rest.split("## Tier B", 1)[1]

    rows, total = [], 0
    for rel in files:
        sub = rel.split("experiments/", 1)[1] if rel.startswith("experiments/") else rel
        dp = os.path.join(dest, "results", sub)
        sz = os.path.getsize(dp)
        total += sz
        rows.append(f"| `results/{sub}` | {human(sz)} | `{sha256(dp)[:16]}…` |")

    block = [
        "## Tier A — 结果记录与图件（入库 GitHub，可 diff）",
        "",
        "> 本表在**发布树内**重算：路径相对仓库根（`results/`），体积与 SHA256 均为脱敏后的实际值。",
        "> 开发树里的清单 SHA 对应脱敏前的文件，两者不通用。校验方式见 `README_audit.md` §1。",
        "",
        "| 文件 | 体积 | SHA256 |",
        "|---|---|---|",
        *rows,
        "",
        f"**Tier A 合计：{human(total)}（{len(files)} 个文件）**",
        "",
    ]
    open(os.path.join(dest, "RELEASE_MANIFEST.md"), "w", encoding="utf-8", newline="").write(
        head + "\n".join(block) + "\n\n" + tail)


def write_gitignore(dest: str) -> None:
    p = os.path.join(dest, ".gitignore")
    body = """# 发布仓库：只放 Tier A（结果记录 + 图件）与代码。
# 权重走 Zenodo；原始影像（CC BY-NC）一律不入库。
__pycache__/
*.py[cod]
.venv/
venv/
/data/raw/
/data/external/
*.tif
*.tiff
*.npy
*.pt
*.pth
"""
    open(p, "w", encoding="utf-8", newline="").write(body)


def audit_residual(dest: str) -> list[str]:
    """发布树里是否还有**本机专属**绝对路径（文本类）。

    只判两类：用户主目录（`C:\\Users\\<name>`）与其他盘符下的工作目录。
    `C:\\Windows\\...` 属系统路径（vcomp140.dll / 中文字体），保留不算泄漏。
    """
    bad = []
    for root, dirs, files in os.walk(dest):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                continue
            p = os.path.join(root, fn)
            try:
                s = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for m in ABS_PATH_RE.findall(s):
                if _is_sys_path(m):
                    continue
                bad.append(f"{os.path.relpath(p, dest)}  ->  {m[:70]}")
                break
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="发布树根目录（建议在开发仓库之外）")
    ap.add_argument("--clean", action="store_true", help="先清空 dest 下的 code/results/data")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    if os.path.abspath(dest) == os.path.abspath(BASE):
        raise SystemExit("dest 不能是开发仓库本身")
    os.makedirs(dest, exist_ok=True)
    if args.clean:
        clean(dest)

    files = tier_a_files()
    total = 0
    replaced = 0
    for rel in files:
        src = os.path.join(BASE, rel)
        # experiments/xxx → results/xxx（避免与本地 35 GB 的 experiments/ 混淆）
        sub = rel.split("experiments/", 1)[1] if rel.startswith("experiments/") else rel
        dst = os.path.join(dest, "results", sub)
        size, n = copy_file(src, dst)
        total += size
        replaced += n

    code_n = copy_code(dest)
    proc = copy_processed(dest)
    proc_n = sum(n for _, n in proc)

    root = copy_root_files(dest)
    rewrite_manifest(dest, files)
    write_gitignore(dest)

    print(f"[ok] Tier A  {len(files)} 文件  {human(total)}")
    print(f"[ok] code/    {human(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(os.path.join(dest, 'code')) for f in fs))}")
    print(f"[ok] data/processed  {len(proc)} 文件（其中清单脱敏 {proc_n} 处路径）")
    print(f"[ok] 根级文件  {len(root)} 个：{', '.join(root)}")
    print(f"[ok] RELEASE_MANIFEST.md 已在发布树内重算（results/ 路径 + 脱敏后 SHA）")
    print(f"[ok] 文本脱敏  工程路径 {replaced + code_n} 处")
    print(f"[ok] -> {dest}")

    bad = audit_residual(dest)
    if bad:
        print(f"\n[FAIL] 仍有本机绝对路径的文本文件 {len(bad)} 个：")
        for b in bad[:10]:
            print("   ", b)
        sys.exit(1)
    print("[PASS] 发布树内无本机绝对路径")


if __name__ == "__main__":
    main()
