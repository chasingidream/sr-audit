#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wordcount_cee.py —— CEE 主文【净字数】统计（投稿硬限 5000）。

⚠⚠ 2026-09-30 重大修正：旧版只计【字母开头】的 token，把 "-7.36"、"3×"、"n = 22"
等数字 / 单位 token 全部剔除，导致同一份稿件比期刊系统（Word）口径**低估约 434 词**，
曾据此误判"4995/5000 余量充足"，而编辑部实际读到的是 5429 词（超限）。
**本脚本现以 JOURNAL（期刊系统）口径为判定依据**，旧口径降级为参考。

两种口径：
  * **JOURNAL（默认，判超限用）**：含字母或数字的 token 均计 1 词（近似 Word 计数）。
  * **PROJECT（--legacy，仅供对照）**：只计字母开头的 token（本项目历史口径）。

共同口径：
  * **只计 Introduction + Results + Discussion** 三段正文；
  * 不计：Abstract、Methods、References、End Notes、Figure legends、Tables；
  * 剥除：`<sup>…</sup>` 引用标记、Markdown 语法标记（# / * / > / 列表编号）、
    **表格行（以 `|` 开头或整行由 `|`/`-` 组成）**；
  * ⚠ **列表段必须计入**（旧脚本误剥列表 → 得到偏小的过松口径 "净 4910"）。

用法：
  python code/tools/wordcount_cee.py                 # 期刊口径（默认，判超限）
  python code/tools/wordcount_cee.py --legacy        # 旧项目口径（对照）
  python code/tools/wordcount_cee.py --file <path>   # 指定文件
"""
from __future__ import annotations

import argparse
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT = os.path.join(BASE, "docs", "主文_CEE_submission.md")
LIMIT = 5000
COUNTED = ("## Introduction", "## Results", "## Discussion")


def _body(text: str) -> str:
    """抽出 Intro+Results+Discussion 的正文并剥除标记/表格。"""
    lines = text.splitlines()
    keep, on = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("## "):
            on = any(s.startswith(c) for c in COUNTED)
            continue
        if not on:
            continue
        # 剥表格行
        if s.startswith("|") or (s and set(s) <= set("|-: ")):
            continue
        keep.append(s)
    body = "\n".join(keep)
    body = re.sub(r"<sup>.*?</sup>", " ", body, flags=re.S)      # 引用标记
    body = re.sub(r"`[^`]*`", " ", body)                          # 行内代码
    body = re.sub(r"^[#*>\-\d\.\s]+", " ", body, flags=re.M)      # Markdown 标记
    body = re.sub(r"[*_`#]", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    return body


def net_words(text: str, journal: bool = True) -> int:
    """journal=True：期刊系统口径（含数字 token，判定超限用）；
    journal=False：旧项目口径（仅字母开头 token，仅供对照）。"""
    body = _body(text)
    if journal:
        return len([w for w in body.split() if re.search(r"[A-Za-z0-9]", w)])
    return len([w for w in body.split() if re.match(r"^[A-Za-z]", w)])


def main():
    ap = argparse.ArgumentParser(
        description="CEE 主文净字数统计（仅 Intro+Results+Discussion；默认期刊口径）")
    ap.add_argument("--file", default=DEFAULT)
    ap.add_argument("--legacy", action="store_true",
                    help="使用旧项目口径（仅字母开头 token）对照输出")
    a = ap.parse_args()
    with open(a.file, encoding="utf-8") as f:
        text = f.read()
    nj = net_words(text, journal=True)
    np_ = net_words(text, journal=False)
    ok = "✅" if nj <= LIMIT else "❌ 超限"
    print(f"[wordcount] {os.path.relpath(a.file, BASE)}")
    print(f"  期刊口径 (Word，判超限) = {nj}  / 上限 {LIMIT}  余量 {LIMIT - nj:+d}   {ok}")
    if a.legacy:
        print(f"  旧项目口径 (仅字母开头) = {np_}   (差值 {nj - np_} 词)")
    elif nj <= LIMIT < np_ + 400:
        print(f"  （旧口径 {np_}；两口径差 {nj - np_} 词，以期刊口径为准）")


if __name__ == "__main__":
    main()
