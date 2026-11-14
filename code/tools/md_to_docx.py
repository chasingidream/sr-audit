# -*- coding: utf-8 -*-
"""Markdown -> DOCX 学术投稿转换器（CEE 投稿稿 / 中文翻译稿共用）。

用法:
    python md_to_docx.py <input.md> <output.docx> [--cjk]

设计要点:
- 学术手稿朴素格式: Times New Roman 12pt / 1.5 倍行距 / 黑色标题（无主题装饰）。
- 内联解析: **bold**、*italic*、<sup>n</sup> 上标引用锚点、`code`。
- 区块: #/##/### 标题、段落、有序/无序列表、| 表格、> 引用(公式行居中斜体)。
- --cjk: 中文文档， east-asia 字体=宋体（正文）/黑体（标题），标点用中文全角。
"""
import io
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

SERIF = "Times New Roman"
CJK_BODY = "宋体"
CJK_HEAD = "黑体"


# ---------------------------------------------------------------- inline ----
def fill_runs(paragraph, text, cjk=False, base_size=None):
    """内联解析: 顺序切 <sup> -> **bold** -> *italic* -> `code`，写入 paragraph。"""
    tokens = []
    for part in re.split(r"(<sup>[^<]*</sup>)", text):
        if part.startswith("<sup>"):
            tokens.append(("sup", part[5:-6]))
            continue
        pos = 0
        for m in re.finditer(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`", part):
            if m.start() > pos:
                tokens.append(("normal", part[pos:m.start()]))
            if m.group(1) is not None:
                tokens.append(("bold", m.group(1)))
            elif m.group(2) is not None:
                tokens.append(("italic", m.group(2)))
            else:
                tokens.append(("code", m.group(3)))
            pos = m.end()
        if pos < len(part):
            tokens.append(("normal", part[pos:]))

    for style, seg in tokens:
        if not seg:
            continue
        r = paragraph.add_run(seg)
        if base_size:
            r.font.size = base_size
        r.font.name = SERIF
        if cjk:
            r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK_BODY if style != "head" else CJK_HEAD)
        if style == "bold":
            r.bold = True
        elif style == "italic":
            r.italic = True
        elif style == "code":
            r.font.name = "Consolas"
            r.font.size = Pt((base_size.pt if base_size else 12) - 1)
        elif style == "sup":
            r.font.superscript = True


# ---------------------------------------------------------------- blocks ----
def convert(md_path, docx_path, cjk=False):
    text = io.open(md_path, encoding="utf-8").read()
    lines = text.split("\n")

    doc = Document()
    # 页面: 1 inch 边距
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = SERIF
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_BODY if cjk else SERIF)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(6)

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        st = raw.strip()
        if not st or st == "---":
            i += 1
            continue

        # 表格块
        if st.startswith("|"):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if not re.match(r"^\|[\s:\-|]+\|$", row):  # 跳过分隔行
                    tbl.append([c.strip() for c in row.strip("|").split("|")])
                i += 1
            if tbl:
                cols = max(len(r) for r in tbl)
                t = doc.add_table(rows=len(tbl), cols=cols)
                t.style = "Table Grid"
                t.alignment = WD_TABLE_ALIGNMENT.CENTER
                for ri, row in enumerate(tbl):
                    for ci in range(cols):
                        cell = t.cell(ri, ci)
                        cell.paragraphs[0].text = ""
                        p = cell.paragraphs[0]
                        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                        p.paragraph_format.space_after = Pt(2)
                        val = row[ci] if ci < len(row) else ""
                        fill_runs(p, val, cjk=cjk, base_size=Pt(9))
                        if ri == 0:
                            for r in p.runs:
                                r.bold = True
                doc.add_paragraph()  # 表后空行
            continue

        # 标题
        m = re.match(r"^(#{1,3})\s+(.*)$", st)
        if m:
            level = len(m.group(1))
            body = m.group(2)
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(14)
                fill_runs(p, body, cjk=cjk, base_size=Pt(16))
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(16)
                    if cjk:
                        r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK_HEAD)
            else:
                h = doc.add_heading("", level=level - 1)
                fill_runs(h, body, cjk=cjk, base_size=Pt(13 if level == 2 else 12))
                for r in h.runs:
                    r.bold = True
                    r.font.name = SERIF
                    if cjk:
                        r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK_HEAD)
                # 标题黑色、去主题蓝
                from docx.shared import RGBColor

                for r in h.runs:
                    r.font.color.rgb = RGBColor(0, 0, 0)
            i += 1
            continue

        # 有序列表
        m = re.match(r"^(\d+)\.\s+(.*)$", st)
        if m:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            fill_runs(p, m.group(1) + ". " + m.group(2), cjk=cjk)
            i += 1
            continue

        # 无序列表
        if st.startswith("- "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            fill_runs(p, "• " + st[2:], cjk=cjk)
            i += 1
            continue

        # 引用（Δ_floor 公式行）→ 居中斜体
        if st.startswith("> "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fill_runs(p, st[2:], cjk=cjk)
            for r in p.runs:
                r.italic = True
            i += 1
            continue

        # 普通段落
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        fill_runs(p, st, cjk=cjk)
        i += 1

    doc.save(docx_path)
    print("saved:", docx_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cjk = "--cjk" in sys.argv
    convert(args[0], args[1], cjk=cjk)
