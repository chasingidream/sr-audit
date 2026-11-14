#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the ISPRS Journal of Photogrammetry and Remote Sensing submission version
from the CEE submission manuscript.

CEE order  : Abstract / Introduction / Results / Discussion / Methods / Refs
ISPRS order: Highlights / Graphical abstract / Abstract / Keywords
             1. Introduction
             2. Materials and methods
             3. Results
             4. Discussion
             5. Conclusions
             Declarations / Acknowledgements / References / Figure legends / Tables

Also:
  * reference calls  <sup>1,2,3</sup>  ->  [1,2,3]   (Elsevier Vancouver numbering)
  * reference list   "1. Author..."    ->  "[1] Author..."
  * subsection renumbering 2.x / 3.x
  * in-text tables promoted to numbered Table 2 / Table 3 (old Table 2 -> Table 4)
  * Data availability / Code availability moved out of Methods into Declarations
  * Outlook paragraph moved from Discussion into a standalone Conclusions section
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "docs", "main_cee.md")
DST = os.path.join(ROOT, "docs", "main_isprsj.md")

SRC = os.path.join(ROOT, "docs", "\u4e3b\u6587_CEE_submission.md")
DST = os.path.join(ROOT, "docs", "\u4e3b\u6587_ISPRSJ_submission.md")


def read():
    with open(SRC, encoding="utf-8") as f:
        return f.read()


def split_top(txt):
    lines = txt.split("\n")
    head, blocks, order = [], {}, []
    cur = None
    for ln in lines:
        if ln.startswith("## "):
            cur = ln[3:].strip()
            blocks[cur] = [ln]
            order.append(cur)
        elif cur is None:
            head.append(ln)
        else:
            blocks[cur].append(ln)
    return "\n".join(head).strip(), blocks, order


def split_subs(body):
    subs, title, buf = [], None, []
    for ln in body:
        if ln.startswith("### "):
            if title is not None:
                subs.append((title, buf))
            title, buf = ln[4:].strip(), [ln]
        elif title is not None:
            buf.append(ln)
    if title is not None:
        subs.append((title, buf))
    return [(t, trim(b)) for t, b in subs]


def trim(b):
    while b and not b[0].strip():
        b.pop(0)
    while b and not b[-1].strip():
        b.pop()
    return b


def find_sub(subs, key):
    for i, (t, _) in enumerate(subs):
        if t.lower().startswith(key.lower()):
            return i
    return -1


HIGHLIGHTS = """## Highlights

- Most reported satellite SR gains describe a collapsed baseline, not recovered detail
- One scalar PSNR(bicubic) minus PSNR(mean) sets a skill floor with no training
- In 92.6% of 3,927 scenes the bicubic reference is worse than a constant predictor
- The recoverable budget is a mappable land-surface property, lowest over vegetation
- An imagery-free climatology reproduces it and transfers to unseen regions
"""

GRAPHICAL_ABSTRACT = """## Graphical abstract

*Graphical abstract artwork to be supplied as a separate file at submission.*

Suggested content, a single panel that can be drawn directly from Fig. 1: a
two-dimensional phase plane whose horizontal axis is the recoverable information
gap (PSNR(bicubic) minus PSNR(mean), dB) and whose vertical axis is the method
gain over bicubic (dB). A single diagonal line divides the plane into "clears the
floor" and "below the floor". Three markers, one per real georeferenced axis, land
in three different regions: a same-sensor multi-frame axis above the line
(positive), a cross-sensor same-day axis far to the right with a near-zero gain
(ceiling), and a cross-sensor multi-frame axis left of zero where bicubic is worse
than a constant (floor-failure). A fourth marker shows an independently
re-implemented published method reproducing the positive result, and a small inset
shows the floor collapsing with factor. The caption line reads: "A single scalar
separates real from illusory super-resolution gains."
"""

KEYWORDS = """**Keywords:** super-resolution; cross-sensor fusion; skill score; evaluation
protocol; benchmarking; reproducibility; satellite imagery

"""

DECL_AI = """### Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used an AI-based writing assistant
for language editing, structural formatting and reference-format conversion. All
scientific content, experiments, analyses and conclusions were conceived,
implemented and verified by the authors, who reviewed and edited every section and
take full responsibility for the content of the published article.
"""

CREDIT = """### CRediT authorship contribution statement

[A.B.]: Conceptualization, Methodology, Writing - original draft, Writing - review
& editing, Supervision, Project administration. [C.D.]: Methodology, Software, Data
curation, Formal analysis, Validation, Visualization, Writing - review & editing.
[E.F.]: Software, Investigation, Validation, Formal analysis, Writing - review &
editing.

[投稿前请将 [A.B.]/[C.D.]/[E.F.] 替换为真实作者姓名，并补充单位、ORCID 与通讯作者邮箱；
角色项按 CRediT 14 类术语（Conceptualization / Methodology / Software / Validation /
Formal analysis / Investigation / Resources / Data curation / Writing - original draft /
Writing - review & editing / Visualization / Supervision / Project administration /
Funding acquisition）据实增减。]
"""

COMPETING = """### Declaration of competing interests

The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported in
this paper.
"""

CONCL_LEAD = """## 5. Conclusions

Cross-sensor super-resolution gains are interpretable only relative to a floor that
the data, not the model, determines. We have shown that a single scalar --
PSNR(bicubic) minus PSNR(mean), computable without any training from the
low-resolution observation and the target alone -- separates three settings that
the super-resolution literature currently reports using the same vocabulary and the
same metric: one in which a genuine gain exists and is certified (3x, same sensor,
multi-frame), one in which classical upsampling has already consumed the
recoverable budget so that the gain metric loses discriminative power (2x,
cross-sensor, same day), and one in which the baseline itself has collapsed below a
constant predictor, so that any positive gain describes the collapse rather than the
method (6.63x, cross-sensor, multi-frame). The practical consequence is a gate
rather than a new metric: no gain over bicubic should be interpreted until it clears
the floor under a per-sample paired test, and the aggregation rule, the loading
convention and the pairing completeness that determine that number should be
reported alongside it. We release the gate as an executable protocol with a
machine-checkable credibility badge, apply it to published methods and to our own
earlier results, and show that its verdict predicts downstream utility.
"""


# (old, new) applied after assembly: CEE writes "(Methods)" for cross references,
# ISPRS J uses numbered sections, so each is pinned to the relevant subsection.
SECTION_FIXES = [
    ("(SR-Audit; see Methods)", "(SR-Audit; Section 2.4)"),
    ("all content classes positive \u2014 Methods)",
     "all content classes positive \u2014 Section 2.15)"),
    ("invariant to global linear rescaling (Methods)",
     "invariant to global linear rescaling (Section 2.6)"),
    ("(four seeds; Methods)", "(four seeds; Section 2.21)"),
    ("standard convention of Methods", "standard convention of Section 2.6"),
    ("declared in Methods", "declared in Section 2.6"),
    ("paired *t*-tests give (Fig. 1, right panel;",
     "paired *t*-tests give (Table 2; Fig. 1, right panel;"),
    ("measuring the **pure-scale** gap (Fig. 3).",
     "measuring the **pure-scale** gap (Table 3, Fig. 3)."),
]


def build():
    txt = read()
    head, blocks, order = split_top(txt)

    title = ""
    for ln in head.split("\n"):
        if ln.startswith("# "):
            title = ln[2:].strip()
            break
    if not title:
        sys.exit("title not found")

    abstract = trim(blocks["Abstract"][1:])
    intro = trim(blocks["Introduction"][1:])
    results_body = blocks["Results"][1:]
    disc_body = blocks["Discussion"][1:]
    methods_body = blocks["Methods"][1:]
    refs = trim(blocks["References"][1:])
    endnotes_body = blocks["End Notes"][1:]
    figleg = trim(blocks["Figure legends"][1:])
    tables = trim(blocks["Tables"][1:])

    m_subs = split_subs(methods_body)
    r_subs = split_subs(results_body)
    e_subs = split_subs(endnotes_body)

    # ---- pull Data / Code availability out of Methods ----
    data_avail = code_avail = None
    for key in ("Data availability", "Code availability"):
        i = find_sub(m_subs, key)
        if i < 0:
            sys.exit("missing %s in Methods" % key)
        body = trim(m_subs[i][1][1:])
        if key.startswith("Data"):
            data_avail = body
        else:
            code_avail = body
    m_subs = [s for s in m_subs
              if not s[0].lower().startswith("data availability")
              and not s[0].lower().startswith("code availability")]

    # ---- pull Outlook out of Discussion (becomes Conclusions part 2) ----
    dt = "\n".join(disc_body)
    m = re.search(r"\n\*\*Outlook\.\*\*", dt)
    if not m:
        sys.exit("Outlook paragraph not found in Discussion")
    disc_keep = dt[:m.start()].rstrip()
    outlook = dt[m.start():].strip().replace("\n", " ")

    # ---- pull Acknowledgements / Funding / Author contributions from End Notes ----
    ack = fund = None
    for t, b in e_subs:
        if t.lower().startswith("acknowledgements"):
            ack = trim(b[1:])
        elif t.lower().startswith("funding"):
            fund = trim(b[1:])
    if ack is None or fund is None:
        sys.exit("missing Acknowledgements or Funding")

    def vanc(s):
        s = s.replace("<sup>", "[").replace("</sup>", "]")
        return s

    def renum_refs(s):
        return re.sub(r"(?m)^(\d+)\. ", lambda mo: "[%s] " % mo.group(1), s)

    def join(parts):
        return "\n\n".join(p.strip() for p in parts if p.strip())

    # ---- assemble body ----
    out = []
    out.append("# %s" % title)
    out.append("")
    out.append("*[Author names, affiliations, ORCID identifiers and the corresponding "
               "author's e-mail address to be inserted before submission.]*")
    out.append("")
    out.append(HIGHLIGHTS.strip())
    out.append("")
    out.append(GRAPHICAL_ABSTRACT.strip())
    out.append("")
    out.append("## Abstract\n\n" + join(["\n".join(abstract)]))
    out.append("")
    out.append(KEYWORDS.strip())
    out.append("")
    out.append("## 1. Introduction\n\n" + "\n".join(intro))
    out.append("")
    out.append("## 2. Materials and methods")
    for i, (t, b) in enumerate(m_subs, 1):
        body = "\n".join(trim(b[1:])).strip()
        out.append("")
        out.append("### 2.%d %s\n\n%s" % (i, t, body))
    out.append("")
    out.append("## 3. Results")
    for i, (t, b) in enumerate(r_subs, 1):
        body = "\n".join(trim(b[1:])).strip()
        out.append("")
        out.append("### 3.%d %s\n\n%s" % (i, t, body))
    out.append("")
    out.append("## 4. Discussion\n\n" + disc_keep)
    out.append("")
    out.append(CONCL_LEAD.strip())
    out.append("")
    out.append(outlook)
    out.append("")
    out.append("## Declarations")
    out.append("")
    out.append(COMPETING.strip())
    out.append("")
    out.append(DECL_AI.strip())
    out.append("")
    out.append("### Data availability\n\n" + "\n".join(data_avail))
    out.append("")
    out.append("### Code availability\n\n" + "\n".join(code_avail))
    out.append("")
    out.append(CREDIT.strip())
    out.append("")
    out.append("### Acknowledgements\n\n" + "\n".join(ack))
    out.append("")
    out.append("### Funding\n\n" + "\n".join(fund))
    out.append("")
    out.append("## References\n\n" + renum_refs("\n".join(refs)))
    out.append("")
    out.append("## Figure legends\n\n" + "\n".join(figleg))
    out.append("")
    out.append("## Tables\n\n" + "\n".join(tables))

    body = "\n".join(out)
    body = vanc(body)

    # ---- in-text tables -> numbered Table 2 / Table 3 ; old Table 2 -> Table 4 ----
    body = body.replace("Table 2", "Table 4")

    cap2 = ("**Table 2 | Controlled test of a cross-attention self-alignment module "
            "(WorldStrat 6.63x, n = 60).** Paired *t*-tests on per-sample PSNR "
            "differences; CI is the 95% interval; the verdict column states whether "
            "the contrast is significant.\n\n")
    anchor2 = "| Contrast | \u0394PSNR (95% CI) | p | Cohen's d | Verdict |"
    if anchor2 not in body:
        sys.exit("table 2 anchor not found")
    body = body.replace(anchor2, cap2 + anchor2, 1)

    cap3 = ("**Table 3 | Scale isolation: pure-scale ceiling versus real arms.** "
            "Pure-scale values are scene-matched to the WorldStrat real arm "
            "(same 60 scenes, same deterministic 424-px windows); the decorrelation "
            "contribution is the real arm minus its scene-matched pure-scale "
            "ceiling.\n\n")
    anchor3 = "| Factor | Pure-scale \u0394_floor (scene-matched) |"
    if anchor3 not in body:
        sys.exit("table 3 anchor not found")
    body = body.replace(anchor3, cap3 + anchor3, 1)

    for old, new in SECTION_FIXES:
        if old not in body:
            print("WARN: fix not applied ->", old)
        else:
            body = body.replace(old, new, 1)

    # strip leftover horizontal rules used as CEE section dividers
    body = re.sub(r"(?m)^---\s*$", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)

    with open(DST, "w", encoding="utf-8") as f:
        f.write(body.strip() + "\n")

    n_words = len(re.findall(r"\S+", body))
    print("written:", DST)
    print("words (all tokens, incl. Methods):", n_words)


if __name__ == "__main__":
    build()
