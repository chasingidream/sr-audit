#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the EarthArXiv preprint body from the CEE submission manuscript.

The preprint needs a front page (title, author block, abstract, keywords, preprint
notice and licence) that the journal-formatted manuscript does not carry, and it must
not repeat the Abstract section that the journal manuscript has in the body.

Output: docs/EarthArXiv_preprint.md
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "docs", "\u4e3b\u6587_CEE_submission.md")
DST = os.path.join(ROOT, "docs", "EarthArXiv_\u9884\u5370\u672c\u6b63\u6587.md")

FRONT = """# {title}

[A. B. Author]<sup>1</sup>, [C. D. Author]<sup>1</sup>, [E. F. Author]<sup>2</sup>

<sup>1</sup>[Affiliation 1 — department, institution, city, country]

<sup>2</sup>[Affiliation 2 — department, institution, city, country]

*Corresponding author: [name] — [e-mail address]. ORCID: [0000-0000-0000-0000]. All
authors: please add ORCID identifiers.*

## Abstract

{abstract}

**Keywords:** super-resolution; cross-sensor fusion; skill score; evaluation
protocol; benchmarking; reproducibility; satellite imagery

---

*This is a preprint of a manuscript submitted for peer review to Communications Earth
& Environment. It has not been peer reviewed. Subsequent versions of this preprint may
differ in content. If accepted, the record of record will be the published article;
comments and corrections are welcome.*

*Preprint licence: CC BY 4.0.*

---
"""


def build():
    with open(SRC, encoding="utf-8") as f:
        txt = f.read()

    title = txt.split("\n")[0].lstrip("# ").strip()

    i = txt.index("## Abstract")
    j = txt.index("## Introduction")
    abstract = txt[i + len("## Abstract"):j].strip()
    abstract = re.sub(r"(?m)^---\s*$", "", abstract).strip()

    body = (txt[:i] + txt[j:]).split("\n", 1)[1].lstrip("\n")

    out = FRONT.format(title=title, abstract=abstract) + "\n" + body.strip() + "\n"

    with open(DST, "w", encoding="utf-8") as f:
        f.write(out)

    print("written:", DST)
    print("title:", title)
    print("abstract words:", len(abstract.split()))
    print("total tokens:", len(re.findall(r"\S+", out)))


if __name__ == "__main__":
    build()
