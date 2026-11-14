# -*- coding: utf-8 -*-
"""向 CEE 投稿主文 docx 的 Figure legends 节插入对应图件，生成"含图版"。

依据 Nature Portfolio 投稿指南："Each figure legend should be presented on the
same page as its figure." 图插在各自图注段之前（同页、图上图下）。

用法:
  python embed_figs_docx.py                          # CEE 版（默认）
  python embed_figs_docx.py <src.docx> <dst.docx>    # 任意版本，如 ISPRS J 版
输出: docs/主文_CEE_submission_含图.docx（默认）
"""
import os
import sys
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "主文_CEE_submission.docx")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "主文_CEE_submission_含图.docx")
FIGDIR = os.path.join(ROOT, "experiments", "figs_final")

# 图注段前缀 -> 图件
MAP = {
    "Fig. 1 |": "fig1_regime.png",
    "Fig. 2 |": "fig2_floor_collapse.png",
    "Fig. 3 |": "fig3_e1_decomposition.png",
    "Fig. 4 |": "fig4_crosssensor_3x.png",
    "Fig. 5 |": "figS12_atlas_tiers.png",   # atlas promoted to main text (2026-10-06, R4)
    "Supplementary Fig. 1 |": "figS1_mtxa_persample.png",
    "Supplementary Fig. 2 |": "figS2_highresnet_dist.png",
    "Supplementary Fig. 3 |": "figS3_robustness_forest.png",
    "Supplementary Fig. 4 |": "figS4_threshold_sweep.png",
    "Supplementary Fig. 5 |": "figS5_downstream_decoupling.png",
    "Supplementary Fig. 6b |": "figS6b_paired_counterfactual.png",
    "Supplementary Fig. 6 |": "figS6_stratified_floor.png",
    "Supplementary Fig. 7 |": "figS7_feasibility_map.png",
    "Supplementary Fig. 8 |": "figS8_independent_surface.png",
    "Supplementary Fig. 9 |": "figS9_decorrelation_selectivity.png",
    "Supplementary Fig. 10 |": "figS10_canopy_dynamics.png",
    "Supplementary Fig. 11 |": "figS11_variable_scaling.png",
}

def main():
    doc = Document(SRC)
    inserted = []
    for para in list(doc.paragraphs):
        text = para.text.strip()
        for prefix, fname in MAP.items():
            if text.startswith(prefix):
                img = os.path.join(FIGDIR, fname)
                if not os.path.exists(img):
                    raise FileNotFoundError(img)
                # 在图注段之前插入图片段（图上图下、同页）
                holder = para.insert_paragraph_before()
                holder.alignment = WD_ALIGN_PARAGRAPH.CENTER
                holder.add_run().add_picture(img, width=Inches(6.0))
                inserted.append(prefix)
                break
    doc.save(DST)
    print("inserted %d figures:" % len(inserted))
    for p in inserted:
        print("  -", p)
    if len(inserted) != len(MAP):
        raise SystemExit("ERROR: expected %d, got %d" % (len(MAP), len(inserted)))
    print("saved:", DST)

if __name__ == "__main__":
    main()
