#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SR-Audit Report Standard -- auto-generate audit cards + credibility badges (P4 deliverable).

Reads one or more "audit card" JSONs (schema defined in docs/P4_报告标准.md),
computes a credibility badge (VERIFIED / QUALIFIED / VOID / UNVERIFIED) and a scientific regime
(positive / saturated / floor-failure), and emits human-readable Markdown cards plus a
machine-readable aggregate JSON.

Pure stdlib (json / argparse / glob / os) -- no torch, no numpy.
Usage:
  python code/sr_audit_report.py --inputs experiments/sota_review/rams.json experiments/sota_review/highresnet.json \
      --report experiments/audit_cards.md --out-json experiments/audit_cards.json
  python code/sr_audit_report.py --inputs-dir experiments/sota_review --report experiments/audit_cards.md
"""
import argparse
import glob
import json
import os
import sys

# ---------------------------------------------------------------- schema keys
GATES = ("g1", "g2", "g3", "g4", "g5")
GATE_NAMES = {
    "g1": "pairing completeness CC(bic,GT)>=0.30",
    "g2": "structural fidelity CC_model/CC_bic",
    "g3": "output statistics std_ratio in [0.5,1.8]",
    "g4": "periodic artifact FFT ratio < 2.5",
    "g5": "leakage risk (HR-in-net / PSNR>45)",
}
SIGNIFICANCE = 0.05


def _norm_gate(v, gate=None):
    """Normalise a gate value to PASS / WARN / FAIL / n/a.

    Registry files store some gates numerically (e.g. g3 = std_ratio "0.95"), so a
    numeric value is interpreted against the gate's own threshold when `gate` is given.
    """
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return _numeric_gate(float(v), gate)
    s = str(v).strip()
    su = s.upper()
    if su.startswith("PASS"):
        return "PASS"
    if su.startswith("WARN"):
        return "WARN"
    if su.startswith("FAIL"):
        return "FAIL"
    try:
        return _numeric_gate(float(s), gate)
    except ValueError:
        return "n/a"


def _numeric_gate(x, gate):
    """Threshold rules: G3 std_ratio in [0.5,1.8]; G4 FFT ratio < 2.5."""
    if gate == "g3":
        return "PASS" if 0.5 <= x <= 1.8 else "FAIL"
    if gate == "g4":
        return "PASS" if x < 2.5 else "FAIL"
    return "n/a"


def _gate_verdict(gates):
    """Aggregate G1..G5: any FAIL -> FAIL; else any WARN -> WARN; else PASS if any PASS; else n/a."""
    norm = [_norm_gate(gates.get(k), k) for k in GATES]
    if "FAIL" in norm:
        return "FAIL"
    if "WARN" in norm:
        return "WARN"
    if "PASS" in norm:
        return "PASS"
    return "n/a"


def _delta_floor(card):
    """Δ_floor = PSNR(bicubic) - PSNR(mean floor). Prefer explicit field, else derive from detail."""
    if card.get("delta_floor_db") is not None:
        return float(card["delta_floor_db"])
    d = card.get("detail") or {}
    bic = d.get("psnr_bic_mean")
    floor = d.get("psnr_floor_mean")
    if bic is not None and floor is not None:
        return float(bic) - float(floor)
    return None


def _passes_floor(card):
    pf = card.get("passes_floor")
    if pf is not None:
        return bool(pf)
    fg = card.get("floor_gap_db")
    if fg is not None:
        return float(fg) > 0
    return None


def derive(card):
    """Derive {credibility, regime, gate_verdict, delta_floor, headline, reasons} for one audit card."""
    # F5 fix: never emit a nameless card. Fall back to stage, then to the source
    # filename stem (set by load_cards as "_source"), and only then to a placeholder.
    src = card.get("_source") or card.get("source") or ""
    src_stem = os.path.splitext(os.path.basename(str(src)))[0] if src else ""
    method = (card.get("method") or card.get("name") or card.get("stage")
              or src_stem or "<unnamed>")
    gain = card.get("audit_gain_db")
    p = card.get("audit_p")
    df = _delta_floor(card)
    pf = _passes_floor(card)
    gv = _gate_verdict(card)

    # -- credibility badge ---------------------------------------------------
    if gv == "FAIL":
        credibility = "VOID"
    elif gv == "WARN":
        credibility = "QUALIFIED"
    elif gv == "PASS":
        credibility = "VERIFIED"
    else:
        # F5 fix: no gate was reported at all -> the number is not *verified*, it is
        # simply un-audited. Downgrade rather than inherit VERIFIED by omission.
        credibility = "UNVERIFIED"

    # -- scientific regime ---------------------------------------------------
    # A method with non-positive gain is NOT "positive": below (or tied with) the
    # simplest bicubic baseline. Distinguish a *significantly worse* method
    # (below the floor -> floor-failure) from a *tied* one (no headroom -> saturated).
    if gain is None:
        regime = "n/a"
    elif df is not None and df <= 0:
        regime = "floor-failure"
    elif pf is False:
        regime = "floor-failure"
    elif gain <= 0 and p is not None and float(p) < SIGNIFICANCE:
        regime = "floor-failure"
    elif gain <= 0:
        regime = "saturated"
    elif p is not None and float(p) >= SIGNIFICANCE:
        regime = "saturated"
    else:
        regime = "positive"

    # -- reasons / headline --------------------------------------------------
    reasons = []
    if gv == "FAIL":
        failed = [k.upper() for k in GATES if _norm_gate(card.get(k), k) == "FAIL"]
        reasons.append("gate FAIL: " + ", ".join(failed) + " -> numbers void")
    elif gv == "WARN":
        warned = [k.upper() for k in GATES if _norm_gate(card.get(k), k) == "WARN"]
        reasons.append("gate WARN: " + ", ".join(warned))
    if df is not None and df <= 0:
        reasons.append("baseline collapsed: Δ_floor=%+.2f dB <= 0, model−bicubic not interpretable as skill" % df)
    if pf is False:
        reasons.append("model does not clear the mean-predictor skill floor")
    if regime == "floor-failure" and gain is not None and float(gain) < 0:
        reasons.append("model worse than bicubic: gain=%+.2f dB < 0 (p=%.3g) -> below the simplest baseline, no genuine skill" % (float(gain), float(p) if p is not None else float("nan")))
    if regime == "saturated":
        reasons.append("gain over bicubic not significant (p=%.3g) -> no discriminative headroom" % (p if p is not None else float("nan")))
    if regime == "positive":
        reasons.append("gain significant and clears the skill floor -> genuine skill")
    if credibility == "UNVERIFIED":
        reasons.append("no gates reported -> not audited; cited qualitatively, badge withheld")

    parts = [method, "regime=%s" % regime, "badge=%s" % credibility]
    if gain is not None:
        parts.append("gain=%+.2f dB" % float(gain))
    if p is not None:
        parts.append("p=%.3g" % float(p))
    if df is not None:
        parts.append("Δ_floor=%+.2f dB" % df)
    headline = " | ".join(parts)

    return {
        "method": method,
        "task": card.get("task"),
        "n_test": card.get("n_test"),
        "split": card.get("split"),
        "gate_verdict": gv,
        "credibility": credibility,
        "regime": regime,
        "audit_gain_db": gain,
        "audit_ci": card.get("audit_ci"),
        "audit_p": p,
        "audit_d": card.get("audit_d"),
        "passes_floor": pf,
        "floor_gap_db": card.get("floor_gap_db"),
        "delta_floor_db": df,
        "gates": {k: _norm_gate(card.get(k), k) for k in GATES},
        "reasons": reasons,
        "headline": headline,
    }


def load_cards(paths):
    cards = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print("[warn] skip %s: %s" % (path, exc), file=sys.stderr)
            continue
        if isinstance(obj, list):
            for entry in obj:
                if isinstance(entry, dict):
                    entry.setdefault("_source", os.path.basename(path))
                    cards.append(entry)
        elif isinstance(obj, dict):
            # registry-style files may nest a list under "entries"/"methods"
            nested = obj.get("entries") or obj.get("methods")
            if isinstance(nested, list):
                for entry in nested:
                    if isinstance(entry, dict):
                        entry.setdefault("_source", os.path.basename(path))
                        cards.append(entry)
            else:
                obj.setdefault("_source", os.path.basename(path))
                cards.append(obj)
    return cards


def md_card(rec):
    badge_icon = {"VERIFIED": "[VERIFIED]", "QUALIFIED": "[QUALIFIED]", "VOID": "[VOID]",
                  "UNVERIFIED": "[UNVERIFIED]"}[rec["credibility"]]
    lines = []
    lines.append("### %s  %s" % (rec["method"], badge_icon))
    lines.append("")
    lines.append("- regime: **%s**" % rec["regime"])
    if rec.get("task"):
        lines.append("- task: %s" % rec["task"])
    if rec.get("n_test") is not None:
        lines.append("- n_test: %s" % rec["n_test"])
    if rec.get("split"):
        lines.append("- split: %s" % rec["split"])
    gain = rec.get("audit_gain_db")
    if gain is not None:
        ci = rec.get("audit_ci")
        ci_s = " [%.2f, %.2f]" % (ci[0], ci[1]) if isinstance(ci, list) and len(ci) == 2 else ""
        p = rec.get("audit_p")
        d = rec.get("audit_d")
        extra = ""
        if p is not None:
            extra += ", p=%.3g" % float(p)
        if d is not None:
            extra += ", d=%.2f" % float(d)
        lines.append("- gain (model-bicubic): **%+.2f dB**%s%s" % (float(gain), ci_s, extra))
    if rec.get("delta_floor_db") is not None:
        lines.append("- Delta_floor (bicubic-mean): %+.2f dB" % rec["delta_floor_db"])
    if rec.get("passes_floor") is not None:
        lines.append("- passes skill floor: %s (gap %s dB)" % (
            rec["passes_floor"],
            ("%+.2f" % float(rec["floor_gap_db"])) if rec.get("floor_gap_db") is not None else "n/a",
        ))
    lines.append("- gates: " + " / ".join("%s=%s" % (k.upper(), rec["gates"][k]) for k in GATES))
    if rec.get("reasons"):
        lines.append("- verdict: " + "; ".join(rec["reasons"]))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="SR-Audit report standard: audit cards + credibility badges")
    ap.add_argument("--inputs", nargs="*", default=[], help="audit card JSON files")
    ap.add_argument("--inputs-dir", default=None, help="directory to glob *.json audit cards from")
    ap.add_argument("--out-json", default="experiments/audit_cards.json", help="aggregate machine-readable output")
    ap.add_argument("--report", default=None, help="Markdown report path")
    args = ap.parse_args()

    paths = list(args.inputs)
    if args.inputs_dir:
        paths.extend(sorted(glob.glob(os.path.join(args.inputs_dir, "*.json"))))
    if not paths:
        print("[error] no inputs. Use --inputs <json...> or --inputs-dir <dir>", file=sys.stderr)
        return 2

    cards = load_cards(paths)
    if not cards:
        print("[error] no valid audit cards loaded", file=sys.stderr)
        return 2

    records = [derive(c) for c in cards]
    order = {"positive": 0, "saturated": 1, "floor-failure": 2, "n/a": 3}
    records.sort(key=lambda r: (order.get(r["regime"], 9), r["method"]))

    summary = {
        "standard": "SR-Audit Report Standard v1",
        "n_cards": len(records),
        "counts": {
            "regime": {k: sum(1 for r in records if r["regime"] == k) for k in ("positive", "saturated", "floor-failure", "n/a")},
            "credibility": {k: sum(1 for r in records if r["credibility"] == k)
                            for k in ("VERIFIED", "QUALIFIED", "VOID", "UNVERIFIED")},
        },
        "cards": records,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("SR-Audit Report Standard v1 -- %d card(s)" % len(records))
    print("  regime:      " + ", ".join("%s=%d" % (k, v) for k, v in summary["counts"]["regime"].items()))
    print("  credibility: " + ", ".join("%s=%d" % (k, v) for k, v in summary["counts"]["credibility"].items()))
    print("=" * 72)
    for r in records:
        print(r["headline"])
    print("[saved] %s" % args.out_json)

    if args.report:
        header = [
            "# SR-Audit Report Standard v1 -- audit cards",
            "",
            "> Auto-generated by `code/sr_audit_report.py`. Badge = credibility (VERIFIED / QUALIFIED / VOID / UNVERIFIED);"
            " regime = scientific position (positive / saturated / floor-failure).",
            "",
            "| method | regime | badge | gain (dB) | p | Delta_floor (dB) | gates |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in records:
            gain = "n/a" if r["audit_gain_db"] is None else "%+.2f" % float(r["audit_gain_db"])
            p = "n/a" if r["audit_p"] is None else "%.3g" % float(r["audit_p"])
            df = "n/a" if r["delta_floor_db"] is None else "%+.2f" % r["delta_floor_db"]
            gates = " ".join("%s:%s" % (k.upper(), r["gates"][k]) for k in GATES)
            header.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["method"], r["regime"], r["credibility"], gain, p, df, gates))
        header.append("")
        header.append("---")
        header.append("")
        body = [md_card(r) for r in records]
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("\n".join(header + body) + "\n")
        print("[saved] %s" % args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
