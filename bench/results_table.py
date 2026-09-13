"""Render the README's results section from the training reports, so the numbers cannot drift from the runs.

    python -m bench.results_table --report models/train_report.json --write README.md

Replaces everything between the <!--RESULTS--> marker and the next '---' heading rule. Without --write it
prints the markdown to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MARKER = "<!--RESULTS-->"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def row(m, name, note=""):
    if not m:
        return None
    err = []
    if m.get("fn"):
        err.append(f"{m['fn']} missed")
    if m.get("fp"):
        err.append(f"{m['fp']} false alarm" + ("s" if m["fp"] > 1 else ""))
    auc = m.get("auc")
    auc_s = "—" if auc is None or auc != auc else f"{auc:.4f}"
    return (f"| {name} | {m['n']} | **{pct(m['acc'])}** | {auc_s} | "
            f"{', '.join(err) or 'none'} |{note}")


def section(rep: dict, ood: dict | None, old_ood: dict | None) -> str:
    out: list[str] = []
    meta = rep.get("deployed", {})
    out.append(f"The shipped model is fitted on {meta.get('n_train', '?')} rows and never saw any call below "
               f"(`--holdout-frac {rep.get('holdout_frac', 0.2)} --refit fit`).\n")

    out.append("### Held out from training\n")
    out.append("| Test set | calls | accuracy | AUC | errors |")
    out.append("|---|---:|---:|---:|---|")
    for key, name in (("val_heldout_train_full", "20 % of the training calls, held out (full calls)"),
                      ("val_full_calls", "the provided `val` split (full calls)"),
                      ("val_clip60", "the `val` split, first 60 s only"),
                      ("val_clip30", "the `val` split, first 30 s only"),
                      ("val_aug_full", "held-out calls re-sent down a different channel"),
                      ("val_all_full", "**everything held out, pooled**")):
        r = row(rep.get(key), name)
        if r:
            out.append(r)
    out.append("")

    cond = rep.get("by_condition", {})
    if cond.get("channel"):
        out.append("### By call condition — the robustness that was missing\n")
        out.append("Same held-out calls, grouped by the path the caller's voice travelled. "
                   "`original` is the dataset's own recordings; the rest are re-renderings of held-out calls "
                   "through a channel the model was not fitted on for that call.\n")
        out.append("| Channel | calls | accuracy | AUC | errors |")
        out.append("|---|---:|---:|---:|---|")
        names = {"original": "original recording", "telephony": "telephony (PSTN, µ-law)",
                 "mobile": "mobile / cellular", "mic_room": "**loudspeaker → room → microphone**",
                 "voip_wide": "wideband VoIP", "handset": "replay through a handset"}
        for k, m in sorted(cond["channel"].items()):
            r = row(m, names.get(k, k))
            if r:
                out.append(r)
        out.append("")

    if cond.get("profile"):
        out.append("### By caller behaviour\n")
        out.append("| Caller | calls | accuracy | AUC | errors |")
        out.append("|---|---:|---:|---:|---|")
        names = {"-": "the dataset's own callers",
                 "bot": "pipeline bot (recognition → LLM → speech, answers in 1.4–3.2 s)",
                 "realtime": "**speech-to-speech bot (answers in 0.25–0.7 s, faster than a human)**",
                 "humanlike": "synthetic voice with human timing (back-channels, false starts)"}
        for k, m in sorted(cond["profile"].items()):
            r = row(m, names.get(k, k))
            if r:
                out.append(r)
        out.append("")

    if ood:
        out.append("### Out-of-distribution: unseen voices, unseen channels\n")
        out.append("Detection rate on generated synthetic callers, by the channel they arrived through. "
                   "The *before* column is the model this round replaced.\n")
        out.append("| Channel | calls | before | after |")
        out.append("|---|---:|---:|---:|")
        for k in ("telephony", "mobile", "voip_wide", "mic_room", "handset"):
            if k not in ood:
                continue
            b = (old_ood or {}).get(k)
            out.append(f"| {k} | {ood[k]['n']} | {pct(b['detect']) if b else '—'} | **{pct(ood[k]['detect'])}** |")
        out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="models/train_report.json")
    ap.add_argument("--ood", default="", help="JSON: {family: {n, detect}} for the shipped model")
    ap.add_argument("--old-ood", default="", help="same, for the previous model")
    ap.add_argument("--write", default="", help="README to update in place")
    a = ap.parse_args()

    rep = json.load(open(a.report, encoding="utf-8"))
    ood = json.load(open(a.ood, encoding="utf-8")) if a.ood and os.path.exists(a.ood) else None
    old = json.load(open(a.old_ood, encoding="utf-8")) if a.old_ood and os.path.exists(a.old_ood) else None
    md = section(rep, ood, old)

    if not a.write:
        # the table has arrows and non-breaking marks in it; a Windows console is cp1252 by default
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(md)
        return
    text = open(a.write, encoding="utf-8").read()
    if MARKER not in text:
        raise SystemExit(f"{a.write} has no {MARKER} marker")
    head, rest = text.split(MARKER, 1)
    tail = rest.split("\n---", 1)
    body = "\n---" + tail[1] if len(tail) > 1 else ""
    open(a.write, "w", encoding="utf-8").write(head + MARKER + "\n\n" + md + body)
    print(f"updated {a.write}")


if __name__ == "__main__":
    main()
