# -*- coding: utf-8 -*-
"""Method-name length distribution statistics.

Usage:
  # camelCase measurement only (no tokenizer, no third-party dependency)
  python name_length_stats.py --json /path/train_....json /path/test_....json

  # both measurements, plus the callee-name comparison
  python name_length_stats.py --json /path/train_....json /path/test_....json \\
      --tokenizer /path/models/CodeT5p-220m --callee --out name_length_stats.json

JSON written by --out: each group (root / callee) has a "camel" and a "bpe" measurement;
each contains mean/median/p90/p95/p99/min/max, share_1, n_over_label_length, and hist --
per-bucket counts for every exact length value, **unaffected by --cap**.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def camel_case_tokenize(text):
    """camelCase tokenization: processFooBar -> ['process', 'Foo', 'Bar']

    Taken verbatim from code/models/model-2025-10-15/generation_task.py:38-39
    (the measurement used by the MNR metrics).
    """
    return re.sub('([a-z])([A-Z])', r'\1 \2', text).split()


def extract_method_name(full_name):
    """org.pkg.Cls.methodName(ParamType) -> methodName
    """
    head = full_name.split("(")[0]
    return head.rsplit(".", 1)[-1]


def collect_names(path, with_callee):
    """Read one dataset JSON and return (root names, callee names).

    root   = the method name in root.method_full_name of every sample (the target name).
    callee = the method names in children[*].method_full_name across the whole call tree
             (recursive).
    When --callee is not requested, children are not traversed (callee is empty).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roots = []
    callees = []

    def walk(node):
        for child in node.get("children") or []:
            full = child.get("method_full_name")
            if full:
                callees.append(extract_method_name(full))
            walk(child)

    for item in data:
        root = item.get("root") or {}
        full = root.get("method_full_name")
        if full:
            roots.append(extract_method_name(full))
        if with_callee:
            walk(root)
    return roots, callees


def quantile(sorted_vals, p):
    """Nearest-rank quantile (stable even for very small n)."""
    n = len(sorted_vals)
    if n == 0:
        return 0
    idx = min(n - 1, int(round(p * (n - 1))))
    return sorted_vals[idx]


def describe(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": round(sum(s) / n, 3),
        "median": quantile(s, 0.5),
        "p90": quantile(s, 0.9),
        "p95": quantile(s, 0.95),
        "p99": quantile(s, 0.99),
        "min": s[0],
        "max": s[-1],
    }


def hist_dict(counter):
    """Counter -> {"length": count} (ascending by numeric key).

    Used for the JSON output and **unaffected by --cap**: the terminal ASCII histogram
    merges lengths above cap into one line, whereas the JSON keeps the exact count for
    every length value.
    """
    return {str(k): counter[k] for k in sorted(counter)}


def histogram_lines(counter, cap, width=40):
    """Draw the ASCII histogram; all lengths above cap are merged into one line."""
    total = sum(counter.values())
    if total == 0:
        return []
    buckets = Counter()
    for k, v in counter.items():
        buckets[min(k, cap + 1)] += v
    peak = max(buckets.values())
    lines = []
    for k in sorted(buckets):
        label = f">{cap}" if k == cap + 1 else str(k)
        bar = "#" * max(1, int(round(buckets[k] / peak * width)))
        pct = buckets[k] / total * 100
        lines.append(f"      {label:>4} | {bar:<{width}} {buckets[k]:>7}  {pct:6.2f}%")
    return lines


def measure(names, tok, label_length):
    """Compute both measurements for one name list.

    Returns (record dict, camel histogram Counter, BPE histogram Counter or None).
    """
    rec = {"n": len(names), "n_unique": len(set(names))}

    # ---- measurement (a): camelCase sub-tokens ----
    sub_lens = [len(camel_case_tokenize(n)) for n in names]
    rec["camel"] = describe(sub_lens)
    if sub_lens:
        rec["camel"]["share_1"] = round(
            sum(1 for v in sub_lens if v == 1) / len(sub_lens) * 100, 2)
        rec["camel"]["n_over_label_length"] = sum(1 for v in sub_lens if v > label_length)
    camel_hist = Counter(sub_lens)
    rec["camel"]["hist"] = hist_dict(camel_hist)

    # ---- measurement (b): CodeT5+ BPE on the pure name, without special tokens ----
    bpe_hist = None
    if tok is not None and names:
        bpe_lens = [len(ids) for ids in tok(names, add_special_tokens=False)["input_ids"]]
        rec["bpe"] = describe(bpe_lens)
        rec["bpe"]["share_1"] = round(
            sum(1 for v in bpe_lens if v == 1) / len(bpe_lens) * 100, 2)
        rec["bpe"]["n_over_label_length"] = sum(1 for v in bpe_lens if v > label_length)
        bpe_hist = Counter(bpe_lens)
        rec["bpe"]["hist"] = hist_dict(bpe_hist)

    return rec, camel_hist, bpe_hist


def report_file(tag, roots, callees, tok, label_length, cap):
    """Compute all metrics for one dataset (or merged set); return (text, dict)."""
    out = {"file": tag, "n_samples": len(roots)}

    groups = [("root (target name)", "root", roots)]
    if callees is not None:
        groups.append(("callee (called name)", "callee", callees))

    lines = ["", "=" * 78, f"[{tag}]  samples {len(roots)}", "=" * 78]

    for title, key, names in groups:
        rec, camel_hist, bpe_hist = measure(names, tok, label_length)
        out[key] = rec

        lines.append("")
        lines.append(f"-- {title} --   n={rec['n']}   unique={rec['n_unique']}")
        if rec["n"] == 0:
            lines.append("      (no samples)")
            continue

        c = rec["camel"]
        lines.append(f"   (a) camelCase sub-tokens: mean {c['mean']}  median {c['median']}  "
                     f"p90 {c['p90']}  p95 {c['p95']}  p99 {c['p99']}  max {c['max']}   "
                     f"single-subtoken share {c['share_1']}%")
        lines.extend(histogram_lines(camel_hist, cap))

        if bpe_hist is not None:
            b = rec["bpe"]
            lines.append(f"   (b) CodeT5+ BPE tokens: mean {b['mean']}  median {b['median']}  "
                         f"p90 {b['p90']}  p95 {b['p95']}  p99 {b['p99']}  max {b['max']}   "
                         f"single-token share {b['share_1']}%")
            lines.extend(histogram_lines(bpe_hist, cap))
            if b["n_over_label_length"] == 0:
                lines.append(f"       count >{label_length}: 0   OK, consistent with the data-construction filter")
            else:
                lines.append(f"       count >{label_length}: {b['n_over_label_length']}   "
                             f"WARNING: non-zero, so the filter did not take effect or the measurement differs")
        else:
            lines.append("   (b) CodeT5+ BPE tokens: skipped (no --tokenizer given)")

    return "\n".join(lines), out


def main():
    ap = argparse.ArgumentParser(
        description="Method-name length distribution (camelCase sub-tokens + CodeT5+ BPE tokens)")
    ap.add_argument("--json", nargs="+", required=True,
                    help="one or more dataset JSON paths (train/test is enough)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF name or local directory of the CodeT5+ tokenizer; without it, "
                         "only the camelCase measurement is produced. "
                         "Example: /path/models/CodeT5p-220m or Salesforce/codet5p-220m")
    ap.add_argument("--callee", action="store_true",
                    help="also measure all callee names in the call tree (a second "
                         "distribution next to root names)")
    ap.add_argument("--label-length", type=int, default=16,
                    help="training-side label_length (default 16, as in CalleeFuse)")
    ap.add_argument("--cap", type=int, default=16,
                    help="largest length listed in the histogram (longer values are merged "
                         "into one line); default 16, i.e. the label_length boundary, so the "
                         ">16 line is the consistency check")
    ap.add_argument("--out", default=None, help="write all numbers to a JSON file")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("[ERROR] transformers is not installed; drop --tokenizer to use the camelCase measurement only.")
            return 2
        print(f"[info] loading tokenizer: {args.tokenizer}")
        try:
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"[ERROR] failed to load the tokenizer: {e}")
            print("        on an offline machine, use --tokenizer /local/models/CodeT5p-220m")
            return 2
        print(f"[info] tokenizer ready, vocabulary size {tok.vocab_size}")

    results = []
    all_roots = []
    all_callees = [] if args.callee else None

    for path in args.json:
        if not os.path.isfile(path):
            print(f"[ERROR] file not found: {path}")
            return 2
        roots, callees = collect_names(path, args.callee)
        all_roots.extend(roots)
        if args.callee:
            all_callees.extend(callees)
        text, rec = report_file(os.path.basename(path), roots,
                                callees if args.callee else None,
                                tok, args.label_length, args.cap)
        print(text)
        results.append(rec)

    if len(args.json) > 1:
        text, rec = report_file("TOTAL (merged)", all_roots, all_callees,
                                tok, args.label_length, args.cap)
        print(text)
        results.append(rec)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[info] numbers written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
