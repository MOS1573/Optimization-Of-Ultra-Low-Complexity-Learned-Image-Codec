#!/usr/bin/env python3
"""
Merge a new partial eval-harness run into the authoritative CSV.

Usage:
    python "Optimization/Evaluation Harness/merge_eval_results.py" \
        --new_dir  results/eval_harness/eval_<new_timestamp> \
        --base_dir results/eval_harness/eval_20260517_164947

The script:
  1. Reads both CSVs (Kodak and pro_2020).
  2. Merges rows: new run takes precedence for any key it contains.
  3. Writes a merged CSV back into --base_dir, overwriting the originals.
  4. Prints a diff table showing what changed.

After merging, the base_dir CSVs are the new authoritative source.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load_csv(path: Path) -> dict[str, dict]:
    """Return {key: row_dict} from a CSV."""
    with open(path) as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fmt(v, dec=4):
    try:
        fv = float(v)
        return "N/A" if math.isnan(fv) else f"{fv:.{dec}f}"
    except (ValueError, TypeError):
        return str(v)


def merge_and_report(base: dict, new: dict, dataset: str) -> dict:
    merged = dict(base)
    print(f"\n{'='*70}")
    print(f"  {dataset}  —  changes introduced by new eval run")
    print(f"{'='*70}")
    print(f"  {'key':<20} {'metric':<12} {'old':>10}  →  {'new':>10}")
    print(f"  {'-'*60}")

    changed = False
    for key, new_row in new.items():
        if key not in merged:
            merged[key] = new_row
            print(f"  {key:<20} [NEW ROW — all metrics from new run]")
            changed = True
        else:
            old_row = merged[key]
            row_changed = False
            for field in ["psnr", "bpp", "ssim", "ms_ssim", "lpips", "delta_psnr"]:
                old_v = old_row.get(field, "")
                new_v = new_row.get(field, "")
                try:
                    if abs(float(old_v) - float(new_v)) > 1e-6:
                        print(f"  {key:<20} {field:<12} {fmt(old_v):>10}  →  {fmt(new_v):>10}")
                        row_changed = True
                except (ValueError, TypeError):
                    pass
            if row_changed:
                merged[key] = new_row
                changed = True

    if not changed:
        print("  (no changes)")
    return merged


def main():
    p = argparse.ArgumentParser(description="Merge partial eval run into authoritative CSV")
    p.add_argument("--new_dir",  required=True,
                   help="Path to new eval_harness output directory (eval_<timestamp>)")
    p.add_argument("--base_dir", required=True,
                   help="Path to authoritative base directory (eval_20260517_164947)")
    args = p.parse_args()

    new_dir  = Path(args.new_dir)
    base_dir = Path(args.base_dir)

    for ds in ("kodak", "pro_2020"):
        base_csv = base_dir / f"evaluation_{ds}.csv"
        new_csv  = new_dir  / f"evaluation_{ds}.csv"

        if not base_csv.exists():
            print(f"[error] Base CSV not found: {base_csv}")
            continue
        if not new_csv.exists():
            print(f"[warn]  New CSV not found for {ds}: {new_csv} — skipping")
            continue

        base_rows = load_csv(base_csv)
        new_rows  = load_csv(new_csv)

        # Get field order from base CSV (preserves column structure)
        with open(base_csv) as f:
            fieldnames = csv.DictReader(f).fieldnames

        merged = merge_and_report(base_rows, new_rows, ds)

        # Write back to base_dir — this makes base_dir the new authoritative source
        ordered = [merged[k] for k in sorted(merged.keys())]
        save_csv(ordered, base_csv, fieldnames)
        print(f"\n  ✓  Written → {base_csv}")

    print("\nDone. The base_dir CSVs are now the merged authoritative source.")
    print("Update comparison_table.md with any changed numbers printed above.")


if __name__ == "__main__":
    main()