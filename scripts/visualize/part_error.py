"""
Simple script to visualize pair error rates from a result file.

Example usage:
    python part_error.py --file logs/my_model/train/result.txt --out plots/error_rates.png
    python part_error.py --result PS S7 20 ps/result.txt --result LPS S7 20 lps/result.txt \
        --reference PS --out plots/position_errors.png

Note: result.txt is appended to on every rerun; this script always reads the LAST
complete 0..7 block (see parse_result_file) and logs which block it used.
"""

import argparse
import csv
import math
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


LineData = Tuple[float, float]


def parse_result_file(file_path: str, verbose: bool = True) -> Dict[int, LineData]:
    """
    Parse a result.txt-like file and extract (acc, total_samples) for indices 0..7.

    Returns {index: (acc, total_samples)} from the LAST complete 0..7 block: result.txt is
    opened in append mode, so reruns add blocks and the newest one belongs to the latest run.
    A block starts at an 'EER=' line (or at index 0 / a repeated index); an incomplete
    trailing block is skipped with a warning. result.txt does not record checkpoint or
    resolution, so use a fresh EVAL_OUTPUT_DIR per run.
    """
    index_pattern = re.compile(
        r"^(?P<idx>[0-7]):\s*acc:\s*(?P<acc>[0-9]+\.[0-9]+),\s*total_samples:\s*(?P<total>[0-9]+\.[0-9]+)",
    )
    blocks: List[Dict[int, LineData]] = []
    current: Optional[Dict[int, LineData]] = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("EER="):
                # Header of a new run -> start a fresh block.
                current = {}
                blocks.append(current)
                continue
            m = index_pattern.match(stripped)
            if not m:
                continue
            idx = int(m.group("idx"))
            # index 0 or a repeated index also starts a block (files without an EER= header)
            if current is None or (idx == 0 and len(current) > 0) or idx in current:
                current = {}
                blocks.append(current)
            current[idx] = (float(m.group("acc")), float(m.group("total")))
    complete = [i for i, b in enumerate(blocks) if len(b) == 8]
    if not complete:
        seen = blocks[-1] if blocks else {}
        missing = sorted(set(range(8)) - set(seen.keys()))
        raise ValueError(
            f"File {file_path} does not contain a complete set of indices 0..7. Missing: {missing}"
        )
    used = complete[-1]
    if verbose:
        # Log which block was used so a stale/duplicated result.txt is visible in the job log.
        print(f"[part_error] {file_path}: using block {used + 1} of {len(blocks)} "
              f"({len(complete)} complete; last complete block is used)")
        if len(blocks) > 1:
            # Several runs were appended to this file; see LIMITATION above.
            print(f"[part_error] WARNING {file_path}: {len(blocks)} result blocks found; result.txt "
                  f"does not record checkpoint or resolution, so it cannot be checked that the "
                  f"last block is the run you want. Prefer a fresh EVAL_OUTPUT_DIR per run.")
        if used != len(blocks) - 1:
            print(f"[part_error] WARNING {file_path}: the final block is incomplete "
                  f"(crashed or still running?); fell back to block {used + 1}")
    return blocks[used]


def compute_pair_error_rates(index_to_data: Dict[int, LineData]) -> List[float]:
    """
    Compute error rates for pairs: (0+4), (1+5), (2+6), (3+7).

    Error count for an index i: (100 - acc) * total / 100
    Pair error rate: (error_i + error_j) / (total_i + total_j)
    """
    pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    rates: List[float] = []
    for a, b in pairs:
        acc_a, total_a = index_to_data[a]
        acc_b, total_b = index_to_data[b]
        error_a = (100.0 - acc_a) * total_a / 100.0
        error_b = (100.0 - acc_b) * total_b / 100.0
        denom = total_a + total_b
        rate = 0.0 if denom == 0 else (error_a + error_b) / denom
        rates.append(rate)
    return rates


def plot_error_rates(
    labels: List[str],
    values: List[float],
    title: Optional[str],
    output_path: str,
) -> None:
    """Plot error rates as a simple bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = list(range(len(labels)))
    
    ax.bar(x, values, color='#abd3e1', edgecolor='#8693a0', linewidth=0.4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel("Error rate", fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


POSITIONS = ["Start", "Middle", "End", "Unit"]


def error_percent(data):
    return [100 - data[i][0] if data[i][1] else float("nan") for i in range(8)]


def analyse_results(specs, output_path, reference):
    results = {}
    for condition, model, resolution, path in specs:
        resolution = float(resolution)
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("Resolution must be a positive number of milliseconds")
        key = (condition, model)
        if key in results:
            raise ValueError(f"Duplicate condition/model: {key}")
        results[key] = (resolution, path, parse_result_file(path))
    deltas = []
    for (condition, model), (resolution, path, data) in results.items():
        if not reference or condition == reference:
            continue
        if (reference, model) not in results:
            raise ValueError(f"Missing {reference} reference for {model}")
        base_resolution, _, base = results[reference, model]
        if resolution != base_resolution:
            raise ValueError(f"{condition}/{model}: {resolution} ms differs from reference {base_resolution} ms")
        deltas.append((f"{condition} / {model}", [a-b for a,b in zip(error_percent(data), error_percent(base))]))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    stem = os.path.splitext(output_path)[0]
    with open(stem + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "model", "resolution_ms", "class", "position", "frames", "error_percent", "estimated_errors", "delta_pp", "source"])
        for (condition, model), (resolution, path, data) in results.items():
            rates = error_percent(data)
            base = error_percent(results[reference, model][2]) if reference else None
            for i, rate in enumerate(rates):
                writer.writerow([condition, model, resolution, "spoof" if i < 4 else "genuine", POSITIONS[i % 4], int(data[i][1]), rate, rate * data[i][1] / 100 if data[i][1] else 0, rate-base[i] if base else "", os.path.abspath(path)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    conditions = list(dict.fromkeys(c for c, _ in results))
    models = list(dict.fromkeys(m for _, m in results))
    palette = ["#3b75af", "#e07b39", "#3a9276", "#8c564b", "#7f7f7f", "#bcbd22"]
    # colour = condition, hatch = model, so every condition/model pair stays distinguishable
    hatches = ["///", None, "xx", "\\\\", "..", "++", "--", "oo"]
    width = .8 / len(results)
    for j, ((condition, model), (resolution, _, data)) in enumerate(results.items()):
        rates = error_percent(data)
        c_idx = conditions.index(condition) if len(conditions) > 1 else models.index(model)
        m_idx = models.index(model)
        color = palette[c_idx % len(palette)]
        hatch = hatches[m_idx % len(hatches)] if len(conditions) > 1 and len(models) > 1 else None
        for panel, ax in enumerate(axes):
            ax.bar([i - .4 + width * (j + .5) for i in range(4)], rates[panel * 4:panel * 4 + 4], width,
                   color=color, alpha=0.65 if hatch else 0.92,
                   edgecolor="#222222" if hatch else "white", linewidth=0.6,
                   hatch=hatch,
                   label=f"{condition} / {model} ({resolution:g} ms)")
    for panel, (ax, title) in enumerate(zip(axes, ["Spoof misses (true spoof → pred genuine)", "Genuine false alarms (true genuine → pred spoof)"])):
        # class shares per condition (from one model's counts), never pooled across datasets
        share_parts = []
        for condition in conditions:
            per_model = [d for (c, _), (_, _, d) in results.items() if c == condition]
            counts = [tuple(d[panel * 4 + i][1] for i in range(4)) for d in per_model]
            if len(set(counts)) > 1:
                print(f"[part_error] WARNING: models in condition {condition!r} have different "
                      f"frame counts; class shares use the first model's counts")
            totals = counts[0]
            class_sum = sum(totals) or 1.0
            share_parts.append((condition, [100.0 * t / class_sum for t in totals]))

        def fmt_share(s):
            return f"{s:.1f}%" if s >= 0.1 else "<0.1%"
        if len(share_parts) == 1:
            shares = share_parts[0][1]
            xlabels = [f"{p}\n({fmt_share(shares[i])})" for i, p in enumerate(POSITIONS)]
        else:
            # one line per condition so the labels do not overlap neighbouring ticks
            xlabels = [f"{p}\n" + "\n".join(f"{c} {fmt_share(sh[i])}" for c, sh in share_parts)
                       for i, p in enumerate(POSITIONS)]
        ax.set(title=title, ylabel="Error rate (%)", xticks=range(4), xticklabels=xlabels)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=min(len(results), 3), fontsize=9, frameon=True)
    fig.tight_layout(rect=(0, .04, 1, .88))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    if deltas:
        fig, ax = plt.subplots(figsize=(12, max(3, len(deltas)*.55+1.5)))
        values = [row for _, row in deltas]
        limit = max([abs(v) for row in values for v in row if math.isfinite(v)] + [1])
        im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(xticks=range(8), xticklabels=[f"{c}\n{p}" for c in ("Spoof", "Genuine") for p in POSITIONS],
               yticks=range(len(deltas)), yticklabels=[name for name, _ in deltas], title=f"Error change relative to {reference}")
        for row, vals in enumerate(values):
            for col, value in enumerate(vals):
                ax.text(col, row, f"{value:+.2f}" if math.isfinite(value) else "N/A", ha="center", va="center")
        fig.colorbar(im, ax=ax, label="Percentage points (positive = worse)")
        fig.tight_layout()
        fig.savefig(stem + "_delta.png", dpi=180)
        plt.close(fig)
    print(f"Saved {output_path} and {stem}.csv" + (f" and {stem}_delta.png" if deltas else ""))


def main():
    parser = argparse.ArgumentParser(
        description="Plot pair error rates from a result file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python part_error.py --file logs/my_model/train/result.txt --out plots/error_rates.png
        """,
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to result.txt file (legacy pooled plot)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output image path (PNG)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot title (leave empty for no title)",
    )
    parser.add_argument("--result", nargs=4, action="append", metavar=("CONDITION", "MODEL", "MS", "FILE"),
                        help="Repeat for each result; e.g. --result PS S3 160 result_S3.txt")
    parser.add_argument("--reference", help="Reference condition for same-resolution differences, e.g. PS")
    args = parser.parse_args()
    if bool(args.file) == bool(args.result):
        parser.error("Choose --file or repeated --result arguments")
    if args.result:
        analyse_results(args.result, args.out, args.reference)
        return
    if args.reference:
        parser.error("--reference requires --result")

    data = parse_result_file(args.file)
    rates = compute_pair_error_rates(data)

    labels = ["Start", "Middle", "End", "Unit"]

    # Print rates for quick inspection
    print(f"Error rates: {[round(r, 6) for r in rates]}")

    plot_error_rates(
        labels=labels,
        values=rates,
        title=args.title,
        output_path=args.out,
    )
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()


