"""
Grouped bar charts from any long-format table (CSV) or nested JSON, for thesis figures.

Not tied to one analysis: rows are filtered and mapped to the chart by column name.
    --where COL=V1,V2   keep rows whose COL is one of the values (repeatable)
    --where COL~REGEX   keep rows whose COL matches the regex
    --x COL             categories on the x axis          --hue COL    bar colour
    --hatch COL         second grouping as texture        --facet COL  one panel column per value
    --y COL [COL ...]   one panel row per metric (never two scales on one axis);
                        A|B = first non-empty of A, B (e.g. false_alarm|miss -> one error panel)
    --order COL=V1,V2   fix category order (also fixes colours: colour follows the value)
    --sub REGEX=REPL    rewrite tick/legend labels, e.g. '^pos:=' (repeatable)
    --values            print values on bars    --fmt "{:.2f}" value format    --ylabel TEXT    --title TEXT   --out FILE.pdf|png
Nested JSON {a: {b: v}} is flattened to columns level0, level1, ..., value.
Missing or empty cells are skipped (no bar).

Example (frame_analysis.csv):
    python scripts/plot_frames.py frame_analysis.csv --where group=all --where cutoff=argmax,frozen_ps_dev \\
        --where run=eval-0.02,lps-cf,lps-cp,lps-oa --x run --hue cutoff --facet model \\
        --y false_alarm miss --ylabel "Error rate (%)" --values --out fig_threshold_regimes.pdf
"""
import argparse, csv, json, re
from functools import reduce
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # validated, fixed order
HATCHES = [None, "///", "\\\\\\", "xx", "..", "++"]
INK, MUTED, SURFACE = "#1f1f1f", "#6b6b6b", "#ffffff"


def load(path):
    if path.endswith(".json"):
        rows = []
        def walk(node, keys):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, keys + [k])
            else:
                rows.append({**{f"level{i}": k for i, k in enumerate(keys)}, "value": node})
        walk(json.load(open(path)), [])
        return rows
    return list(csv.DictReader(open(path, newline="")))


def keep(row, cond):
    if "~" in cond and ("=" not in cond or cond.index("~") < cond.index("=")):
        col, pat = cond.split("~", 1)
        return re.search(pat, str(row.get(col, ""))) is not None
    col, vals = cond.split("=", 1)
    return str(row.get(col, "")) in vals.split(",")


def levels(rows, col, order):
    seen = list(dict.fromkeys(str(r[col]) for r in rows)) if col else [None]
    want = order.get(col)
    return [v for v in want if v in seen] if want else seen


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table")
    ap.add_argument("--where", action="append", default=[])
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", nargs="+", required=True)
    ap.add_argument("--hue"); ap.add_argument("--hatch"); ap.add_argument("--facet")
    ap.add_argument("--order", action="append", default=[])
    ap.add_argument("--sub", action="append", default=[])
    ap.add_argument("--values", action="store_true"); ap.add_argument("--fmt", default="{:.1f}")
    ap.add_argument("--ylabel", default=None); ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [r for r in load(a.table) if all(keep(r, c) for c in a.where)]
    if not rows:
        raise SystemExit("no rows left after --where filters")
    order = {c.split("=", 1)[0]: c.split("=", 1)[1].split(",") for c in a.order}
    subs = [s.split("=", 1) for s in a.sub]
    label = lambda v: reduce(lambda t, s: re.sub(s[0], s[1], t), subs, str(v))

    xs, hues, hatches, facets = (levels(rows, c, order) for c in (a.x, a.hue, a.hatch, a.facet))
    if len(hues) > len(PALETTE) or len(hatches) > len(HATCHES):
        raise SystemExit("too many groups for distinct colours/textures; filter or use --facet")
    groups = list(product(hues, hatches))
    width = 0.8 / len(groups)

    fig, axes = plt.subplots(len(a.y), len(facets), squeeze=False, sharey="row",
                             figsize=(max(4.5, 0.5 * len(xs) * len(groups) * len(facets) + 2), 2.8 * len(a.y)))
    fig.patch.set_facecolor(SURFACE)
    for (i, metric), (j, facet) in product(enumerate(a.y), enumerate(facets)):
        ax = axes[i][j]
        for g, (hue, hatch) in enumerate(groups):
            for k, x in enumerate(xs):
                vals = [next((num(r.get(m)) for m in metric.split("|") if num(r.get(m)) is not None), None) for r in rows
                        if str(r[a.x]) == x and (hue is None or str(r[a.hue]) == hue)
                        and (hatch is None or str(r[a.hatch]) == hatch)
                        and (facet is None or str(r[a.facet]) == facet)]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                if len(vals) > 1:
                    raise SystemExit(f"{len(vals)} rows for one bar ({metric}, x={x}, hue={hue}, hatch={hatch}, "
                                     f"facet={facet}); add a --where filter or another grouping")
                pos = k - 0.4 + width * (g + 0.5)
                ax.bar(pos, vals[0], width, color=PALETTE[hues.index(hue)] if hue else PALETTE[0],
                       hatch=HATCHES[hatches.index(hatch)] if hatch else None,
                       edgecolor=SURFACE, linewidth=1,
                       label=" / ".join(label(v) for v in (hue, hatch) if v is not None))
                if a.values:
                    up = vals[0] >= 0
                    ax.annotate(a.fmt.format(vals[0]), (pos, vals[0]), xytext=(0, 2 if up else -2),
                                textcoords="offset points", ha="center", va="bottom" if up else "top",
                                fontsize=7, color=INK)
        ax.set_xticks(range(len(xs)), [label(x) for x in xs], rotation=30 if len(xs) > 4 else 0,
                      ha="right" if len(xs) > 4 else "center", rotation_mode="anchor", color=INK, fontsize=8)
        ax.set_title(" · ".join(t for t in (label(facet) if facet else "", label(metric)) if t), color=INK, fontsize=10)
        if j == 0:
            ax.set_ylabel(a.ylabel or "", color=INK)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK)
    if len(groups) > 1:  # legend for >= 2 series; deduplicated handles
        h, l = axes[0][-1].get_legend_handles_labels()
        uniq = dict(zip(l, h))
        fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=min(len(uniq), 4), frameon=False,
                   bbox_to_anchor=(0.5, 1.0), labelcolor=INK, fontsize=8)
    if a.title:
        fig.suptitle(a.title, color=INK, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if len(groups) > 1 else 1))
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {a.out} ({len(rows)} rows, {len(xs)} x {len(groups)} groups, {len(a.y)}x{len(facets)} panels)")


if __name__ == "__main__":
    main()
