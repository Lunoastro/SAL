"""
A stretch is a continuous run of frames with the same label inside one recording (SAL's frame
categories: first frame = start, last = end, rest = middle, a one-frame stretch = unit).
Per file it prints
  - stretch counts, median / mean length (s) and single-frame share, per class;
  - where single-frame stretches sit: at a recording edge, or between two stretches of the other class;
  - the error rate (argmax, score = logit[1] - logit[0] > 0 means genuine) of start and end frames,
    split into "edge" (first / last frame of the recording) and "splice" (next to a label change,
    i.e. a splice up to the 20 ms grid). Counts match frame_analysis.py's start/end categories.

    python scripts/stretch_stats.py OUT/S3/eval-0.02/frame_scores.pkl OUT/S7/eval-0.02/frame_scores.pkl [--frame 0.02]
"""
import argparse, pickle
from collections import defaultdict
import numpy as np

CLS = {0: "spoof", 1: "genuine"}


def stretches(y):
    b = np.flatnonzero(np.diff(y)) + 1
    return zip(np.r_[0, b], np.r_[b, len(y)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", nargs="+"); ap.add_argument("--frame", type=float, default=0.02)
    a = ap.parse_args()
    for path in a.pkl:
        lengths, unit_place, err = defaultdict(list), defaultdict(int), defaultdict(lambda: [0, 0])
        for _, logits, y in pickle.load(open(path, "rb")):
            y = np.asarray(y).astype(int); wrong = (logits[:, 1] - logits[:, 0] > 0).astype(int) != y
            for s, e in stretches(y):
                c = CLS[y[s]]; lengths[c].append(e - s); edge_s, edge_e = s == 0, e == len(y)
                if e - s == 1:
                    unit_place[c, "edge" if edge_s or edge_e else "between other class"] += 1
                    continue
                for side, i, at_edge in (("start", s, edge_s), ("end", e - 1, edge_e)):
                    k = err[c, side, "edge" if at_edge else "splice"]; k[0] += 1; k[1] += int(wrong[i])
        print(f"== {path}")
        for c in ("spoof", "genuine"):
            n = np.array(lengths[c])
            if not len(n):
                continue
            units = {p: unit_place[c, p] for p in ("edge", "between other class")}
            print(f"  {c:7} stretches {len(n):7d}  median {np.median(n) * a.frame:.2f} s  mean {n.mean() * a.frame:.2f} s  "
                  f"single-frame {100 * (n == 1).mean():.1f}% {units}")
        for (c, side, place), (k, w) in sorted(err.items()):
            print(f"  {c:7} {side:5} {place:6}: {k:7d} frames, {100 * w / k:5.1f}% misclassified")


if __name__ == "__main__":
    y = np.array([1, 1, 0, 1, 1, 0, 0])
    assert list(map(tuple, stretches(y))) == [(0, 2), (2, 3), (3, 5), (5, 7)]
    main()
