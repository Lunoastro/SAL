import csv, json, pickle, re, sys
from pathlib import Path
import numpy as np

POS = ["spoof_start", "spoof_middle", "spoof_end", "spoof_unit",
       "genuine_start", "genuine_middle", "genuine_end", "genuine_unit"]


def positions(y):
    """SAL's 8 frame categories (label_generators.seg2bd_label_new), per recording."""
    c = np.empty(len(y), np.int8)
    b = np.flatnonzero(np.diff(y)) + 1
    for s, e in zip(np.r_[0, b], np.r_[b, len(y)]):
        base = 0 if y[s] == 0 else 4
        if e - s == 1:
            c[s] = base + 3
        else:
            c[s], c[e - 1], c[s + 1:e - 1] = base, base + 2, base + 1
    return c


def eer(score, y):
    """EER (%) and threshold; decision genuine iff score > threshold."""
    o = np.argsort(score, kind="stable")
    s, yy = score[o], y[o]
    frr = np.cumsum(yy) / yy.sum()                       # genuine at or below threshold
    far = 1 - np.cumsum(1 - yy) / (len(yy) - yy.sum())   # spoof above threshold
    i = np.argmin(np.abs(frr - far))
    return 100 * (frr[i] + far[i]) / 2, float(s[i])


def load(d):
    utts = pickle.load(open(d / "frame_scores.pkl", "rb"))
    ids = [u for u, _, _ in utts]
    score = [l[:, 1] - l[:, 0] for _, l, _ in utts]
    y = [lab.astype(np.int8) for _, _, lab in utts]
    return ids, score, y


def rates(pred, y):
    g, s = y == 1, y == 0
    return dict(n_genuine=int(g.sum()), n_spoof=int(s.sum()),
                false_alarm=100 * float((pred[g] == 0).mean()) if g.any() else "",
                miss=100 * float((pred[s] == 1).mean()) if s.any() else "")


def lps_subset(utt):
    """'dev01-yourtts-partial-oa_...' -> 'yourtts/partial' ; anything else -> 'other'."""
    p = utt.split("-")
    return f"{p[1]}/{p[2].split('_')[0]}" if len(p) >= 3 else "other"


def result_txt(d):
    t = (d / "result.txt").read_text()
    m = re.search(r"EER=([\d.]+)", t)
    totals = [float(x) for x in re.findall(r"^[0-7]: acc: [\d.]+, total_samples: ([\d.]+)", t, re.M)]
    return (float(m.group(1)) if m else None), totals


def main(root):
    root = Path(root)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    runs = sorted(p.parent for p in root.glob("*/*/frame_scores.pkl"))
    res_of = lambda d: d.name.split("-")[-1] if d.name.split("-")[0] in ("dev", "eval") else "0.02"
    rows, thr, data = [], {}, {}

    for d in runs:  # PS-dev thresholds first
        model, name = d.parent.name, d.name
        ids, score, y = data[d] = load(d)
        sc, yy = np.concatenate(score), np.concatenate(y)
        e, t = eer(sc, yy) if 0 < yy.sum() < len(yy) else (float("nan"), float("nan"))  # single-class run
        if name.startswith("dev-"):
            thr.setdefault(model, {})[res_of(d)] = t
        ref_eer, ref_tot = result_txt(d)
        tot = np.bincount(np.concatenate([positions(v) for v in y]), minlength=8)
        print(f"{model}/{name}: {len(ids)} utts, {len(yy)} frames, EER {e:.4f} (result.txt {ref_eer}), "
              f"position counts {'MATCH' if list(tot) == ref_tot else 'differ: ' + str(ref_tot)}")
        rows.append(dict(model=model, run=name, res=res_of(d), cutoff="own_eer", threshold=t,
                         group="all", eer=e, **rates((sc > t).astype(np.int8), yy)))
        us = np.array([v.min() for v in score]); uy = np.array([int(v.min() == 1) for v in y], np.int8)
        ue, ut = eer(us, uy) if 0 < uy.sum() < len(uy) else (float("nan"), float("nan"))
        rows.append(dict(model=model, run=name, res=res_of(d), cutoff="own_eer", threshold=ut,
                         group="recording", eer=ue, **rates((us > ut).astype(np.int8), uy)))
        rows.append(dict(model=model, run=name, res=res_of(d), cutoff="argmax", threshold=0.0,
                         group="recording", **rates((us > 0).astype(np.int8), uy)))
        print(f"    recording-level EER {ue:.2f} ({int(uy.sum())} genuine / {int(len(uy) - uy.sum())} with spoof)")

    json.dump(thr, open(out / "thresholds.json", "w"), indent=2)

    for d, (ids, score, y) in data.items():
        model, name, res = d.parent.name, d.name, res_of(d)
        sc, yy = np.concatenate(score), np.concatenate(y)
        pos = np.concatenate([positions(v) for v in y])
        sub = None
        if name.startswith("lps"):
            names, code = np.unique([lps_subset(u) for u in ids], return_inverse=True)
            sub = np.repeat(code, [len(v) for v in y])
        cutoffs = {"argmax": 0.0}
        if res in thr.get(model, {}) and not name.startswith("dev"):
            cutoffs["frozen_ps_dev"] = thr[model][res]
        for cname, t in cutoffs.items():
            pred = (sc > t).astype(np.int8)
            base = dict(model=model, run=name, res=res, cutoff=cname, threshold=t)
            rows.append(dict(base, group="all", **rates(pred, yy)))
            for k, lab in enumerate(POS):
                m = pos == k
                rows.append(dict(base, group=f"pos:{lab}", **rates(pred[m], yy[m])))
            if sub is not None:
                for k, g in enumerate(names):
                    m = sub == k
                    rows.append(dict(base, group=f"subset:{g}", **rates(pred[m], yy[m])))

    # pooled vs fine on PS eval: repeat each 160 ms decision 8x on the 20 ms grid, score vs 20 ms labels
    for model in thr:
        fine, coarse = root / model / "eval-0.02", root / model / "eval-0.16"
        if fine not in data or coarse not in data or "0.16" not in thr[model]:
            continue
        f_ids, _, f_y = data[fine]
        c_map = {u: s for u, s in zip(data[coarse][0], data[coarse][1])}
        y20, p20 = [], []
        for u, lab in zip(f_ids, f_y):
            rep = np.repeat(c_map[u], 8)[:len(lab)]
            rep = np.pad(rep, (0, len(lab) - len(rep)), mode="edge")  # ponytail: edge-pad short tails
            y20.append(lab); p20.append(rep)
        y20, s20 = np.concatenate(y20), np.concatenate(p20)
        for cname, t in (("argmax", 0.0), ("frozen_ps_dev", thr[model]["0.16"])):
            rows.append(dict(model=model, run="eval-0.16-on-0.02-grid", res="0.16->0.02", cutoff=cname,
                             threshold=t, group="all", **rates((s20 > t).astype(np.int8), y20)))

    # split-audio copies (make_split_audio.py): argv[2] = manifest; ROOT/<model>/split_{genuine,spoof}
    # are compared with the same frames scored inside their original recordings (ROOT/<model>/eval-0.02)
    man = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    for model in (thr if man else []):
        ctx = root / model / "eval-0.02"
        if ctx not in data:
            continue
        by_utt = {u: (s, y) for u, s, y in zip(*data[ctx])}
        for kind in ("genuine", "spoof"):
            seg = [l.split("\t") for l in open(man).read().split("\n")[1:] if l.split("\t")[1:2] == [kind]]
            if not seg:
                continue
            s_ = np.concatenate([by_utt[u][0][int(a):int(b)] for u, _, a, b in seg])
            y_ = np.concatenate([by_utt[u][1][int(a):int(b)] for u, _, a, b in seg])
            for cname, t in (("argmax", 0.0), ("frozen_ps_dev", thr[model].get("0.02", 0.0))):
                rows.append(dict(model=model, run=f"split_{kind}-in_context", res="0.02", cutoff=cname,
                                 threshold=t, group="all", **rates((s_ > t).astype(np.int8), y_)))

    cols = ["model", "run", "res", "cutoff", "threshold", "group", "eer",
            "n_genuine", "n_spoof", "false_alarm", "miss"]
    with open(out / "frame_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {out/'thresholds.json'} and {out/'frame_analysis.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    assert positions(np.array([0, 0, 0, 1, 0, 1, 1])).tolist() == [0, 1, 2, 7, 3, 4, 6]
    assert eer(np.array([-2., -1, 1, 2]), np.array([0, 0, 1, 1]))[0] == 0
    main(sys.argv[1])
