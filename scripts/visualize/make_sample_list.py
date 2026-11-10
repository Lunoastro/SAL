import argparse
import os
import random
import re
from collections import defaultdict

import numpy as np

# DO NOT CHANGE IT <-- after figures are generated 
DEFAULT_SEED = 20261029

# Native PartialSpoof label resolution; anything coarser distorts edit durations.
FINE_RESOLUTION = 0.02

# Fixed duration bin edges in seconds (total spoofed time per utterance).
SHORT_MAX_S = 0.32
MEDIUM_MAX_S = 1.0

POSITIONS = ("start", "middle", "end")
DURATIONS = ("short", "medium", "long")


def runs(labels):
    """Return [(start, end)] index ranges (end exclusive) of consecutive spoof (0) frames."""
    spoof = np.asarray(labels, dtype=int) == 0
    if not spoof.any():
        return []
    d = np.diff(np.r_[0, spoof.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def describe(labels, resolution):
    """Classify one utterance.

    Returns (group, duration_bin, spoof_seconds, n_spoof_runs) where group is one of
    start / middle / end (single spoof run), "genuine", "all_spoof" or "multi_run".
    n_spoof_runs is computed here from the labels (the label files have no such column).
    """
    labels = np.asarray(labels, dtype=int)
    spoof_runs = runs(labels)
    n_runs = len(spoof_runs)
    spoof_s = float((labels == 0).sum()) * resolution
    if n_runs == 0:
        return "genuine", None, 0.0, 0
    if n_runs > 1:
        return "multi_run", None, spoof_s, n_runs
    s, e = spoof_runs[0]
    if s == 0 and e == len(labels):
        return "all_spoof", None, spoof_s, 1
    position = "start" if s == 0 else "end" if e == len(labels) else "middle"
    if spoof_s < SHORT_MAX_S:
        duration = "short"
    elif spoof_s < MEDIUM_MAX_S:
        duration = "medium"
    else:
        duration = "long"
    return position, duration, spoof_s, 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels_path", required=True,
                        help="segment label .npy (dict utt_id -> 0/1 array), e.g. eval_seglab_0.02.npy")
    parser.add_argument("--resolution", type=float, default=FINE_RESOLUTION,
                        help="seconds per label in --labels_path (default 0.02; must match the file)")
    parser.add_argument("--allow_coarse_labels", action="store_true",
                        help="accept labels coarser than 0.02 s (duration bins become biased; warns)")
    parser.add_argument("--per_stratum", type=int, default=3,
                        help="utterances drawn per position|duration stratum (default 3)")
    parser.add_argument("--n_genuine", type=int, default=3,
                        help="fully genuine control utterances drawn (default 3; 0 disables)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default {DEFAULT_SEED}); report it with the figures")
    parser.add_argument("--out", required=True, help="output TSV path")
    args = parser.parse_args()
    if args.per_stratum < 1 or args.n_genuine < 0 or args.resolution <= 0:
        parser.error("--per_stratum must be >= 1, --n_genuine >= 0 and --resolution > 0")

    # Guard 1: the file name says which resolution it holds (e.g. eval_seglab_0.16.npy);
    # refuse an obvious mismatch with --resolution, which would mis-scale every duration (doubtful that this will happen but better to be certain)
    m = re.search(r"seglab_([0-9.]+?)(?:\.npy)?$", os.path.basename(args.labels_path))
    if m:
        try:
            file_res = float(m.group(1).rstrip("."))
        except ValueError:
            file_res = None
        if file_res is not None and abs(file_res - args.resolution) > 1e-9:
            parser.error(f"{args.labels_path} looks like {file_res:g} s labels but --resolution is "
                         f"{args.resolution:g}; pass the matching --resolution")
    # Guard 2: coarse labels inflate short edits (see module docstring), so the duration
    # bins would be wrong <-- Warning only if --allow_coarse_labels is passed, otherwise error out (for now)
    warnings = []
    if args.resolution > FINE_RESOLUTION + 1e-9:
        if not args.allow_coarse_labels:
            parser.error(f"--resolution {args.resolution:g} is coarser than {FINE_RESOLUTION:g} s: "
                         "a coarse segment is spoof if any part is, so short edits look longer and "
                         "land in the wrong duration bin. Use the 0.02 s labels "
                         "(or --allow_coarse_labels to override).")
        warnings.append(f"labels at {args.resolution:g} s (coarser than {FINE_RESOLUTION:g} s): "
                        "spoof durations are overestimated and duration bins are biased")

    labels = np.load(args.labels_path, allow_pickle=True).item()
    strata = defaultdict(list)
    excluded = defaultdict(int)
    info = {}
    for utt_id in sorted(labels):
        group, duration, spoof_s, n_runs = describe(labels[utt_id], args.resolution)
        dur_s = len(labels[utt_id]) * args.resolution
        if group in POSITIONS:
            key = f"{group}|{duration}"
        elif group == "genuine":
            key = "genuine"
        else:  # multi_run / all_spoof
            excluded[group] += 1
            continue
        strata[key].append(utt_id)
        info[utt_id] = (key, spoof_s, n_runs, dur_s)

    # Fixed visiting order: genuine control first, then all 9 position|duration strata
    # (listed even when empty, so missing strata are visible). One RNG seeded once, and
    # each pool sorted by ID before shuffling -> the draw is fully determined by
    # (labels file, resolution, seed, per_stratum, n_genuine).
    plan = ([("genuine", args.n_genuine)] if args.n_genuine > 0 else []) + \
           [(f"{p}|{d}", args.per_stratum) for p in POSITIONS for d in DURATIONS]
    rng = random.Random(args.seed)
    chosen = []
    for key, n in plan:
        pool = sorted(strata.get(key, []))
        if len(pool) < n:
            warnings.append(f"stratum {key} has only {len(pool)} utterance(s) < requested {n}")
        rng.shuffle(pool)
        chosen.extend(pool[:n])

    for w in warnings:
        print(f"WARNING: {w}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# generated by scripts/visualize/make_sample_list.py\n")
        f.write(f"# labels_path={os.path.abspath(args.labels_path)} resolution={args.resolution:g}\n")
        f.write(f"# seed={args.seed} per_stratum={args.per_stratum} n_genuine={args.n_genuine}\n")
        f.write(f"# position strata use utterances with n_spoof_runs == 1 only; "
                f"duration bins: short<{SHORT_MAX_S:g}s<=medium<{MEDIUM_MAX_S:g}s<=long\n")
        size_keys = ["genuine"] + [f"{p}|{d}" for p in POSITIONS for d in DURATIONS]
        f.write("# stratum sizes: " + ", ".join(f"{k}={len(strata.get(k, []))}" for k in size_keys) + "\n")
        f.write("# excluded (no single position): " +
                (", ".join(f"{k}={v}" for k, v in sorted(excluded.items())) or "none") + "\n")
        for w in warnings:
            f.write(f"# WARNING: {w}\n")
        f.write("utt_id\tstratum\tspoof_seconds\tn_spoof_runs\tduration_seconds\n")
        for utt_id in chosen:
            key, spoof_s, n_runs, dur = info[utt_id]
            f.write(f"{utt_id}\t{key}\t{spoof_s:.2f}\t{n_runs}\t{dur:.2f}\n")
    print(f"Wrote {len(chosen)} utterances to {args.out} (seed {args.seed}, "
          f"{len(warnings)} warning(s))")


if __name__ == "__main__":
    main()
