"""
Split-audio copies of a PartialSpoof split for the controlled context test:
from every recording that contains both classes keep only its longest continuous GENUINE run
(split <split>_genuine) or its longest continuous SPOOFED run (split <split>_spoof). Cuts follow
20 ms label boundaries, so the copies contain no splice point. Single-class recordings are skipped.

Writes OUT/<split>_{genuine,spoof}/<utt>.wav, OUT/segment_labels/<split>_{genuine,spoof}_seglab_{0.02,0.16}.npy
and OUT/<split>_split_manifest.tsv (utt, kind, start, end in 20 ms frames of the original recording).
Evaluate with PS_ROOT=OUT and PART=<split>_genuine / <split>_spoof in eval_lps.sbatch.

    python scripts/make_split_audio.py --ps_root $PARTIALSPOOF_ROOT --out /scratch/$USER/ps-split
"""
import argparse, math
from pathlib import Path
import numpy as np
import soundfile as sf

HOP = 320  # samples per 20 ms label at 16 kHz


def longest_run(y, cls):
    """(start, end) of the longest run of value cls in y, end exclusive; None if absent."""
    m = np.r_[0, (y == cls).astype(int), 0]
    starts, ends = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    if not len(starts):
        return None
    i = np.argmax(ends - starts)
    return int(starts[i]), int(ends[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps_root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--min_frames", type=int, default=25, help="skip runs shorter than this (25 = 0.5 s)")
    a = ap.parse_args()
    root, out = Path(a.ps_root), Path(a.out)
    labels = np.load(root / "segment_labels" / f"{a.split}_seglab_0.02.npy", allow_pickle=True).item()
    wavs = {p.stem: p for p in (root / a.split).rglob("*.wav")}
    kinds = {"genuine": 1, "spoof": 0}
    new = {k: {} for k in kinds}
    (out / "segment_labels").mkdir(parents=True, exist_ok=True)
    for k in kinds:
        (out / f"{a.split}_{k}").mkdir(exist_ok=True)
    skipped = 0
    with open(out / f"{a.split}_split_manifest.tsv", "w") as man:
        man.write("utt_id\tkind\tstart\tend\n")
        for utt, y in sorted(labels.items()):
            y = np.asarray(y).astype(int)
            if y.min() == y.max():
                continue
            audio, sr = sf.read(wavs[utt])
            assert sr == 16000, (utt, sr)
            for k, cls in kinds.items():
                r = longest_run(y, cls)
                if r is None or r[1] - r[0] < a.min_frames:
                    skipped += 1
                    continue
                s, e = r
                sf.write(out / f"{a.split}_{k}" / f"{utt}.wav", audio[s * HOP:e * HOP], sr, subtype="PCM_16")
                new[k][utt] = e - s
                man.write(f"{utt}\t{k}\t{s}\t{e}\n")
    for k, cls in kinds.items():
        for res, step in (("0.02", 1), ("0.16", 8)):  # pure runs: same class at any resolution
            np.save(out / "segment_labels" / f"{a.split}_{k}_seglab_{res}.npy",
                    {u: np.full(math.ceil(n / step), cls, dtype=int) for u, n in new[k].items()})
        print(f"{a.split}_{k}: {len(new[k])} recordings, {sum(new[k].values())} frames (20 ms)")
    print(f"skipped {skipped} runs shorter than {a.min_frames} frames")


if __name__ == "__main__":
    assert longest_run(np.array([1, 1, 0, 0, 0, 1]), 0) == (2, 5) and longest_run(np.array([1, 1]), 0) is None
    main()
