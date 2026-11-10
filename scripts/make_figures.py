"""
The Addition of this script is only for the purpose of reproducing the figures in the thesis. It is not intended to be a general-purpose plotting script.

Thesis figures from frame_analysis.csv (scripts/frame_analysis.py), drawn with plot_frames.py.

Run names in the CSV are the evaluation output folder names. Map yours in the block below;
figures whose runs are missing are skipped, and models are taken from the CSV.

    python scripts/make_figures.py ANALYSIS_DIR [--ext pdf|png] [--models S3,S7,S7_fixed]
"""
import argparse, csv, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import plot_frames

# ---- adapt to your folder names ------------------------------------------------------------
PS, PS_COARSE = "eval-0.02", "eval-0.16"           # PartialSpoof eval at 20 ms and 160 ms
LPS = ["lps-cf", "lps-cp", "lps-oa"]; LPS_MAIN = "lps-cf"
SPLIT = ["split_genuine-in_context", "split_genuine", "split_spoof-in_context", "split_spoof"]
NAMES = {PS: "PartialSpoof eval\n(20 ms)", PS_COARSE: "PartialSpoof eval\n(160 ms)", "lps-cf": "LPS crossfade",
         "lps-cp": "LPS cut/paste", "lps-oa": "LPS overlap-add",
         SPLIT[0]: "genuine stretch\nin its recording", SPLIT[1]: "genuine stretch\nisolated",
         SPLIT[2]: "spoofed stretch\nin its recording", SPLIT[3]: "spoofed stretch\nisolated"}
# ---- labels for values written by frame_analysis.py (no need to change) ---------------------
FIXED = {"argmax": "argmax rule: spoof if P(spoof) > 0.5", "frozen_ps_dev": "threshold fixed once on PartialSpoof dev",
         "false_alarm": "False alarms: genuine frames classified as spoof",
         "miss": "Misses: spoofed frames classified as genuine", "false_alarm|miss": "% of frames misclassified",
         "eer": "Equal error rate (best threshold per test set)", "subset:other": "fully genuine"}
REGEX = ["^pos:=", "^(spoof|genuine)_=\\1\n", "^subset:=", "/full$=\nfully spoofed", "/partial$=\npartially spoofed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--ext", default="pdf"); ap.add_argument("--models")
    a = ap.parse_args()
    table, out = str(Path(a.dir) / "frame_analysis.csv"), Path(a.dir) / "figures"
    out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(table)))
    models = a.models.split(",") if a.models else list(dict.fromkeys(r["model"] for r in rows))
    with_split = [m for m in models if any(r["model"] == m and r["run"] == SPLIT[1] for r in rows)]
    subs = [f"^{re.escape(k)}$={v}" for k, v in {**NAMES, **FIXED}.items()] + REGEX
    common = [a for s in subs for a in ("--sub", s)] + ["--order", "model=" + ",".join(models),
              "--order", "run=" + ",".join([PS, PS_COARSE, *LPS, *SPLIT])]  # same model colours and run order everywhere
    one = lambda xs: ",".join(xs)
    figs = {
        "fig1_decision_rules": ["--where", "group=all", "--where", "cutoff=argmax,frozen_ps_dev",
            "--where", "run=" + one([PS] + LPS), "--x", "run", "--hue", "model", "--facet", "cutoff",
            "--y", "false_alarm", "miss", "--order", "cutoff=argmax,frozen_ps_dev", "--ylabel", "% of frames",
            "--values", "--title", "Frame errors at two decision rules (20 ms frames)"],
        "fig2_eer": ["--where", "group=all", "--where", "cutoff=own_eer", "--where", "run=" + one([PS, PS_COARSE] + LPS),
            "--x", "run", "--hue", "model", "--y", "eer", "--ylabel", "EER (%)", "--values", "--title", "Frame-level EER"],
        "fig3_positions": ["--where", "cutoff=argmax", "--where", "run=" + one([PS, LPS_MAIN]),
            "--where", "group~^pos:(?!.*unit)", "--x", "group", "--hue", "model", "--facet", "run",
            "--y", "false_alarm|miss", "--ylabel", "% of frames", "--values", "--title",
            "Errors by position: first / inner / last frame of each spoofed or genuine stretch (argmax rule)"],
        "fig4_generators": ["--where", "cutoff=argmax", "--where", "run=" + LPS_MAIN, "--where", "group~^subset:",
            "--x", "group", "--hue", "model", "--y", "false_alarm", "miss", "--ylabel", "% of frames",
            "--title", f"{NAMES.get(LPS_MAIN, LPS_MAIN)} by generator and recording type (argmax rule)"],
        "fig5_split_test": ["--where", "model=" + one(with_split), "--where", "cutoff=argmax", "--where", "group=all",
            "--where", "run=" + one(SPLIT), "--x", "run", "--hue", "model",
            "--y", "false_alarm|miss", "--ylabel", "% of frames", "--values",
            "--title", "Split-audio test: the same frames with and without their surrounding audio"],
        "fig6_recording_level": ["--where", "group=recording", "--where", "cutoff=own_eer",
            "--where", "run=" + one([PS, PS_COARSE] + LPS), "--x", "run", "--hue", "model", "--y", "eer",
            "--ylabel", "EER (%)", "--values", "--title", "Recording level: does the recording contain any spoof?"],
    }
    failed = []
    for name, args in figs.items():
        try:
            plot_frames.main([table, *args, *common, "--out", str(out / f"{name}.{a.ext}")])
        except SystemExit as e:
            if str(e).startswith("no rows left"):  # a run or model was not evaluated
                print(f"skipped {name}: {e}")
            else:  # e.g. two rows for one bar: a data problem, not a missing run
                print(f"ERROR in {name}: {e}", file=sys.stderr); failed.append(name)
    if failed:
        sys.exit(f"{len(failed)} figure(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
