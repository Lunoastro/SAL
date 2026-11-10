"""A Self-check for Cross-Segment Mixing (CSM).

Every audio frame is tagged with 10 * source_sample + label, so the output
reveals which utterance each frame came from and whether audio and labels are
still aligned.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trainers.sal_trainer import SALTrainer

B, T, SCALE = 32, 25, 4


def make_batch():
    labels = torch.randint(0, 2, (B, T)).float()
    src = torch.arange(B).float()[:, None]
    inputs = (10 * src + labels).repeat_interleave(SCALE, dim=1)
    lengths = torch.randint(10, 40, (B,))  # train pad keeps original length (but can exceed T)
    return [f"u{i}" for i in range(B)], inputs, labels, lengths


def run(csm_fix):
    trainer = SimpleNamespace(mixup_ratio=0.2, csm_fix=csm_fix)
    trainer._csm_batch = lambda b: SALTrainer._csm_batch(trainer, b)
    batch = make_batch()
    out = batch
    for _ in range(2):  # S7 = two rounds (mixup2)
        out = SALTrainer._mixup_batch(trainer, out)
    return batch, out


torch.manual_seed(0)

# Upstream (csm_fix=False): each sample is spliced with itself -> identity (finding F1)
(_, x, y, _), (_, x2, y2, _) = run(csm_fix=False)
assert torch.equal(x, x2) and torch.equal(y, y2), "upstream CSM changed the batch"
print("upstream CSM is a no-op: confirmed")

# My proposed Fix: frames come from a different utterance, audio and labels stay aligned
(_, x, y, _), (_, x2, y2, _) = run(csm_fix=True)
frames = x2.view(B, T, SCALE)
assert (frames == frames[..., :1]).all(), "cut not frame-aligned"
assert torch.equal(frames[..., 0] % 10, y2), "audio/label misaligned"
assert ((frames[..., 0] // 10) != torch.arange(B)[:, None]).any(), "no cross-utterance splice"
print("fixed CSM: cross-utterance, aligned, frame-aligned: ok")
