import argparse
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluator.evaluator import F1Metric
from src.models.net.model import BaseSSLModel
from src.trainers.baseline_trainer import BaselineTrainer
from src.trainers.csm_trainer import CSMTrainer
from src.trainers.sal_trainer import SALTrainer
from src.trainers.spl_trainer import SPLTrainer
from src.trainers.transition_trainer import TransitionTrainer
from src.trainers.label_generators import SPLLabelGenerator


def check_labels(device):
    # Exhaust all short binary sequences, including singletons and edge runs.
    for width in range(1, 9):
        labels = torch.tensor(list(product((0, 1), repeat=width)), device=device).float()
        segments, lengths = SPLLabelGenerator._seg2bd_label_new(labels)
        expected = torch.tensor(np.stack(SPLLabelGenerator.seg2bd_label_new(segments, lengths)),
                                device=device).squeeze(-1).long()
        actual = SPLLabelGenerator.positional_labels(labels)
        assert actual.device == labels.device and actual.dtype == torch.long
        assert torch.equal(actual, expected), f"SPL differs at width {width}"
        assert torch.equal(SPLLabelGenerator.positional_labels(labels[0]), expected[0])
    # Preserve padding behavior: a valid false singleton followed by false padding
    # remains the start of a longer run. Fixing that is a separate experiment.
    padded = torch.tensor([[1, 0, 0, 0]], device=device)
    assert SPLLabelGenerator.positional_labels(padded).tolist() == [[7, 0, 1, 2]]


def check_masks(device):
    labels = torch.randint(0, 2, (5, 9), device=device).float()
    preds = torch.randn(5, 9, 2, device=device, requires_grad=True)
    lengths = torch.tensor([0, 1, 5, 9, 14], device=device)
    expected_mask = torch.ones_like(labels, dtype=torch.bool)
    for i, length in enumerate(lengths):
        expected_mask[i, length:] = False
    expected_preds = torch.cat([preds[i, :length] for i, length in enumerate(lengths)])
    expected_labels = torch.cat([labels[i, :length] for i, length in enumerate(lengths)])
    for cls in (BaselineTrainer, CSMTrainer, SALTrainer, SPLTrainer, TransitionTrainer):
        trainer = SimpleNamespace()
        trainer._get_label_mask = lambda labs, lens: cls._get_label_mask(trainer, labs, lens)
        for lens in (lengths, lengths.cpu()):
            assert torch.equal(trainer._get_label_mask(labels, lens), expected_mask)
            p, y = cls._get_pred_label(trainer, preds, labels, lens)
            assert torch.equal(p, expected_preds) and torch.equal(y, expected_labels)
            grad = torch.autograd.grad(p.sum(), preds, retain_graph=True)[0]
            assert torch.equal(grad, expected_mask[..., None].expand_as(preds).float())


def legacy_f1(logits, targets, percent):
    pred = logits.argmax(dim=1)
    correct = (pred == targets).sum().float()
    acc = correct / targets.numel() if targets.numel() else correct
    tp = (pred * targets).sum().float()
    fp = (pred * (1 - targets)).sum().float()
    fn = ((1 - pred) * targets).sum().float()
    zero = torch.zeros_like(tp)
    precision = tp / (tp + fp) if tp + fp > 0 else zero
    recall = tp / (tp + fn) if tp + fn > 0 else zero
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else zero
    return (acc * 100, f1 * 100) if percent else (acc, f1)


def check_f1(device):
    random_logits = torch.randn(1031, 2, device=device)
    random_logits[:10] = 0  # Preserve argmax tie behavior.
    cases = [(random_logits, torch.randint(0, 2, (1031,), device=device).float())]
    for pred, target in product((0, 1), repeat=2):
        logits = torch.zeros(37, 2, device=device)
        logits[:, pred] = 1
        cases.append((logits, torch.full((37,), target, device=device).float()))
    cases.append((torch.empty(0, 2, device=device), torch.empty(0, device=device)))
    for percent in (False, True):
        metric = F1Metric(percent).to(device)
        for logits, targets in cases:
            metric.reset()
            for p, y in zip(logits.split(17), targets.split(17)):
                metric.update(p, y)
            actual = metric.compute()
            expected = legacy_f1(logits, targets, percent)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=1e-6, atol=1e-6)
            states = [getattr(metric, n) for n in ("correct", "total", "tp", "fp", "fn")]
            assert sum(v.numel() for v in states) == 5
            assert all(not v.requires_grad for v in states)
        metric.reset()
        assert all(getattr(metric, n).item() == 0 for n in ("correct", "total", "tp", "fp", "fn"))
    # Summing worker states must reproduce the single-worker result.
    left, right, merged = (F1Metric().to(device) for _ in range(3))
    logits, targets = cases[0]
    left.update(logits[:500], targets[:500])
    right.update(logits[500:], targets[500:])
    merged.update(logits, targets)
    for name in ("correct", "total", "tp", "fp", "fn"):
        setattr(left, name, getattr(left, name) + getattr(right, name))
    for a, e in zip(left.compute(), merged.compute()):
        torch.testing.assert_close(a, e)


def check_features(device):
    # A fake encoder isolates aggregation without loading SSL weights.
    for dtype in (torch.float32, torch.bfloat16):
        layers = [torch.randn(2, 7, 11, device=device, dtype=dtype, requires_grad=True)
                  for _ in range(25)]
        weights = torch.randn(25, device=device, requires_grad=True)
        model = SimpleNamespace(weight_layer=weights,
                                ssl_encoder=SimpleNamespace(extract_feat=lambda _: layers))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            expected = (torch.stack(layers) * weights.view(25, 1, 1, 1)).sum(0)
            actual = BaseSSLModel._extract_features(model, torch.zeros(2, 20, device=device))
        assert actual.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        probe = torch.randn_like(actual)
        old_grad = torch.autograd.grad((expected * probe).sum(), [weights, *layers])
        new_grad = torch.autograd.grad((actual * probe).sum(), [weights, *layers])
        for a, e in zip(new_grad, old_grad):
            torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-5)
    plain = SimpleNamespace(ssl_encoder=SimpleNamespace(extract_feat=lambda _: layers[-1]))
    assert BaseSSLModel._extract_features(plain, torch.zeros(2, 20, device=device)) is layers[-1]
    model.weight_layer = weights[:24]
    try:
        BaseSSLModel._extract_features(model, torch.zeros(2, 20, device=device))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched layer count accepted")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='also check CUDA on a compute node')
    args = parser.parse_args()
    torch.manual_seed(666)
    devices = [torch.device('cpu')]
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA requested but unavailable'
        devices.append(torch.device('cuda'))
    for device in devices:
        check_labels(device)
        check_masks(device)
        check_f1(device)
        check_features(device)
        print(f'{device}: SPL/padding, masks/gradients, streaming F1, SSL outputs/gradients passed')
