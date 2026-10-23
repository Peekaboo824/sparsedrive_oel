"""Exact regression on REAL frozen outputs/pseudo detections, free of CUDA atomics.

This checks loss algebra/selection/matching/output gradients and an optimizer step.
It does not claim deterministic end-to-end backbone execution.
"""
import argparse
import ast
import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from projects.mmdet3d_plugin.models.sparsedrive import SparseDrive
from projects.mmdet3d_plugin.models.dynamics.dynamics_consistency import selected_predictions


def method(name):
    tree = ast.parse(Path('tests/ctrv_v1/baseline/sparsedrive.py.txt').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseDrive')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = vars(importlib.import_module('projects.mmdet3d_plugin.models.sparsedrive')).copy()
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'baseline', 'exec'), namespace)
    return namespace[name]


class ReplayHead:
    def __init__(self, outputs, pseudo, losses):
        self.outputs, self.pseudo, self.losses = outputs, pseudo, losses
        self.next_frame = False
    def eval(self):
        self.next_frame = True
        return self
    def __call__(self, *args):
        return (self.pseudo, None, None, None, None) if self.next_frame else self.outputs
    def loss(self, *args):
        return self.losses.copy()


def run(golden, original, flags, weight):
    det = copy.deepcopy(golden['det_output'])
    motion = copy.deepcopy(golden['motion'])
    det['prediction'][-1].requires_grad_()
    motion['prediction'][-1].requires_grad_()
    outputs = (det, None, motion, golden['plan_output'], golden['diagnostics'])
    data = {**golden['input_data'], 'next_img': torch.zeros(len(det['prediction'][-1]), 1)}
    gt_losses = {k: v for k, v in golden['losses'].items() if k != 'loss_evo'}
    fake = SimpleNamespace(evo_cfg=dict(TOP_K_CAUSAL_AGENTS=10, loss_weight=weight, **flags),
                           head=ReplayHead(outputs, golden['pseudo_detection'], gt_losses),
                           extract_feat=lambda *a: (None, None) if len(a) > 1 else None)
    pairs = []
    l1 = torch.nn.functional.l1_loss
    def recorded_l1(x, y, *a, **kw):
        pairs.append((x.detach().clone(), y.detach().clone()))
        return l1(x, y, *a, **kw)
    evo = method('get_evo_loss') if original else SparseDrive.get_evo_loss
    fake.get_evo_loss = lambda *args: evo(fake, *args)
    forward = method('forward_train') if original else SparseDrive.forward_train
    with patch('torch.nn.functional.l1_loss', side_effect=recorded_l1):
        losses = forward(fake, torch.zeros(1), **data)
        raw = fake.get_evo_loss(outputs, data, None)['loss_evo']
    parameters = [det['prediction'][-1], motion['prediction'][-1]]
    raw_gradients = torch.autograd.grad(raw, parameters, retain_graph=True)
    optimizer = torch.optim.AdamW(parameters, lr=.001)
    sum(losses.values()).backward()
    optimizer.step()
    selections = [selected_predictions(outputs, b, fake.evo_cfg)[:2] for b in range(len(det['prediction'][-1]))]
    return dict(losses=losses, raw=raw.detach(), gradients=raw_gradients, pairs=pairs,
                selected=selections, updated=parameters, optimizer=optimizer.state_dict())


def exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    golden = torch.load(args.golden)
    result = {}
    for weight in (0., 1.):
        for label, flags in [('disabled', {'enable_dynamics_loss': False}),
                             ('zero', {'enable_dynamics_loss': True, 'lambda_dyn': 0.})]:
            a, b = [run(golden, original, flags, weight) for original in (True, False)]
            exact(a, b)
            result['%s_evo_weight_%s' % (label, weight)] = dict(
                bitwise_equal=True, raw_evo=float(a['raw']),
                matched_agents=sum(len(p) for p, _ in a['pairs'])//2,
                compared='loss_dict, raw_evo, topk, modes, matched_positions, pseudo_targets, output_gradients, AdamW parameters/state')
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
