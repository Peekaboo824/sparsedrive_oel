"""Real GPU mini-batch validation. Writes only to the explicit output directory."""
import argparse
import ast
import importlib
import json
import time
import types
from unittest.mock import patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from capture_baseline import seed, make_batch, norms, cpu
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


def bind_original(model):
    tree = ast.parse(Path('tests/ctrv_v1/baseline/sparsedrive.py.txt').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseDrive')
    namespace = vars(importlib.import_module('projects.mmdet3d_plugin.models.sparsedrive')).copy()
    for name in ('forward_train', 'get_evo_loss'):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'baseline', 'exec'), namespace)
        setattr(model, name, types.MethodType(namespace[name], model))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='projects/configs/sparsedrive_small_stage2.py')
    p.add_argument('--mode', choices=['original', 'disabled', 'zero', 'enabled'], required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--index', type=int, default=10)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--steps', type=int, default=0, help='Additional real optimizer steps on sequential samples')
    args = p.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = mmcv.Config.fromfile(args.config)
    if args.mode in ('disabled', 'zero'):
        cfg.model.evo_cfg.update(enable_dynamics_loss=args.mode == 'zero', lambda_dyn=0.)
    seed()
    model = build_detector(cfg.model).cuda().train()
    load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False)
    if args.fp16:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
    if args.mode == 'original':
        bind_original(model)
    dataset, batch = make_batch(cfg, args.index)
    recorded = {}
    def capture_head(m, args, outs):
        recorded['outs' if m is model.head else 'next_outs'] = outs
    hook = model.head.register_forward_hook(capture_head)
    original_evo = model.get_evo_loss
    def evo(*args, **kwargs):
        original_l1 = torch.nn.functional.l1_loss
        matched = []
        def l1(predicted, target, *a, **kw):
            matched.append((predicted.detach().cpu(), target.detach().cpu()))
            return original_l1(predicted, target, *a, **kw)
        with patch('torch.nn.functional.l1_loss', side_effect=l1):
            result = original_evo(*args, **kwargs)
        recorded['matched'] = matched
        recorded['evo'] = result['loss_evo']
        return result
    model.get_evo_loss = evo
    seed()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=args.fp16):
        losses = model.forward_train(**batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    outs = recorded['outs']
    summary = dict(mode=args.mode, fp16=args.fp16, matched_agent_count=sum(len(p) for p, _ in recorded['matched']), forward_seconds=elapsed,
                   peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                   losses={k: float(v.detach()) for k, v in losses.items()},
                   raw_evo=float(recorded['evo'].detach()),
                   dynamics=getattr(model, 'dynamics_diagnostics', None), gradients={})
    gradients = {}
    for name, loss in [('evo', recorded['evo'])] + ([('dyn', losses['loss_dyn']/cfg.model.evo_cfg.lambda_dyn)] if 'loss_dyn' in losses else []):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        summary['gradients'][name] = norms(model)
        if name == 'evo':
            gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        summary['gradients'][name]['motion_regression_parameters'] = {
            n: float(p.grad.float().norm()) for n, p in model.named_parameters()
            if 'motion_reg' in n and p.grad is not None}
    result = dict(matched=recorded['matched'],
                  det_output=cpu({k: outs[0][k] for k in ('prediction', 'classification')}),
                  plan_output=cpu({'classification': outs[3]['classification']}),
                  pseudo_detection=cpu({k: recorded['next_outs'][0][k] for k in ('prediction', 'classification')}),
                  input_data=cpu({k: v for k, v in batch.items() if k in ('img_metas', 'next_img_metas', 'gt_ego_fut_cmd')}),
                  losses=cpu(losses), raw_evo=cpu(recorded['evo']),
                  grad_norms=summary['gradients']['evo'],
                  det=cpu(outs[0]['prediction']), motion=cpu(outs[2]),
                  diagnostics=cpu({k: outs[4][k] for k in ('attn_weights_gnn', 'topk_indices_det')}),
                  grads=gradients)
    torch.save(result, output/'golden.pt')
    hook.remove()
    # Capture online updates with the actual final weighted loss dictionary.
    if args.steps:
        from mmcv.parallel import collate, scatter
        from mmcv.runner import build_optimizer
        optimizer = build_optimizer(model, cfg.optimizer)
        optimizer.zero_grad(set_to_none=True)
        update_start = time.perf_counter()
        sum(losses.values()).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 25))
        optimizer.step()
        torch.cuda.synchronize()
        summary['online_steps'] = [dict(index=args.index,
            seconds=time.perf_counter()-update_start, loss=float(sum(losses.values()).detach()),
            grad_norm=grad_norm, dynamics=getattr(model, 'dynamics_diagnostics', None))]
        del result, gradients, outs, losses, batch
        recorded.clear()
        for step in range(1, args.steps):
            seed()
            batch = scatter(collate([dataset[args.index+step]], samples_per_gpu=1), [0])[0]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                losses = model.forward_train(**batch)
            sum(losses.values()).backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 25))
            optimizer.step()
            torch.cuda.synchronize()
            summary['online_steps'].append(dict(index=args.index+step, seconds=time.perf_counter()-start,
                loss=float(sum(losses.values()).detach()), grad_norm=grad_norm,
                dynamics=getattr(model, 'dynamics_diagnostics', None)))
            recorded.clear()
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
