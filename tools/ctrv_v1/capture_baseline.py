"""Capture the unmodified real model before V1 edits. No dataset files are edited."""
import argparse
import os
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import projects.mmdet3d_plugin
from mmdet.models import build_detector
from projects.mmdet3d_plugin.datasets import NuScenes3DDataset


def seed():
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_batch(cfg, index=10, scatter_to=0):
    dataset = NuScenes3DDataset(**{k: v for k, v in cfg.data.train.items() if k != 'type'})
    # Original annotations contain an obsolete absolute LiDAR path. Resolve only
    # this input path in memory; annotations/checkpoints/configs stay untouched.
    for info in dataset.data_infos:
        path = info['lidar_path']
        if not os.path.exists(path) and '/samples/' in path:
            info['lidar_path'] = 'data/nuscenes/samples/' + path.split('/samples/', 1)[1]
    seed()
    batch = collate([dataset[index]], samples_per_gpu=1)
    if scatter_to is not None:
        batch = scatter(batch, [scatter_to])[0]
    return dataset, batch


def norms(model):
    groups = {'backbone': 'img_backbone.', 'neck': 'img_neck.',
              'detection': 'head.det_head.', 'motion': 'head.motion_plan_head.',
              'prediction': 'head.motion_plan_head.layers.'}
    return {g: sum(float(p.grad.float().square().sum()) for n, p in model.named_parameters()
                   if n.startswith(prefix) and p.grad is not None) ** .5
            for g, prefix in groups.items()}


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='projects/configs/sparsedrive_small_stage2.py')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    seed()
    model = build_detector(cfg.model).cuda().train()
    load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False)
    _, batch = make_batch(cfg)
    seed()
    features, _ = model.extract_feat(batch['img'], True, batch)
    outs = model.head(features, batch)
    losses = model.head.loss(outs, batch)
    evo = model.get_evo_loss(outs, batch, features)['loss_evo']
    losses['loss_evo'] = evo * cfg.model.evo_cfg.loss_weight
    evo.backward()
    result = dict(losses=cpu(losses), raw_evo=cpu(evo), grad_norms=norms(model),
                  det=cpu(outs[0]['prediction']), motion=cpu(outs[2]),
                  diagnostics=cpu({k: outs[4][k] for k in ('attn_weights_gnn', 'topk_indices_det')}),
                  grads={n: p.grad.cpu() for n, p in model.named_parameters() if p.grad is not None})
    torch.save(result, args.output)
    print({k: float(v) for k, v in losses.items()})
    print('raw_evo', float(evo), 'grad_norms', norms(model), 'saved', args.output)

if __name__ == '__main__':
    main()
