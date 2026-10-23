"""Two-rank real-data forward/backward smoke, no checkpoints are written."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from capture_baseline import seed, make_batch, norms
import torch
import torch.distributed as dist
import mmcv
from mmcv.runner import load_checkpoint, wrap_fp16_model, HOOKS
from mmcv.utils import build_from_cfg, import_modules_from_strings
from mmdet.models import build_detector
from mmcv.parallel import MMDistributedDataParallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/sparsedrive_small_stage2_ctrv_v1.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    seed()
    model = build_detector(cfg.model).cuda().train()
    load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False)
    if args.fp16:
        wrap_fp16_model(model)
    dataset, batch = make_batch(cfg, 10 + rank, scatter_to=None)
    model = MMDistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False,
                                     find_unused_parameters=True)
    from mmcv.runner import IterBasedRunner, build_optimizer
    # Use the real MMCV Runner optimizer hook, including configured mixed precision.
    optimizer = build_optimizer(model, cfg.optimizer)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = IterBasedRunner(model=model, optimizer=optimizer, work_dir=str(output),
                             logger=__import__('logging').getLogger('ctrv_ddp'), max_iters=1, meta={})
    if args.fp16:
        from mmcv.runner import Fp16OptimizerHook
        hook = Fp16OptimizerHook(loss_scale=32., grad_clip=dict(max_norm=25, norm_type=2), distributed=True)
    else:
        from mmcv.runner import OptimizerHook
        hook = OptimizerHook(grad_clip=dict(max_norm=25, norm_type=2))
    runner.register_hook(hook)
    diagnostics_hook = None
    for hook_cfg in cfg.get('custom_hooks', []):
        settings = dict(hook_cfg)
        priority = settings.pop('priority', 'NORMAL')
        custom = build_from_cfg(settings, HOOKS)
        runner.register_hook(custom, priority=priority)
        if settings['type'] == 'DynamicsDiagnosticsHook':
            diagnostics_hook = custom
    runner.run([[batch]], [('train', 1)])
    summary = dict(rank=rank, fp16=args.fp16, reduced_dynamics=getattr(diagnostics_hook, 'latest_metrics', None), log_vars=runner.outputs['log_vars'],
                   dynamics=model.module.dynamics_diagnostics, gradients=norms(model.module))
    (output / ('rank%d.json' % rank)).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
