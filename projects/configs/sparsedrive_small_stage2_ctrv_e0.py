"""Explicitly activated OEL baseline, NOT the old script's zero-evo run.

The old configuration remains untouched. 1.0 is the repository's implicit Evo
weight in configs without evo_cfg; it is an experiment setting, not a tuned value.
Launch with 3 ranks as in the inherited schedule, or override the SAME schedule
and per-GPU batch settings for both E0/E1.
"""
_base_ = ['./sparsedrive_small_stage2.py']
model = dict(evo_cfg=dict(loss_weight=1.0, enable_dynamics_loss=False))

# Remove only the audited early-stop hook; retain unrelated parent hooks.
from mmcv import Config as _Config
custom_hooks = [hook for hook in _Config.fromfile(
    '{{fileDirname}}/sparsedrive_small_stage2.py').get('custom_hooks', [])
    if hook['type'] != 'WindowLossStopHook']
del _Config
custom_hooks.append(dict(type='DynamicsDiagnosticsHook', priority='NORMAL'))
custom_imports = dict(
    imports=['projects.mmdet3d_plugin.models.dynamics.diagnostics'],
    allow_failed_imports=False)
# Both comparison runs must pass --seed 3407 --deterministic to tools/train.py;
# that existing entrypoint overwrites cfg.seed with its CLI seed (default 0).
seed = 3407
