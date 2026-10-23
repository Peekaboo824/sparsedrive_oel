"""Explicitly activated OEL baseline, NOT the old script's zero-evo run.

The old configuration remains untouched. 1.0 is the repository's implicit Evo
weight in configs without evo_cfg; it is an experiment setting, not a tuned value.
Launch with 3 ranks as in the inherited schedule, or override the SAME schedule
and per-GPU batch settings for both E0/E1.
"""
_base_ = ['./sparsedrive_small_stage2.py']
model = dict(evo_cfg=dict(loss_weight=1.0, enable_dynamics_loss=False))
