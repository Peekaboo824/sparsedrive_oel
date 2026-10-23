"""Run with torch.distributed.run: actual two-rank reduction with a zero-agent rank."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from projects.mmdet3d_plugin.models.dynamics.diagnostics import reduce_dynamics_diagnostics

p = argparse.ArgumentParser()
p.add_argument('--output-dir', required=True)
args = p.parse_args()
dist.init_process_group('gloo')
rank = dist.get_rank()
stats = (dict(batch_size=2, entropy_triggered_samples=0, active_samples=0, matched_dyn_count=0)
         if rank == 0 else dict(batch_size=3, entropy_triggered_samples=2, active_samples=1,
                               matched_dyn_count=3, raw_loss_dyn=4., weighted_loss_dyn=.4))
rng = torch.get_rng_state().clone()
metrics = reduce_dynamics_diagnostics(stats, torch.device('cpu'))
assert metrics['entropy_trigger_rate'] == .4
assert metrics['matched_dyn_count'] == 3
assert metrics['dyn_active_sample_ratio'] == .2
assert metrics['raw_loss_dyn'] == 4.
assert metrics['raw_loss_dyn_rank_mean'] == 2.
assert metrics['raw_loss_dyn_batch_mean'] == 2.4
torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
(output/('rank%d.json' % rank)).write_text(json.dumps(metrics, indent=2))
print('PASS: rank', rank, 'zero-agent/unequal-batch global reduction', metrics, flush=True)
dist.destroy_process_group()
