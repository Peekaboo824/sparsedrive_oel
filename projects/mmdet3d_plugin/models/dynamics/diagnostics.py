"""Detached per-step diagnostics. Never add metrics to the training loss dict."""
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


_FIELDS = ('batch_size', 'entropy_triggered_samples', 'active_samples',
           'topk_vehicle_count', 'reliable_ekf_count', 'matched_dyn_count',
           'association_attempted_count', 'association_rejection_count',
           'prior_rejection_count', 'unavailable_dyn_target_count', 'prior_variance_sum',
           'track_observation_sum', 'track_age_sum')


def diagnostics_totals(stats):
    """Additive sufficient statistics; losses are local valid-agent means."""
    matched = stats.get('matched_dyn_count', 0)
    batch = stats.get('batch_size', 0)
    raw, weighted = stats.get('raw_loss_dyn', 0.), stats.get('weighted_loss_dyn', 0.)
    return [float(stats.get(k, 0)) for k in _FIELDS] + [
        raw*matched, weighted*matched, raw*batch, weighted*batch, raw, weighted, 1.]


def metrics_from_totals(values):
    """Ratios of global sums, not averages of per-rank ratios.

    Counts are global counts per step. Agent metrics use matched-agent counts;
    sample ratios include all images, including low entropy/invalid history.
    Rank-mean loss is the actual DDP mean of local-agent-mean objectives.
    """
    s = dict(zip(_FIELDS, values[:len(_FIELDS)]))
    raw_agent, weighted_agent, raw_batch, weighted_batch, raw_rank, weighted_rank, ranks = values[len(_FIELDS):]
    batch, agents = s['batch_size'], s['matched_dyn_count']
    divide = lambda numerator, denominator: numerator/denominator if denominator else 0.
    result = dict(
        entropy_trigger_rate=divide(s['entropy_triggered_samples'], batch),
        topk_vehicle_count=s['topk_vehicle_count'],
        reliable_ekf_count=s['reliable_ekf_count'],
        matched_dyn_count=agents,
        dyn_active_sample_ratio=divide(s['active_samples'], batch),
        raw_loss_dyn=divide(raw_agent, agents),
        weighted_loss_dyn=divide(weighted_agent, agents),
        raw_loss_dyn_batch_mean=divide(raw_batch, batch),
        weighted_loss_dyn_batch_mean=divide(weighted_batch, batch),
        raw_loss_dyn_rank_mean=divide(raw_rank, ranks),
        weighted_loss_dyn_rank_mean=divide(weighted_rank, ranks),
        ekf_prior_position_variance=divide(s['prior_variance_sum'], agents),
        track_observation_count=divide(s['track_observation_sum'], agents),
        track_age=divide(s['track_age_sum'], agents),
        association_attempted_count=s['association_attempted_count'],
        association_rejection_count=s['association_rejection_count'],
        prior_rejection_count=s['prior_rejection_count'],
        unavailable_dyn_target_count=s['unavailable_dyn_target_count'],
        dyn_batch_sample_count=batch,
    )
    for k in ('topk_vehicle_count', 'reliable_ekf_count', 'matched_dyn_count'):
        result[k+'_per_sample'] = divide(result[k], batch)
    return result


def reduce_dynamics_diagnostics(stats, device):
    # A fresh detached vector: all_reduce cannot touch model tensors or gradients.
    vector = torch.tensor(diagnostics_totals(stats), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    return metrics_from_totals(vector.cpu().tolist())


@HOOKS.register_module()
class DynamicsDiagnosticsHook(Hook):
    """Runs after optimizer, before MMCV loggers; no extra backward/optimizer step.

    All ranks must use the same branch enable/weight settings as normal DDP.
    Logger windows average per-step metrics; agent averages here are per step.
    """
    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        cfg = getattr(model, 'evo_cfg', {})
        if not cfg.get('enable_dynamics_loss', False) or cfg.get('lambda_dyn', .1) == 0.:
            return
        stats = getattr(model, 'dynamics_diagnostics', None)
        if stats is None:
            return
        parameter = next(model.parameters())
        device = parameter.device
        if dist.is_initialized() and dist.get_backend() == 'gloo':
            device = torch.device('cpu')
        metrics = reduce_dynamics_diagnostics(stats, device)
        runner.log_buffer.update(metrics, 1)
        self.latest_metrics = metrics
