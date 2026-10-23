"""Independent dynamics supervision; no calls to or changes in get_evo_loss()."""
import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent_track_bank import AgentTrackBank, associate


def compute_dynamics_loss(predicted_positions, ekf_prior_positions, valid_mask):
    if predicted_positions.shape != ekf_prior_positions.shape:
        raise ValueError('Neural predictions and EKF priors must have identical shapes')
    if valid_mask.shape != predicted_positions.shape[:-1]:
        raise ValueError('Invalid dynamics mask shape')
    if valid_mask.any():
        return F.l1_loss(predicted_positions[valid_mask].float(),
                         ekf_prior_positions[valid_mask].detach().float())
    return predicted_positions.float().sum() * 0.


def to_world(positions, transform):
    """LiDAR/augmented LiDAR -> global; row-vector multiplication by T_global.T.

    2D motion points are lifted with z=0, matching the repository's BEV convention.
    Full detected xyz positions use their actual height.
    """
    with torch.cuda.amp.autocast(enabled=False):
        t = torch.as_tensor(transform, device=positions.device, dtype=torch.float32)
        p = positions.float()
        if p.shape[-1] == 2:
            p = torch.cat([p, torch.zeros_like(p[..., :1])], -1)
        homogeneous = torch.cat([p, torch.ones_like(p[..., :1])], -1)
        return (homogeneous @ t.T)[..., :2]


@torch.no_grad()
def planning_entropy_trigger(outputs, data, b, cfg):
    """Read-only copy of the original Evo gate, including missing-command bypass.

    Keep dtype/autocast, reshape, argmax, softmax, epsilon and <= boundary exactly
    as get_evo_loss; do not cast logits or substitute a different command source.
    """
    plan_cls_scores_b = outputs[3]['classification'][-1][b]
    reshaped_scores = plan_cls_scores_b.reshape(3, 6)
    if 'gt_ego_fut_cmd' in data:
        gt_cmd_tensor = data['gt_ego_fut_cmd'][b]
        cmd_idx = gt_cmd_tensor.argmax(dim=-1).item()
        scores_for_entropy = reshaped_scores[cmd_idx, :]
        plan_probs = torch.softmax(scores_for_entropy, dim=-1)
        entropy = -(plan_probs * torch.log(plan_probs + 1e-9)).sum()
        if entropy <= cfg.get('ENTROPY_THRESHOLD', .2):
            return False
    return True


def selected_predictions(outputs, b, cfg):
    diagnostic = outputs[4]
    attention = diagnostic['attn_weights_gnn'][b][-1, :-1]
    k = min(cfg.get('TOP_K_CAUSAL_AGENTS', 10), len(attention))
    indices = torch.gather(diagnostic['topk_indices_det'][b], 0, torch.topk(attention, k=k)[1])
    offsets = outputs[2]['prediction'][-1][b][indices]
    modes = outputs[2]['classification'][-1][b][indices].argmax(-1)
    gather = modes.reshape(k, 1, 1, 1).expand(k, 1, offsets.shape[-2], 2)
    best = torch.gather(offsets, 1, gather).squeeze(1)
    prediction = best.cumsum(dim=1) + outputs[0]['prediction'][-1][b][indices, :2].unsqueeze(1)
    return indices, modes, prediction[:, 0]


def observations(det, meta, cfg):
    scores, labels = det['classification'][-1][0].detach().float().sigmoid().max(-1)
    boxes = det['prediction'][-1][0].detach().float()
    vehicles = torch.as_tensor(cfg.get('dynamics_vehicle_class_ids', [0, 1, 2, 3, 4]), device=boxes.device)
    valid = (labels[:, None] == vehicles).any(-1)
    valid &= scores > cfg.get('ekf_detection_threshold', .4)
    # Use the same z=0 BEV lifting for observations and future predictions.
    world = to_world(boxes[:, :2], meta['T_global'])
    valid &= torch.isfinite(world).all(-1)
    return world[valid].cpu().double().numpy(), labels[valid].cpu().numpy()


class HistoricalPerception:
    """Fresh copy of CURRENT online weights, eval/no-grad, reset per sample.

    No fixed teacher/cache. Only the detection head is evaluated; no GT fields,
    no planning commands, and no mutation of live temporal memories or BN stats.
    """
    def __init__(self, model):
        self.encoder = nn.Module()
        self.encoder.img_backbone = copy.deepcopy(model.img_backbone)
        self.encoder.img_neck = copy.deepcopy(model.img_neck) if model.img_neck is not None else None
        self.encoder.eval()
        self.encoder.use_grid_mask = False
        self.encoder.depth_branch = None
        self.encoder.use_deformable_func = model.use_deformable_func
        self.encoder.fp16_enabled = getattr(model, 'fp16_enabled', False)
        self.extract = type(model).extract_feat
        self.det = copy.deepcopy(model.head.det_head)
        self.det.eval()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, frames, cfg):
        self.det.instance_bank.reset()
        if hasattr(self.det.sampler, 'dn_metas'):
            self.det.sampler.dn_metas = None
        result = []
        for frame in frames:
            meta = frame['img_metas']
            data = {key: frame[key].unsqueeze(0).to(self.device)
                    for key in ('timestamp', 'projection_mat', 'image_wh')}
            data['img_metas'] = [meta]
            image = frame['img'].unsqueeze(0).to(self.device)
            features = self.extract(self.encoder, image, metas=data)
            output = self.det(features, data)
            xy, labels = observations(output, meta, cfg)
            result.append((float(meta['timestamp']), xy, labels))
        return result


def dynamics_loss(model, outputs, data):
    started = time.perf_counter()
    cfg = model.evo_cfg
    if cfg.get('dynamics_model', 'CTRV_EKF') != 'CTRV_EKF':
        raise ValueError('V1 supports only CTRV_EKF')
    ref = outputs[2]['prediction'][-1]
    zero = ref.float().sum() * 0. + outputs[0]['prediction'][-1].float().sum()*0.
    contexts = data.get('dynamics_context')
    predictions, priors = [], []
    stats = dict(batch_size=ref.shape[0], entropy_triggered_samples=0, active_samples=0,
                 topk_vehicle_count=0, reliable_ekf_count=0, matched_dyn_count=0,
                 prior_variance_sum=0., track_observation_sum=0., track_age_sum=0.,
                 association_attempted_count=0, association_rejection_count=0,
                 prior_rejection_count=0, unavailable_dyn_target_count=0,
                 valid_agents=0, history_frames=0, matched_observations=0, observations=0,
                 teacher_copy_seconds=0., history_perception_seconds=0., ekf_seconds=0.)
    # Preserve original stochastic training stream, even if a backend draws RNG.
    py_state, np_state = random.getstate(), np.random.get_state()
    device = ref.device
    devices = [device.index] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            teacher = None
            for b in range(ref.shape[0]):
                if not planning_entropy_trigger(outputs, data, b, cfg):
                    continue  # no teacher copy, history inference or EKF for low entropy
                stats['entropy_triggered_samples'] += 1
                if contexts is None:
                    raise ValueError('Dynamics enabled but dataset has no dynamics_context; configure dynamics_history_length/pipeline')
                context = contexts[b]
                indices, _, nn_xy = selected_predictions(outputs, b, cfg)
                labels = outputs[0]['classification'][-1][b].detach().argmax(-1)[indices]
                vehicles = labels.new_tensor(cfg.get('dynamics_vehicle_class_ids', [0, 1, 2, 3, 4]))
                vehicle_count = int((labels[:, None] == vehicles).any(-1).sum().item())
                stats['topk_vehicle_count'] += vehicle_count
                # Also count missing-history/prior failures separately from association.
                stats['unavailable_dyn_target_count'] += vehicle_count
                frames = context['history']
                if not context['has_valid_next_frame'] or len(frames) < max(3, cfg.get('ekf_min_observations', 3)):
                    continue
                if len(frames) > cfg.get('ekf_history_length', 5):
                    frames = frames[-cfg.get('ekf_history_length', 5):]
                dt = context['next_timestamp'] - context['current_timestamp']
                if abs(dt - cfg.get('ekf_motion_step_seconds', .5)) > cfg.get('ekf_time_tolerance', .05):
                    continue  # a dropped/subsampled next frame is not the first motion point
                times = [float(f['img_metas']['timestamp']) for f in frames]
                if any(t1 >= t2 for t1, t2 in zip(times, times[1:])) or times[-1] != context['current_timestamp']:
                    raise ValueError('Invalid causal history timestamps')
                if any(f['scene_token'] != context['scene_token'] for f in frames):
                    raise ValueError('History crossed a scene boundary')
                if teacher is None:
                    before = time.perf_counter()
                    teacher = HistoricalPerception(model)
                    stats['teacher_copy_seconds'] += time.perf_counter() - before
                before = time.perf_counter()
                sequence = teacher(frames, cfg)
                stats['history_perception_seconds'] += time.perf_counter() - before
                before = time.perf_counter()
                bank = AgentTrackBank(cfg)
                for timestamp, xy, history_labels in sequence:
                    bank.step(timestamp, xy, history_labels)
                    stats['history_frames'] += 1
                stats['matched_observations'] += bank.matched_count
                stats['observations'] += bank.observation_count
                tracks = bank.reliable_tracks()
                stats['reliable_ekf_count'] += len(tracks)
                if not tracks:
                    stats['ekf_seconds'] += time.perf_counter() - before
                    continue
                boxes = outputs[0]['prediction'][-1][b]
                world = to_world(boxes[indices, :2].detach(), data['img_metas'][b]['T_global'])
                pairs = associate(world.cpu().double().numpy(), labels.cpu().numpy(), tracks,
                                  cfg.get('ekf_association_gate', 9.21),
                                  cfg.get('ekf_max_association_distance', 6.))
                stats['association_attempted_count'] += vehicle_count
                stats['association_rejection_count'] += vehicle_count-len(pairs)
                world_prediction = to_world(nn_xy, data['img_metas'][b]['T_global'])
                sample_matches = 0
                for i, j in pairs:
                    prior = bank.predict_prior(tracks[j], context['next_timestamp'], return_covariance=True)
                    if prior is not None:
                        position, prior_covariance = prior
                        predictions.append(world_prediction[i])
                        priors.append(position)
                        sample_matches += 1
                        stats['prior_variance_sum'] += float(np.trace(prior_covariance)/2)
                        stats['track_observation_sum'] += len(tracks[j]['observations'])
                        stats['track_age_sum'] += bank.time-tracks[j]['last_seen']
                stats['active_samples'] += int(sample_matches > 0)
                stats['prior_rejection_count'] += len(pairs)-sample_matches
                stats['unavailable_dyn_target_count'] -= sample_matches
                stats['ekf_seconds'] += time.perf_counter() - before
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    stats['valid_agents'] = stats['matched_dyn_count'] = len(predictions)
    if predictions:
        predicted = torch.stack(predictions)
        target = torch.as_tensor(np.stack(priors), device=device, dtype=torch.float32)
        loss = compute_dynamics_loss(predicted, target, torch.ones(len(priors), device=device, dtype=torch.bool))
    else:
        loss = zero
    stats['raw_loss_dyn'] = float(loss.detach())
    stats['weighted_loss_dyn'] = stats['raw_loss_dyn'] * cfg.get('lambda_dyn', .1)
    stats['total_seconds'] = time.perf_counter() - started
    model.dynamics_diagnostics = stats  # detached logging only; no tracks survive
    return loss
