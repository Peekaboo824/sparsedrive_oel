"""Opt-in local-batch gradient conflict analysis. No DDP wrapper or step hooks."""
import torch


GROUPS = dict(backbone='img_backbone.', neck='img_neck.',
              detection='head.det_head.', motion='head.motion_plan_head.')


def _summary(dot, norm_e_squared, norm_d_squared, epsilon):
    norm_e, norm_d = norm_e_squared**.5, norm_d_squared**.5
    valid = norm_e > 0. and norm_d > 0.
    return dict(cosine=dot/(norm_e*norm_d+epsilon) if valid else None,
                evo_norm=norm_e, dyn_norm=norm_d,
                status='ok' if valid else 'zero_or_unused_gradient')


def gradient_conflict(model, raw_evo, raw_dyn, evo_weight=1., dyn_weight=.1, epsilon=1e-12):
    """Same trainable parameter vector for each loss; unused entries are zeros.

    autograd.grad never populates .grad, never runs a step, and retains the graph
    for ordinary backward. This is a standalone analysis of unwrapped local model
    outputs, not a training-time DDP collective or the globally averaged gradient.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad
             and any(n.startswith(prefix) for prefix in GROUPS.values())]
    parameters = [p for _, p in named]
    def gradients(loss):
        if not parameters or loss is None or not loss.requires_grad:
            return [None]*len(parameters)
        return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    grads_e, grads_d = gradients(raw_evo), gradients(raw_dyn)
    result = {}
    for group, prefix in GROUPS.items():
        dot = norm_e = norm_d = 0.
        count = used_e = used_d = joint = 0
        for (name, _), ge, gd in zip(named, grads_e, grads_d):
            if not name.startswith(prefix):
                continue
            count += 1
            used_e += ge is not None
            used_d += gd is not None
            joint += ge is not None and gd is not None
            if ge is not None:
                norm_e += float(ge.detach().double().square().sum())
            if gd is not None:
                norm_d += float(gd.detach().double().square().sum())
            if ge is not None and gd is not None:
                dot += float((ge.detach().double()*gd.detach().double()).sum())
        result[group] = dict(raw=_summary(dot, norm_e, norm_d, epsilon),
            weighted=_summary(dot*evo_weight*dyn_weight, norm_e*evo_weight**2,
                              norm_d*dyn_weight**2, epsilon),
            parameter_tensors=count, evo_used=used_e, dyn_used=used_d, jointly_used=joint)
    return dict(scope='local batch, unwrapped model, unused parameters treated as zero',
                evo_weight=float(evo_weight), dyn_weight=float(dyn_weight), groups=result)
