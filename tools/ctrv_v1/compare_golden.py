"""Compare stochastic CUDA replays and save errors; AST/toy checks are exact."""
import argparse
import json
from pathlib import Path
import torch


def flat_tensors(value, path=''):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key, v in value.items():
            yield from flat_tensors(v, path+'.'+key)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from flat_tensors(v, path+'.'+str(i))


def metrics(a, b):
    ta, tb = dict(flat_tensors(a)), dict(flat_tensors(b))
    assert ta.keys() == tb.keys(), (ta.keys()-tb.keys(), tb.keys()-ta.keys())
    output = {}
    for group in ('losses', 'det', 'motion', 'diagnostics', 'grads', 'matched'):
        errors, denom, maximum, count, exact = 0., 0., 0., 0, True
        for name in ta:
            if not name.startswith('.'+group+'.'):
                continue
            x, y = ta[name].double(), tb[name].double()
            assert x.shape == y.shape, name
            d = x-y
            errors += float(d.square().sum())
            denom += float(x.square().sum())
            maximum = max(maximum, float(d.abs().max()) if d.numel() else 0.)
            count += d.numel()
            exact &= torch.equal(x, y)
        output[group] = dict(max_abs=maximum, relative_l2=(errors/max(denom, 1e-30))**.5,
                             elements=count, bitwise_equal=exact)
    output['matched_agent_count_equal'] = sum(len(p) for p, _ in a.get('matched', [])) == sum(len(p) for p, _ in b.get('matched', []))
    output['raw_evo_abs'] = abs(float(a['raw_evo'])-float(b['raw_evo']))
    output['topk_equal'] = torch.equal(a['diagnostics']['topk_indices_det'], b['diagnostics']['topk_indices_det'])
    output['mode_equal'] = torch.equal(a['motion']['classification'][-1].argmax(-1), b['motion']['classification'][-1].argmax(-1))
    # Stated acceptance bounds; report control errors as well, never call this bitwise equivalence.
    output['within_tolerance'] = (output['raw_evo_abs'] < 5e-4
        and output['losses']['max_abs'] < 1e-3 and output['det']['relative_l2'] < 1e-3
        and output['motion']['relative_l2'] < 1e-3 and output['grads']['relative_l2'] < .02
        and output['topk_equal'] and output['mode_equal'])
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--disabled', required=True)
    p.add_argument('--zero', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    reference = torch.load(args.reference)
    result = {label: metrics(reference, torch.load(getattr(args, label)))
              for label in ('control', 'disabled', 'zero')}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert all(r['within_tolerance'] for r in result.values())

if __name__ == '__main__':
    main()
