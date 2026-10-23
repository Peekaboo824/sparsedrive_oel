"""Real sensor pipelines with two workers, different scenes and scene boundaries."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from capture_baseline import make_batch
from functools import partial
import mmcv
from mmcv.parallel import collate
from torch.utils.data import DataLoader, Subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile('projects/configs/sparsedrive_small_stage2_ctrv_v1.py')
    dataset, _ = make_batch(cfg, scatter_to=None)
    scene = dataset.data_infos[10]['scene_token']
    next_scene = next(i for i in range(11, len(dataset)) if dataset.data_infos[i]['scene_token'] != scene)
    indices = [10, next_scene+4, next_scene-1, next_scene]
    loader = DataLoader(Subset(dataset, indices), batch_size=2, num_workers=2,
                        collate_fn=partial(collate, samples_per_gpu=2))
    result = []
    for batch in loader:
        for context in batch['dynamics_context'].data[0]:
            assert all(f['scene_token'] == context['scene_token'] for f in context['history'])
            times = [f['img_metas']['timestamp'] for f in context['history']]
            assert times == sorted(set(times)) and max(times) == context['current_timestamp']
            assert all(t <= context['current_timestamp'] for t in times)
            assert all(not any(k.startswith('gt_') or 'instance_id' == k for k in f) for f in context['history'])
            result.append(dict(scene=context['scene_token'], times=times,
                               has_valid_next_frame=context['has_valid_next_frame']))
    assert result[0]['scene'] != result[1]['scene']
    assert not result[2]['has_valid_next_frame']
    assert len(result[3]['times']) == 1
    Path(args.output).write_text(json.dumps(dict(workers=2, batch_size=2, samples=result), indent=2))
    print('PASS: real two-worker/two-sample collate, scene boundaries, causal history and sensor-only fields')

if __name__ == '__main__':
    main()
