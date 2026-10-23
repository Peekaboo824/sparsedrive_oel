"""Copy legacy annotations to a NEW runtime file with resolved sensor paths.

Does not change labels/splits or overwrite source files. Optional convenience for
this checkout's obsolete absolute pts_filename paths; not part of EKF supervision.
"""
import argparse
from pathlib import Path
import mmcv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='data/nuscenes')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    data = mmcv.load(args.input)
    resolved = 0
    for info in data['infos']:
        source = info['lidar_path']
        if not Path(source).exists():
            if '/samples/' not in source:
                raise ValueError('Unknown LiDAR path: ' + source)
            target = Path(args.data_root)/'samples'/source.split('/samples/', 1)[1]
            if not target.exists():
                raise FileNotFoundError(str(target))
            info['lidar_path'] = str(target)
            resolved += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump(data, str(output))
    print('Wrote NEW runtime annotations:', output, 'resolved', resolved, 'paths; labels unchanged')

if __name__ == '__main__':
    main()
