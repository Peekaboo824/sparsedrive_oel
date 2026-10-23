"""Sensor-only causal history input, deliberately independent of get_ann_info()."""
import copy
import numpy as np
import pyquaternion


def scene_indices(data_infos):
    scenes = {}
    for i, info in enumerate(data_infos):
        scenes.setdefault(info['scene_token'], []).append(i)
    return {scene: sorted(indices, key=lambda i: data_infos[i]['timestamp'])
            for scene, indices in scenes.items()}


def history_indices(data_infos, scenes, index, length):
    current = data_infos[index]
    indices = scenes[current['scene_token']]
    # Include t, exclude all future observations. Works with interleaved scenes.
    return [i for i in indices if data_infos[i]['timestamp'] <= current['timestamp']][-length:]


def sensor_frame_info(info):
    def pose(rotation, translation):
        t = np.eye(4)
        t[:3, :3] = pyquaternion.Quaternion(rotation).rotation_matrix
        t[:3, 3] = translation
        return t
    result = dict(timestamp=info['timestamp']/1e6,
                  lidar2global=pose(info['ego2global_rotation'], info['ego2global_translation'])
                  @ pose(info['lidar2ego_rotation'], info['lidar2ego_translation']))
    paths, projections = [], []
    for camera in info['cams'].values():
        paths.append(camera['data_path'])
        inv_rotation = np.linalg.inv(camera['sensor2lidar_rotation'])
        translation = camera['sensor2lidar_translation'] @ inv_rotation.T
        lidar2cam = np.eye(4)
        lidar2cam[:3, :3] = inv_rotation.T
        lidar2cam[3, :3] = -translation
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = copy.deepcopy(camera['cam_intrinsic'])
        projections.append(intrinsic @ lidar2cam.T)
    result.update(img_filename=paths, lidar2img=projections)
    return result
