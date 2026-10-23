import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER

import pdb
import sys


class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace():
    ForkedPdb().set_trace(sys._getframe().f_back)


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        assert not self.shuffle
        if "data_infos" in dir(self.dataset):
            timestamps = [
                x["timestamp"] / 1e6 for x in self.dataset.data_infos
            ]
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.data_infos
            ]
        else:
            timestamps = [
                x["timestamp"] / 1e6
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)

        sequence_splits = []
        for i in range(len(timestamps)):
            if i == 0 or (
                abs(timestamps[i] - timestamps[i - 1]) > 4
                or vehicle_idx[i] != vehicle_idx[i - 1]
            ):
                sequence_splits.append([i])
            else:
                sequence_splits[-1].append(i)

        indices = []
        perfix_sum = 0
        split_length = len(self.dataset) // self.num_replicas
        for i in range(len(sequence_splits)):
            if perfix_sum >= (self.rank + 1) * split_length:
                break
            elif perfix_sum >= self.rank * split_length:
                indices.extend(sequence_splits[i])
            perfix_sum += len(sequence_splits[i])

        self.num_samples = len(indices)
        return iter(indices)


@SAMPLER.register_module()
class DistributedSamplerTTT(_DistributedSampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False
    ):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0
        self.drop_last = drop_last


        if not self.drop_last:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas
            )
        else:
            self.num_samples = len(self.dataset) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        assert not self.shuffle
        if "data_infos" in dir(self.dataset):
            timestamps = [
                x["timestamp"] / 1e6 for x in self.dataset.data_infos
            ]
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.data_infos
            ]
        else:
            timestamps = [
                x["timestamp"] / 1e6
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)
            vehicle_idx = [
                x["lidar_path"].split("/")[-1][:4]
                if "lidar_path" in x
                else None
                for x in self.dataset.datasets[0].data_infos
            ] * len(self.dataset.datasets)

        sequence_splits = []
        for i in range(len(timestamps)):
            if i == 0 or (
                abs(timestamps[i] - timestamps[i - 1]) > 4
                or vehicle_idx[i] != vehicle_idx[i - 1]
            ):
                sequence_splits.append([i])
            else:
                sequence_splits[-1].append(i)

        indices = []
        for seq in sequence_splits:
            indices.extend(seq)

        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        elif padding_size < 0:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]

        assert len(indices) == self.num_samples

        return iter(indices)

