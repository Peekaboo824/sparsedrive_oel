# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook
from mmcv.runner.hooks import HOOKS, Hook
import torch
import numpy as np
from collections import deque


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list]
    )
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list]
    )
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):
    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            (
                self.dynamic_milestones,
                self.dynamic_intervals,
            ) = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if (
                    isinstance(module, _BatchNorm)
                    and module.track_running_stats
                ):
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        from projects.mmdet3d_plugin.apis.test import (
            custom_multi_gpu_test,
        )  # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
        )
        if runner.rank == 0:
            print("\n")
            runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class WindowLossStopHook(Hook):
    """
    通用型早停 Hook：基于滑动窗口平均值判断收敛。

    Args:
        window_size (int): 窗口大小。建议设为 0.5 到 1 个 Epoch 的步数。
                           比如总步数 30k，设为 2000-5000 都是通用的。
        patience (int): 容忍多少个 step 平均值没有改善。
        threshold (float): 改善阈值 (0 ~ 1)。0.001 代表 0.1% 的提升。
    """

    def __init__(self, window_size=5000, patience=5000, threshold=0.001):
        self.window_size = window_size
        self.patience = patience
        self.threshold = threshold

        self.loss_queue = deque(maxlen=window_size)
        self.current_sum = 0.0  # 维护滑动和，避免每次 sum()

        self.best_avg_loss = float('inf')
        self.steps_since_improvement = 0

    def after_train_iter(self, runner):
        # 1. 获取 Loss
        if 'loss' not in runner.outputs['log_vars']:
            return

        loss_item = runner.outputs['log_vars']['loss']
        if isinstance(loss_item, torch.Tensor):
            loss_item = loss_item.item()

        # 2. 更新滑动窗口 (O(1) 复杂度)
        if len(self.loss_queue) == self.window_size:
            # 队列满了，把最左边的旧 loss 减掉
            removed = self.loss_queue.popleft()
            self.current_sum -= removed

        self.loss_queue.append(loss_item)
        self.current_sum += loss_item

        # 3. 只有当窗口攒满了才开始判断 (Warmup 阶段不判断)
        if len(self.loss_queue) == self.window_size:
            current_avg = self.current_sum / self.window_size

            # 判断逻辑：平均值是否创新低？
            if current_avg < self.best_avg_loss * (1 - self.threshold):
                self.best_avg_loss = current_avg
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1

            # 4. 触发停止
            if self.steps_since_improvement >= self.patience:
                runner.logger.info(
                    f"\n[AutoStop] Window Average Loss stabilized for {self.patience} steps. "
                    f"Best Avg: {self.best_avg_loss:.4f}, Current Avg: {current_avg:.4f}. "
                    f"Stopping training..."
                )

                # 强制保存权重
                if runner.work_dir:
                    save_filename = f'early_stop_iter_{runner.iter + 1}.pth'
                    runner.logger.info(f"Saving checkpoint to {save_filename}...")
                    runner.save_checkpoint(
                        runner.work_dir,
                        filename_tmpl=save_filename,
                        save_optimizer=False,
                        create_symlink=True
                    )

                runner._max_iters = runner.iter + 1