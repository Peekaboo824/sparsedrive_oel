from inspect import signature
import copy
import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format

    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
            self,
            img_backbone,
            head,
            img_neck=None,
            init_cfg=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            use_grid_mask=True,
            use_deformable_func=False,
            depth_branch=None,
            # TTA/Evo Hyperparameters (Defaults can be adjusted)
            evo_cfg=dict(
                ENTROPY_THRESHOLD=0.2,
                TOP_K_CAUSAL_AGENTS=10,
                PSEUDO_LABEL_THRESHOLD=0.4
            )
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            self.img_backbone.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        self.evo_cfg = evo_cfg

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    # def forward_train(self, img, **data):
    #     # 1. 正常的前向传播 (Current Frame t)
    #     # 这里的 img 是当前帧
    #     feature_maps, depths = self.extract_feat(img, True, data)
    #     model_outs = self.head(feature_maps, data)
    #
    #     # 计算原本的 Loss (包含 GT Planning Loss, GT Detection Loss 等)
    #     # 注意：如果你只想用 Evo Loss 更新感知/预测，可能需要调整 self.head.loss 内部的权重，
    #     # 或者在这里手动过滤 losses 字典。目前代码保留了所有原始 GT Loss。
    #     losses = self.head.loss(model_outs, data)
    #
    #     if depths is not None and "gt_depth" in data:
    #         losses["loss_dense_depth"] = self.depth_branch.loss(
    #             depths, data["gt_depth"]
    #         )
    #
    #     # 2. Evo Self-Supervised Loss 计算逻辑
    #     # 我们需要下一帧的数据 (img_next) 来计算自监督损失。
    #     # 假设 DataLoader 已经修改，能够在 data 中提供 'next_img' 和 'next_img_metas'
    #     if 'next_img' in data and 'next_img_metas' in data:
    #         evo_loss_dict = self.get_evo_loss(model_outs, data, feature_maps)
    #
    #         if evo_loss_dict is not None:
    #             # 建议：在这里硬编码一个权重，或者从 self.evo_cfg 读取
    #             # 例如：自监督 Loss 通常比监督 Loss 大很多或小很多，需要平衡
    #             weight = self.evo_cfg.get('loss_weight', 1.0)
    #
    #             for k, v in evo_loss_dict.items():
    #                 losses[k] = v * weight
    #
    #     return losses

    def forward_train(self, img, **data):
        # 1. 正常的前向传播 (Current Frame t)
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)

        # 2. 获取原始的所有 GT Loss
        all_gt_losses = self.head.loss(model_outs, data)

        # -------------------------------------------------------------
        # 【修改重点】过滤 Loss：去掉感知和预测的 GT Loss，只保留 Planning
        # -------------------------------------------------------------
        losses = {}
        for key, value in all_gt_losses.items():
            losses[key] = value
            # # 这里的筛选逻辑是：只有键名中包含 'planning' 的 Loss 才被保留
            # # 通常 Planning Loss 的键名为 'loss_planning_cls', 'loss_planning_reg' 等
            # if 'planning' in key:
            #     losses[key] = value
            #
            # # 如果你有深度辅助分支并且想保留深度的监督，也可以保留 'depth'
            # # 如果连深度 GT 也要去掉，就注释掉下面这两行
            # if 'depth' in key:
            #     losses[key] = value
            #
            # if 'map' in key:
            #     losses[key] = value

        # 处理 Dense Depth 分支 (如果存在且需要保留，否则也可以注释掉)
        # if depths is not None and "gt_depth" in data:
        #     losses["loss_dense_depth"] = self.depth_branch.loss(depths, data["gt_depth"])

        # -------------------------------------------------------------
        # 3. Evo Self-Supervised Loss 计算 (之前添加的代码)
        # -------------------------------------------------------------
        # Build the prior BEFORE get_evo_loss extracts the future frame with
        # the live training backbone (and updates its BatchNorm buffers).
        # The isolated branch preserves live memories, BN buffers and RNG state.
        dynamics_value = None
        if (self.evo_cfg.get('enable_dynamics_loss', False)
                and self.evo_cfg.get('lambda_dyn', 0.1) != 0.0):
            from .dynamics import dynamics_loss
            dynamics_value = self.evo_cfg.get('lambda_dyn', 0.1) * dynamics_loss(
                self, model_outs, data)

        if 'next_img' in data:
            evo_loss_dict = self.get_evo_loss(model_outs, data, feature_maps)

            if evo_loss_dict is not None:
                # 获取权重配置，默认为 1.0
                weight = self.evo_cfg.get('loss_weight', 1.0)
                for k, v in evo_loss_dict.items():
                    losses[k] = v * weight

        if dynamics_value is not None:
            losses['loss_dyn'] = dynamics_value

        return losses

    def get_evo_loss(self, model_outs_t, data_t, feature_maps_t):
        """
        修正版 2：确保始终返回 Tensor，防止 DDP 报错
        """
        # 获取配置参数
        ENTROPY_THRESHOLD = self.evo_cfg.get('ENTROPY_THRESHOLD', 0.2)
        TOP_K_CAUSAL_AGENTS = self.evo_cfg.get('TOP_K_CAUSAL_AGENTS', 10)
        PSEUDO_LABEL_THRESHOLD = self.evo_cfg.get('PSEUDO_LABEL_THRESHOLD', 0.4)

        next_img = data_t['next_img']
        next_img_metas = data_t['next_img_metas']



        # 2. 准备 data_next
        data_next = copy.copy(data_t)
        data_next['img_metas'] = next_img_metas
        if 'next_projection_mat' in data_t:
            data_next['projection_mat'] = data_t['next_projection_mat']
        if 'next_image_wh' in data_t:
            data_next['image_wh'] = data_t['next_image_wh']



        temp_head = copy.deepcopy(self.head)
        temp_head.eval()  # 强制进入 eval 模式以关闭某些 inplace 更新

        with torch.no_grad():
            next_feature_maps = self.extract_feat(next_img)
            # 2. 注意：这里改用 temp_head 来进行推理
            model_outs_next = temp_head(next_feature_maps, data_next)

        # 3. 推理完直接删除，释放显存
        del temp_head


        # -----------------------------------------------------------
        # Step B: 逐样本计算 Loss
        # -----------------------------------------------------------
        loss_list = []
        batch_size = len(data_t['img_metas'])

        # 为了生成 Zero Loss，我们需要一个参照的 Tensor (获取 device 和 dtype)
        # 这里随便取一个模型输出的 tensor 作为参照
        ref_tensor = model_outs_t[0]['prediction'][-1]

        for b in range(batch_size):
            # ... (中间的筛选逻辑保持不变) ...
            current_metas = data_t['img_metas'][b]
            next_metas = next_img_metas[b]

            # 1. Planning Entropy Check
            plan_cls_scores_b = model_outs_t[3]['classification'][-1][b]
            reshaped_scores = plan_cls_scores_b.reshape(3, 6)

            if 'gt_ego_fut_cmd' in data_t:
                gt_cmd_tensor = data_t['gt_ego_fut_cmd'][b]
                cmd_idx = gt_cmd_tensor.argmax(dim=-1).item()
                scores_for_entropy = reshaped_scores[cmd_idx, :]
                plan_probs = torch.softmax(scores_for_entropy, dim=-1)
                entropy = -(plan_probs * torch.log(plan_probs + 1e-9)).sum()

                if entropy <= ENTROPY_THRESHOLD:
                    continue

                    # 2. Top-k Objects Selection
            attn_weights = model_outs_t[4]['attn_weights_gnn'][b]
            topk_indices_map = model_outs_t[4]['topk_indices_det'][b]
            ego_to_agent_attn = attn_weights[-1, :-1]

            _, top_indices_in_selected = torch.topk(ego_to_agent_attn,
                                                    k=min(TOP_K_CAUSAL_AGENTS, len(ego_to_agent_attn)))
            causal_agent_original_indices = torch.gather(topk_indices_map, 0, top_indices_in_selected)

            # 3. 获取当前帧 (t) 的轨迹预测
            motion_preds_t = model_outs_t[2]['prediction'][-1][b]
            causal_agent_traj_offsets = motion_preds_t[causal_agent_original_indices]
            motion_cls_t = model_outs_t[2]['classification'][-1][b]
            causal_agent_cls_scores = motion_cls_t[causal_agent_original_indices]

            best_mode_indices = causal_agent_cls_scores.argmax(dim=-1)
            K_agents = causal_agent_traj_offsets.shape[0]
            best_mode_indices_expanded = best_mode_indices.reshape(K_agents, 1, 1, 1).expand(K_agents, 1, 12, 2)
            best_traj_offsets = torch.gather(causal_agent_traj_offsets, 1, best_mode_indices_expanded).squeeze(1)

            # 4. 转换为绝对坐标
            relative_path = best_traj_offsets.cumsum(dim=1)
            det_boxes_t = model_outs_t[0]['prediction'][-1][b]
            anchor_pos_t = det_boxes_t[causal_agent_original_indices][:, :2]
            absolute_path_prediction = relative_path + anchor_pos_t.unsqueeze(1)

            prediction_at_next = absolute_path_prediction[:, 0, :]

            # -----------------------------------------------------------
            # 5. 坐标系转换 (将当前帧预测的下一帧坐标，转换到下一帧坐标系下)
            # -----------------------------------------------------------
            T_curr_to_global = current_metas['T_global']
            T_global_to_next = next_metas['T_global_inv']
            T_curr_to_next = T_global_to_next @ T_curr_to_global
            device = prediction_at_next.device
            T_curr_to_next = torch.from_numpy(T_curr_to_next).to(device=device, dtype=torch.float32)

            num_preds = prediction_at_next.shape[0]
            preds_homogeneous = torch.cat([
                prediction_at_next,
                torch.zeros(num_preds, 1, device=device),
                torch.ones(num_preds, 1, device=device)
            ], dim=-1)

            # 转换到 t+1 坐标系
            preds_transformed = (preds_homogeneous @ T_curr_to_next.T)
            prediction_at_next_transformed = preds_transformed[:, :2]

            # -----------------------------------------------------------
            # 6. 与下一帧进行空间距离匹配 (L2 Matching，彻底废弃 ID)
            # -----------------------------------------------------------
            det_boxes_next = model_outs_next[0]['prediction'][-1][b].detach()
            det_confidence_next = model_outs_next[0]['classification'][-1][b].sigmoid().max(dim=-1).values

            # 过滤高置信度的下一帧目标作为 Pseudo-GT
            valid_next_mask = det_confidence_next > PSEUDO_LABEL_THRESHOLD

            target_at_next_list = []
            valid_preds_mask = []

            if valid_next_mask.any():
                valid_boxes_next = det_boxes_next[valid_next_mask][:, :2]  # (M, 2)

                for k in range(num_preds):
                    # 获取该 object 转换后的预测位置
                    pred_pos = prediction_at_next_transformed[k]

                    # 计算与下一帧所有高置信度目标的距离
                    dists = torch.norm(valid_boxes_next - pred_pos.unsqueeze(0), dim=-1)

                    # 找到最近的 Pseudo-GT 目标
                    min_dist, min_idx = torch.min(dists, dim=0)

                    # 如果最近距离小于 3.0 米 (可调节阈值，防止匹配到其他车)
                    if min_dist < 2.0:
                        target_at_next_list.append(valid_boxes_next[min_idx])
                        valid_preds_mask.append(k)

            # -----------------------------------------------------------
            # 7. 计算 Loss 并加入列表
            # -----------------------------------------------------------
            if len(target_at_next_list) > 0:
                target_at_next = torch.stack(target_at_next_list)
                # 取出成功匹配的那些预测坐标
                valid_prediction_at_next_transformed = prediction_at_next_transformed[valid_preds_mask]

                # L1 Loss 更新
                loss_b = F.l1_loss(valid_prediction_at_next_transformed, target_at_next)
                loss_list.append(loss_b)

            # # 5. 与下一帧匹配
            # det_ids_t = model_outs_t[0]['instance_id'][b]
            # causal_agent_ids = det_ids_t[causal_agent_original_indices]
            #
            # det_ids_next = model_outs_next[0]['instance_id'][b]
            # det_boxes_next = model_outs_next[0]['prediction'][-1][b]
            # det_confidence_next = model_outs_next[0]['classification'][-1][b].sigmoid().max(dim=-1).values
            #
            # target_at_next_list = []
            # valid_preds_mask = []
            #
            # for k, target_id in enumerate(causal_agent_ids):
            #     match_idx = (det_ids_next == target_id).nonzero(as_tuple=True)[0]
            #     if len(match_idx) > 0:
            #         matched_original_idx = match_idx[0]
            #         confidence = det_confidence_next[matched_original_idx]
            #
            #         if confidence > PSEUDO_LABEL_THRESHOLD:
            #             target_at_next_list.append(det_boxes_next[matched_original_idx, :2])
            #             valid_preds_mask.append(k)
            #
            # if len(target_at_next_list) > 0:
            #     # 6. 计算 Loss 并加入列表
            #     T_curr_to_global = current_metas['T_global']
            #     T_global_to_next = next_metas['T_global_inv']
            #     T_curr_to_next = T_global_to_next @ T_curr_to_global
            #     device = prediction_at_next.device
            #     T_curr_to_next = torch.from_numpy(T_curr_to_next).to(device=device, dtype=torch.float32)
            #
            #     target_at_next = torch.stack(target_at_next_list)
            #     valid_prediction_at_next = prediction_at_next[valid_preds_mask]
            #
            #     num_preds = valid_prediction_at_next.shape[0]
            #     preds_homogeneous = torch.cat([
            #         valid_prediction_at_next,
            #         torch.zeros(num_preds, 1, device=device),
            #         torch.ones(num_preds, 1, device=device)
            #     ], dim=-1)
            #
            #     preds_transformed = (preds_homogeneous @ T_curr_to_next.T)
            #     prediction_at_next_transformed = preds_transformed[:, :2]
            #
            #     loss_b = F.l1_loss(prediction_at_next_transformed, target_at_next)
            #     loss_list.append(loss_b)

        # -----------------------------------------------------------
        # Step C: 返回结果 (关键修改)
        # -----------------------------------------------------------
        if len(loss_list) > 0:
            return {'loss_evo': torch.stack(loss_list).mean()}
        else:
            # 如果没有计算出任何 Loss，返回一个带梯度的 0.0
            # 必须乘以 0 * ref_tensor.sum() 确保它连在计算图上，防止 DDP 报 unused parameters
            zero_loss = 0.0 * ref_tensor.sum()
            return {'loss_evo': zero_loss}



    # def get_evo_loss(self, model_outs_t, data_t, feature_maps_t):
    #     """
    #     根据 evo_test.py 中的逻辑，计算预测的下一帧位置与实际感知的下一帧位置之间的差异。
    #     Args:
    #         model_outs_t: 当前帧 t 的模型输出 (包含梯度)
    #         data_t: 当前帧 t 的输入数据
    #         feature_maps_t: 当前帧 t 的特征图
    #     """
    #     # 获取配置参数
    #     ENTROPY_THRESHOLD = self.evo_cfg.get('ENTROPY_THRESHOLD', 0.5)
    #     TOP_K_CAUSAL_AGENTS = self.evo_cfg.get('TOP_K_CAUSAL_AGENTS', 10)
    #     PSEUDO_LABEL_THRESHOLD = self.evo_cfg.get('PSEUDO_LABEL_THRESHOLD', 0.4)
    #
    #     next_img = data_t['next_img']
    #     # 注意：next_img_metas 需要是一个 list (batch size)，里面包含字典
    #     next_img_metas = data_t['next_img_metas']
    #
    #     # -----------------------------------------------------------
    #     # Step A: 获取下一帧 (t+1) 的感知结果作为 Pseudo-GT
    #     # -----------------------------------------------------------
    #     with torch.no_grad():
    #         # 提取下一帧特征
    #         # 注意：这里我们为了节省显存，可能不想在下一帧上计算梯度
    #         next_feature_maps = self.extract_feat(next_img)
    #
    #         # 构建下一帧的 data dict 供 head 使用
    #         data_next = copy.copy(data_t)
    #         data_next['img_metas'] = next_img_metas
    #         # 其他必要的字段如 projection_mat 等，如果随时间变化，也需要从 data 中获取对应的 next 版本
    #         # 这里假设 img_metas 包含了大部分变换信息。如果 projection_mat 是独立的 tensor，
    #         # 需要 DataLoader 提供 data['next_projection_mat']。
    #         if 'next_projection_mat' in data_t:
    #             data_next['projection_mat'] = data_t['next_projection_mat']
    #         if 'next_image_wh' in data_t:
    #             data_next['image_wh'] = data_t['next_image_wh']
    #
    #         # Forward 下一帧，设置 is_tta=True (参考 evo_test, 如果 head 支持) 或默认模式
    #         # 这里的 model_outs_next 是 "实际感知到的下一帧"
    #         model_outs_next = self.head(next_feature_maps, data_next)
    #
    #     # -----------------------------------------------------------
    #     # Step B: 处理当前帧 (t) 的输出，计算不确定性和提取轨迹
    #     # -----------------------------------------------------------
    #     # model_outs 结构通常是: [det_output, map_output, motion_output, planning_output, tta_diagnostics]
    #     # 注意：在 train 模式下，self.head 可能返回的结构与 test 模式略有不同，
    #     # 这里假设结构一致，或者我们基于 index/key 访问。
    #
    #     # 1. 检查 Planning 不确定性 (Diagnostic Signal)
    #     # 假设 model_outs_t[3] 是 planning output
    #     plan_cls_scores = model_outs_t[3]['classification'][-1][0]  # Taking first sample in batch for simplicity demo
    #     # 注意：如果 Batch Size > 1，这里需要循环处理。为了代码清晰，这里演示 Batch=1 的逻辑，
    #     # 或者你需要将其向量化处理整个 Batch。
    #
    #     # 简便起见，我们计算 Batch 中每个样本的 Loss 并求平均
    #     total_evo_loss = 0.0
    #     valid_samples = 0
    #
    #     batch_size = len(data_t['img_metas'])
    #
    #     for b in range(batch_size):
    #         # 获取当前样本的特定数据
    #         current_metas = data_t['img_metas'][b]
    #         next_metas = next_img_metas[b]
    #
    #         # Planning Entropy Check
    #         # 假设 output 结构: (B, ...)
    #         plan_cls_scores_b = model_outs_t[3]['classification'][-1][b]  # (18,)
    #         reshaped_scores = plan_cls_scores_b.reshape(3, 6)
    #
    #         # 获取 GT Command 对应的 scores (用于计算 entropy)
    #         if 'gt_ego_fut_cmd' in data_t:
    #             gt_cmd_tensor = data_t['gt_ego_fut_cmd'][b]
    #             cmd_idx = gt_cmd_tensor.argmax(dim=-1).item()
    #             scores_for_entropy = reshaped_scores[cmd_idx, :]
    #             plan_probs = torch.softmax(scores_for_entropy, dim=-1)
    #             entropy = -(plan_probs * torch.log(plan_probs + 1e-9)).sum()
    #
    #             # 如果不确定性低，可能不需要 Evo Loss？或者加权？
    #             # evo_test.py 中是: if is_uncertain ... then update.
    #             # 在训练中，我们可以始终计算，或者使用 entropy 作为 loss 权重。
    #             # 这里为了完全复刻 evo_test，我们设置阈值。
    #             if entropy <= ENTROPY_THRESHOLD:
    #                 continue  # Skip this sample
    #
    #         valid_samples += 1
    #
    #         # 2. Top-k Objects Selection (关键对象选择)
    #         # 获取 Attention Weights
    #         # 假设 model_outs_t[4] 是 tta_diagnostics，包含 attn_weights_gnn
    #         attn_weights = model_outs_t[4]['attn_weights_gnn'][b]  # (901, 51)
    #         topk_indices_map = model_outs_t[4]['topk_indices_det'][b]  # (50,)
    #         ego_to_agent_attn = attn_weights[-1, :-1]  # (50,)
    #
    #         # 找到 Attention 最高的 Top-K
    #         _, top_indices_in_selected = torch.topk(ego_to_agent_attn,
    #                                                 k=min(TOP_K_CAUSAL_AGENTS, len(ego_to_agent_attn)))
    #         causal_agent_original_indices = torch.gather(topk_indices_map, 0, top_indices_in_selected)
    #
    #         # 3. 获取当前帧 (t) 的轨迹预测 (Targeted Update)
    #         motion_preds_t = model_outs_t[2]['prediction'][-1][b]  # (900, 6, 12, 2)
    #         causal_agent_traj_offsets = motion_preds_t[causal_agent_original_indices]  # (K, 6, 12, 2)
    #         motion_cls_t = model_outs_t[2]['classification'][-1][b]  # (900, 6)
    #         causal_agent_cls_scores = motion_cls_t[causal_agent_original_indices]  # (K, 6)
    #
    #         # 找到概率最高的模态
    #         best_mode_indices = causal_agent_cls_scores.argmax(dim=-1)
    #         K_agents = causal_agent_traj_offsets.shape[0]
    #         best_mode_indices_expanded = best_mode_indices.reshape(K_agents, 1, 1, 1).expand(K_agents, 1, 12, 2)
    #         best_traj_offsets = torch.gather(causal_agent_traj_offsets, 1, best_mode_indices_expanded).squeeze(
    #             1)  # (K, 12, 2)
    #
    #         # 4. 转换为绝对坐标
    #         relative_path = best_traj_offsets.cumsum(dim=1)
    #         det_boxes_t = model_outs_t[0]['prediction'][-1][b]
    #         anchor_pos_t = det_boxes_t[causal_agent_original_indices][:, :2]
    #         absolute_path_prediction = relative_path + anchor_pos_t.unsqueeze(1)
    #
    #         # 取第一个预测点作为 t+1 时刻的预测位置
    #         prediction_at_next = absolute_path_prediction[:, 0, :]  # (K, 2)
    #
    #         # 5. 与下一帧 (t+1) 的检测结果进行匹配
    #         # 下一帧的“真值”来自 model_outs_next (No Grad)
    #         det_ids_t = model_outs_t[0]['instance_id'][b]  # (900,)
    #         causal_agent_ids = det_ids_t[causal_agent_original_indices]
    #
    #         det_ids_next = model_outs_next[0]['instance_id'][b]
    #         det_boxes_next = model_outs_next[0]['prediction'][-1][b]
    #         det_confidence_next = model_outs_next[0]['classification'][-1][b].sigmoid().max(dim=-1).values
    #
    #         target_at_next_list = []
    #         valid_preds_mask = []
    #
    #         for k, target_id in enumerate(causal_agent_ids):
    #             # 尝试通过 ID 匹配
    #             match_idx = (det_ids_next == target_id).nonzero(as_tuple=True)[0]
    #             if len(match_idx) > 0:
    #                 matched_original_idx = match_idx[0]
    #                 confidence = det_confidence_next[matched_original_idx]
    #
    #                 # 只有置信度高的才作为 Pseudo GT
    #                 if confidence > PSEUDO_LABEL_THRESHOLD:
    #                     target_at_next_list.append(det_boxes_next[matched_original_idx, :2])
    #                     valid_preds_mask.append(k)
    #
    #         if len(target_at_next_list) > 0:
    #             # 6. 坐标变换与 Loss 计算
    #             T_curr_to_global = current_metas['T_global']
    #             T_global_to_next = next_metas['T_global_inv']
    #             T_curr_to_next = T_global_to_next @ T_curr_to_global
    #             device = prediction_at_next.device
    #             T_curr_to_next = torch.from_numpy(T_curr_to_next).to(device=device, dtype=torch.float32)
    #
    #             target_at_next = torch.stack(target_at_next_list)
    #             valid_prediction_at_next = prediction_at_next[valid_preds_mask]
    #
    #             # 转换预测点到下一帧坐标系
    #             num_preds = valid_prediction_at_next.shape[0]
    #             preds_homogeneous = torch.cat([
    #                 valid_prediction_at_next,
    #                 torch.zeros(num_preds, 1, device=device),
    #                 torch.ones(num_preds, 1, device=device)
    #             ], dim=-1)
    #
    #             preds_transformed = (preds_homogeneous @ T_curr_to_next.T)
    #             prediction_at_next_transformed = preds_transformed[:, :2]
    #
    #             # L1 Loss
    #             loss_b = F.l1_loss(prediction_at_next_transformed, target_at_next)
    #             total_evo_loss += loss_b
    #
    #     if valid_samples > 0:
    #         return {'loss_evo': total_evo_loss / valid_samples}
    #     else:
    #         return None

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)

