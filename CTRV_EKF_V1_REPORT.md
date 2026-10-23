**CTRV-EKF V1 实施与验证记录（2026-10-23）**

本次只增加独立 `loss_dyn`。原 `get_evo_loss()`、Detection/Map/Motion/Planning Head 和旧配置文件均未修改。关闭分支或令 `lambda_dyn=0`，模型不执行历史推理、滤波、随机数操作，也不添加损失键。关闭时如仍使用 E1 的 Dataset 配置，历史图像的加载开销仍存在；完整 E0 配置同时关闭历史加载。

**1. Baseline 审计与恢复点**

- 初始工作区干净，初始分支 `master`。
- Baseline Commit：`013b5c34abc3dea3215224a4c765f811d657c3f4`。
- 独立工作分支：`feat/ctrv-ekf-v1`。实现、测试和验证材料在该分支提交，具体 Commit 以 Git 历史为准；未覆盖旧实验、Checkpoint 或标注。
- 修改前原始模型/数据集源码保存在 `tests/ctrv_v1/baseline/*.txt`，原始真实 mini-batch Golden 为 `/tmp/ctrv_v1_original_golden.pt`；日志在 `tests/ctrv_v1/results/original_baseline.log`。大的 Golden 张量没有放入源码目录。
- 真实入口 `scripts/train.sh` 调用 `tools/dist_train.sh projects/configs/sparsedrive_small_stage2.py 2 --deterministic`。
- 实际配置：`num_gpus=3`、`total_batch_size=9`、每卡 3 样本/3 Worker、20 epochs，按 3 卡估算 1673 iter/epoch；脚本和当前实际可见 GPU 均为 2 卡，因此原入口的有效全局 Batch 是 6，并非配置假定的 9。没有改写该旧设置。
- `load_from=ckpt/nuscenes_boston_sg_split_v2/sparsedrive_stage2.pth`，文件实际存在。按仓库 `readme.md`，这是 Singapore 两阶段预训练权重；没有重新执行预训练或独立认证权重来源。
- 目标训练集 `data/infos/custom_split_boston/nuscenes_boston_infos_train.pkl`，15058 帧；验证集同目录 `nuscenes_boston_infos_val.pkl`。仓库说明这是 Boston 8:2 划分。
- **当前脚本的 `evo_cfg.loss_weight=0.0`**：原时序损失虽执行，其有效梯度为零。修改前真实样本 index=10 的 raw `loss_evo=0.2010682821`，加权后为 0。不能把这个实验描述成启用了 Temporal Consistency 的 OEL。
- `sg_boston_v2` 是 Singapore→Boston 预训练划分，`sg_boston_v3` 是反向划分，不能混作同一适应实验。这些 stage2 配置未显式指定 `evo_cfg`，按模型默认 `.get('loss_weight', 1.0)` 有效 Evo 权重是 1；但路径含旧机器绝对地址，且训练数据/预训练阶段与当前脚本不同。
- IterBasedRunner 实际选择 `GroupInBatchSampler`；`with_seq_flag=True`、`sequences_split_num=2`、一致序列增强，并不保证不同 Batch/Rank 构成同一条完整 Scene 流。
- 有效 Hook 为 `WindowLossStopHook(window_size=1600, patience=3200, threshold=.0007)`。配置中的 PerceptionFreezeHook 仅为注释；Backbone `frozen_stages=-1`、`norm_eval=False`。没有新增冻结。
- AdamW：`lr=5.625e-5`、`weight_decay=.001`，Backbone 学习率倍率 `.1`，裁剪范数 25；FP16 固定 loss scale 32；DDP 默认 `find_unused_parameters=True`。本次均保持原设置。

**2. 原始损失路径和监督协议**

`forward_train → extract_feat → head → head.loss`，然后原样调用 `get_evo_loss` 并乘原 `evo_cfg.loss_weight`。原损失内部保留命令对应 Planning Entropy 门控、Attention Top-K、原 query 索引映射、最高分模态、12 步 offset cumsum、检测 anchor 平移、第一未来点、`T_global_inv(next) @ T_global(current)`、高置信伪检测和距离 `<2.0m` 的逐目标最近邻、`F.l1_loss`、有效样本 Batch 均值、原图连接零回退。

下一帧使用 deepcopy 的 eval Head、no_grad；但是原特征提取仍调用主干，训练态 BN 会更新，GridMask/随机流也有原行为。这些均未修改。新动力学分支必须在原下一帧特征提取之前计算，防止复制已经受未来帧影响的 BN buffers。专门回归测试先复现了这一风险，再验证调用顺序修复。

最终已有损失键为：`map_loss_cls_0..5`、`map_loss_line_0..5`、`planning_loss_cls_0`、`planning_loss_reg_0`、`planning_loss_status_0`，有 `next_img` 时加 `loss_evo`。启用 V1 只再加一个 `loss_dyn`。Dense Depth Loss 仍注释；Detection/Motion GT Loss 仍计算但不加入最终字典。

| 字段 | 原代码真实用途 | V1 EKF 是否读取 |
| --- | --- | --- |
| `gt_bboxes_3d`, `gt_labels_3d` | Detection GT loss/sampling，训练前向 denoising anchors | 否 |
| GT `instance_id`/`instance_inds` | Dataset/Adaptor 仍提供，原 DN 路径可查询特定元数据接口；没有当作可靠在线 ID | 否 |
| `gt_agent_fut_trajs`, `gt_agent_fut_masks` | Motion GT loss 的采样/计算，结果未加入总损失 | 否 |
| `gt_map_labels`, `gt_map_pts` | Map Loss，保留且参与优化 | 否 |
| `gt_ego_fut_trajs`, `gt_ego_fut_masks` | Planning 轨迹与模态监督，保留且参与优化 | 否 |
| `gt_ego_fut_cmd` | Planning 的命令分支选择、原 Evo Entropy 选择；测试管线也提供 | 否 |
| `ego_status` | Planning Status Loss | 否 |
| `gt_depth` | 加载/生成，Dense Depth Loss 当前未计入 | 否 |

**整个既有训练流程不是“目标域 GT 仅离线评估”的协议**：地图和 Planning GT 明确参与训练，Detection GT 还用于 denoising。本次按要求没有改变这一监督协议。新 EKF 的初始化、关联、测量更新与先验不读取这些 GT；不能把这个局部性质推广为整个 OEL 训练没有 GT 使用。

**3. 修改清单与生产接口**

| 文件 | 内容 |
| --- | --- |
| `projects/mmdet3d_plugin/models/sparsedrive.py` | 仅新增受配置控制的独立分支；在原 Evo 前生成先验，原 Evo 后添加单独加权结果 |
| `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py` | 默认关闭的历史参数、scene 索引、CPU-only 历史 payload 和单独下一帧有效标志；恢复历史管线消耗的随机流 |
| `projects/mmdet3d_plugin/datasets/dynamics_history.py` | `scene_indices` 建同 Scene 时间索引；`history_indices` 取截至 t 的 H 帧；`sensor_frame_info` 仅构造相机/位姿/时间，不调用 `get_ann_info` |
| `projects/mmdet3d_plugin/models/dynamics/ctrv_ekf.py` | `wrap_angle`、`covariance` 参数校验、`transition_and_jacobian` 稳定 sinc CTRV 及解析 Jacobian；`CTRVEKF` 的 initialize/predict/update/get_prior_position/get_prior_covariance，内部 `_stable` 对称化与特征值维护 |
| `projects/mmdet3d_plugin/models/dynamics/agent_track_bank.py` | `associate` 类别一致、Mahalanobis+绝对距离门控、一对一 Hungarian；`AgentTrackBank._new_filter/step/reliable_tracks/predict_prior` 进行窗口内初始化、递归更新、可靠性筛选和复制后先验预测 |
| `projects/mmdet3d_plugin/models/dynamics/dynamics_consistency.py` | `compute_dynamics_loss` 独立 masked L1；`to_world` FP32 世界 BEV 变换；`selected_predictions` 独立复现原 Top-K/最高模态/cumsum 解码；`observations` 筛选模型机动车伪检测；`HistoricalPerception.__init__/__call__` 隔离最新模型；`dynamics_loss` 按样本滤波、关联、聚合及诊断 |
| `projects/mmdet3d_plugin/models/dynamics/__init__.py` | 导出独立动力学接口 |
| `projects/configs/sparsedrive_small_stage2_ctrv_e0.py` | 明确启用 Evo 的对照配置，历史/动力学关闭 |
| `projects/configs/sparsedrive_small_stage2_ctrv_v1.py` | E1，同 E0，仅增加动力学分支及其历史输入 |
| `tests/ctrv_v1/test_ctrv_v1.py` | 21 项数学/因果性/隔离/梯度/回归/配置/容器测试 |
| `tests/ctrv_v1/baseline/*`, `results/*` | 修改前源码快照、实际测试日志、梯度/数据/回归 JSON 与 manifest |
| `tools/ctrv_v1/capture_baseline.py` | `seed/make_batch/norms/cpu/main`：固定真实 batch、内存路径解析、原始 Golden 捕获 |
| `tools/ctrv_v1/validate_real_batch.py` | `bind_original/main`：原版方法绑定、forward_train、独立梯度、配置优化器三步 smoke |
| `tools/ctrv_v1/compare_golden.py` | `flat_tensors/metrics/main`：独立 CUDA 重放严格比较，保留失败项 |
| `tools/ctrv_v1/replay_fixed_outputs.py` | `method/ReplayHead/run/exact/main`：固定真实输出和伪检测精确回归，隔离 CUDA 前向非确定性 |
| `tools/ctrv_v1/ddp_smoke.py` | 双实际 GPU、MMCV DDP/IterBasedRunner/FP16 Optimizer Hook 前后向及更新 |
| `tools/ctrv_v1/validate_data_loader.py` | 真实双 Worker、Batch=2、跨 Scene/边界/历史不足验证 |
| `tools/ctrv_v1/prepare_runtime_annotations.py` | 只写**新** runtime 标注副本，修正旧绝对 LiDAR 路径，不改变标签/划分，拒绝覆盖已有输出 |
| `CTRV_EKF_V1_REPORT.md` | 本记录及复现命令 |

**4. CTRV-EKF 输入输出与历史模式**

- 状态 `[global_x, global_y, speed, motion_heading, yaw_rate]`，单位 m/s/rad/s；输入观测仅为模型检测中心的世界 BEV `[x,y]`，按真实秒时间差递归。
- 初始化只有位置；前 2–3 次关联观测用位移估计速度和运动方向变化。低速不信任航向，初始化角速度为 0，保留较大初始不确定性；至少 3 次观测才可靠。检测框 yaw/预测 velocity 均不用于初始化。
- CPU NumPy FP64 滤波，固定可配置 Q/R/初始 P；Q 为每次 predict 的固定协方差，不宣称最优调参。使用解析 sinc Jacobian、solve、Joseph covariance update、角度归一化、正速度等价表达与异常协方差失效。原始 predict 的先验单独保存，update 后也不会覆盖 get_prior_*。
- `predict(dt)` 输出先验二维位置；`get_prior_position()` 为 `(2,)`，`get_prior_covariance()` 为位置先验 `(2,2)`。Track 的 prior 在复制的滤波器中预测，绝不 update t+1。
- 默认 H=5（包含 t），每个样本独立重建同 Scene 的历史，不以 query index/GT ID 关联，不共享跨 Batch/Rank 状态。缺测会按秒龄失效；要求当前 t 被有效观测且满足数量/协方差门限。
- 历史来自真实过去相机图像。sensor-only Pipeline 无 GT/导航字段；每次 forward 复制**当前在线模型权重**的 Backbone/Neck/Detection Head，eval+no_grad，清空检测 temporal bank 和 DN memory，仅在该样本历史窗口内递归感知。主模型的 BN、训练模式、缓存和随机流不受这段额外推理改变。
- **不使用固定教师或历史伪观测缓存。接收在线更新后模型的感知输出。仅窗口内连续维护状态，没有无限长、跨 Batch 的连续 Track Bank。**
- 时间戳原始微秒除 `1e6`，历史传入感知模型时保留 FP64，防止 Unix 秒 FP32 丢掉 0.5s 间隔。
- 单独 `has_valid_next_frame` 和历史信息放入 CPU-only `DataContainer`；DDP 后各样本自己移动历史输入到本 Rank GPU，长度可变。Scene 末尾仍保留原 `next_img=current_img` 的 Evo 行为；仅 Dynamics 跳过。
- 动力学先验仅用截至 t 的图像、位姿、时间和 t+1 的预测时间间隔；不用下一帧图像/检测/ego pose。若下一帧间隔偏离首个 motion point 的 0.5s 超过配置容差，则跳过该动力学监督。
- `T_global` 实际为 `ego2global @ lidar2ego`，BBoxRotation 时还乘增强逆矩阵。使用列向量等价的 `homogeneous @ T_global.T`；神经首未来点和感知观测均按本仓库 BEV 的 `z=0` 提升到世界二维坐标。不是将 offset 直接当世界位置。非水平姿态下存在 BEV 平面近似，未声称完整 3D 运动模型。

**5. forward_train 接入与配置权重**

保持原前向和 `head.loss` 后，先计算可选 `dynamics_value=lambda_dyn*dynamics_loss(...)`，然后原样执行 `get_evo_loss` 和原 `loss_weight` 缩放，最后只新增 `losses['loss_dyn']=dynamics_value`。EKF target detach；神经解码和当前 detection anchor 保持梯度。有效目标按所有 Batch 的有效 Agent/二维坐标 mean L1，空集合由 Motion 和 Detection 输出构造图连接零。

旧配置仍是 Evo 权重 0。本次提供的 **E0/E1 是显式激活 Evo 权重 1.0 的新实验对**（与模型未配置该项时的默认值一致），不是声称旧脚本原来已启用它。E1 继承 E0，原 Map/Planning/Temporal 权重不变，`lambda_dyn=.1` 独立管理。配置测试检查除了 Dynamics 模型/历史输入参数外 E0/E1 所有有效参数相同。

`dynamics_vehicle_class_ids=[0,1,2,3,4]` 对应本配置的 car/truck/construction_vehicle/bus/trailer，不包括 motorcycle；不能套用 Dataset 默认类顺序。E1 中 Q/R/P、confidence、association gate、距离、年龄、位置方差均为初始可配置实验值。

**6. 已执行验证与明确范围**

| 验证 | 实际结果 |
| --- | --- |
| 21 项 CPU 单元测试 | 通过；包括直行/转弯/近零 omega Jacobian/不同 dt/缺测/低速/warmup/500步协方差/角度/过期/类别一对一关联 |
| 因果性 | 修改 t+1 图像/pose 不改变固定历史先验；校验 BN 复制必须发生于原下一帧前；在线权重复制、主模型 BN/缓存不变测试通过 |
| Scene/乱序 | 合成不同 Scene、interleaved 时间索引、顺序交换、边界与无历史通过；没有全局 bank |
| `get_evo_loss` | AST 与修改前完全一致；该函数正文未改 |
| 固定**真实网络输出**回放 | Evo 权重 0/1 × disabled/lambda=0 四组逐位通过：损失字典、raw Evo、Top-K、最高模态、匹配位置、伪检测 target、对输出的梯度、AdamW 更新及状态 |
| 独立 CUDA 完整前后向重复 | **严格全量比较未通过**。原版对原版本身也会改变近似并列 Top-K/部分模态；不能报告端到端逐位确定性 |
| 真实梯度与 3 步优化 | 通过；使用配置 AdamW 参数组，首样本 2 个有效 Dynamics Agent，后两样本有效数见 `real_batch.json` |
| 两卡 FP16 MMCV Runner | 通过；2 Rank 实际 forward/backward/step，梯度同步；有效 Dynamics Agent 为每 Rank 1/2（见日志） |
| 原 `tools/train.py` 入口 | E0/E1 各4次真实更新通过；两卡、每卡1个Worker、每卡1样本、FP16，E1四次均有非零 loss_dyn，E0无该键 |
| 真实数据加载 | 通过；2 Worker，Batch=2，4 样本含不同 Scene、末帧、Scene 首帧；历史无未来、无 GT 字段、DC collate 正确 |

独立 CUDA 重复的比较详见 `results/golden_comparison.json`：原版自身 raw Evo 差约 `3.12e-5`，关闭/零权重约 `3.62e-5` / `2.44e-5`；Map/Planning Loss 最大绝对差小于 `2.1e-4`，全参数梯度相对 L2 差约 1.5%–1.9%。但 Top-K/模态存在离散变化，不能靠扩大容差宣布全部通过。固定真实输出的精确回放与源码不变证明的是损失算法及其输出梯度不变；没有把这等同于完整 Backbone CUDA 执行确定性。Golden/比较工具均保留可复查。


实际固定样本独立 raw loss 反传范数（来自最终配置优化器测试）：

| 模块 | raw loss_evo | raw loss_dyn |
| --- | ---: | ---: |
| backbone | 25.904316 | 26.491343 |
| neck | 4.355371 | 3.418405 |
| detection | 20.047679 | 17.602498 |
| motion | 1.142242 | 1.145432 |

固定输出回放中的 Map/Planning 值取自 Golden，验证的是损失字典的保留；并未在 CPU 上重新执行完整 Head Loss。两个原 Head 文件与旧配置的逐字节 SHA256 校验也记录在 manifest，均与 Baseline 一致。

真实 `loss_dyn` 的独立反传确实到达 Backbone、Neck、Detection、Motion Head，以及 `motion_reg_branch`，而不是仅对 Motion 输出叶子做测试。Motion query 由 `instance_feature + anchor_embed` 与 mode query 组成，上游 detection/shared features 保持可微；argmax/Top-K/关联本来不求导。范数详见 `results/real_batch.json`，不能据此推断 Detection mAP 已改善。

效率测量为当前共享 A100 环境的单样本 smoke：历史推理约 1.1s，模型复制约 .06–.09s，滤波和关联约 .01–.02s；新增分支约 1.2s/样本，主要成本是重复 H 帧感知。峰值分配约 4.6GiB 是单卡 FP32 mini-batch 整体值，非独占 GPU 基准或额外显存净增。首次冷前向约 12–15s，不适合直接当稳定吞吐。纯 L1 kernel 与 DataLoader 额外耗时未做单独可靠基准。

**7. 测试与训练命令**

在仓库根目录执行。可用环境已实际验证为 Python3.8 / Torch1.13.0+cu116 / MMCV1.7.1 / MMDetection2.28.2，不是默认 Python（默认 Python 无 torch）。

```bash
CTRV_PY=/vepfs-mlp2/c20250505/240906016/jjy/Anaconda/envs/sparsedrive/bin/python
PYTHONPATH=. "$CTRV_PY" -m unittest discover -s tests/ctrv_v1 -v
"$CTRV_PY" tools/ctrv_v1/replay_fixed_outputs.py \
  --golden /tmp/ctrv_v1_fixed_outputs/golden.pt \
  --output /tmp/ctrv_v1_fixed_replay_again.json
"$CTRV_PY" tools/ctrv_v1/validate_real_batch.py \
  --mode enabled --config projects/configs/sparsedrive_small_stage2_ctrv_v1.py \
  --output-dir /tmp/ctrv_v1_new_smoke --steps 3
"$CTRV_PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/ctrv_v1/ddp_smoke.py --fp16 --output-dir /tmp/ctrv_v1_new_ddp
"$CTRV_PY" tools/ctrv_v1/validate_data_loader.py --output /tmp/ctrv_v1_new_data.json
```

若 `/tmp` Golden 已被清理，可用 `validate_real_batch.py --mode original --output-dir /tmp/ctrv_v1_fixed_outputs` 从 Baseline 方法和现有 Checkpoint 重新捕获；原始源码快照仍在仓库。独立 CUDA 对照用 `original` 两次、`disabled`、`zero` 四次分别捕获，再运行 `compare_golden.py --reference ... --control ... --disabled ... --zero ... --output ...`；当前环境下这个严格检查会失败，原因如上。

本机器标注的 LiDAR 路径原指向 `/root/autodl-tmp/SparseDrive_OEL/...`。测试工具只在内存修正；正式入口需先准备**新文件**。本次已生成 `/tmp/ctrv_v1_runtime_train.pkl`（解析全部15058条路径），原 PKL 未改。工具拒绝覆盖，重跑请选新输出名：

```bash
"$CTRV_PY" tools/ctrv_v1/prepare_runtime_annotations.py \
  --input data/infos/custom_split_boston/nuscenes_boston_infos_train.pkl \
  --output /tmp/ctrv_v1_runtime_train_new.pkl
```

本次还通过原 `tools/train.py` 分别执行了 E0/E1 各4步，使用上述 runtime PKL、seed=3407、2 GPU、每卡1样本/1 Worker、FP16；CLI 仅为 smoke 覆盖 `runner.max_iters=4`、`log_config.interval=1`、`checkpoint_config.interval=1000000`、`checkpoint_config.save_last=False`、`custom_hooks=[]`，并关闭验证。原配置没有改写，也没有保存或覆盖 Checkpoint。E1 的四次 DDP 平均加权 `loss_dyn` 为 `0.0126/0.0323/0.0328/0.0368`；E0 无 `loss_dyn` 键。完整日志见 `results/training_e0.log` 与 `training_e1.log`。这只是训练连通性验证，不能据训练 Loss 解释性能改善。

```bash
PYTHONPATH=. "$CTRV_PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/train.py projects/configs/sparsedrive_small_stage2_ctrv_v1.py \
  --launcher pytorch --seed 3407 --deterministic --no-validate \
  --work-dir /tmp/ctrv_v1_training_smoke_new \
  --cfg-options data.train.ann_file=/tmp/ctrv_v1_runtime_train.pkl \
    data.samples_per_gpu=1 data.workers_per_gpu=1 runner.max_iters=4 \
    log_config.interval=1 checkpoint_config.interval=1000000 \
    checkpoint_config.save_last=False custom_hooks=[]
```

当前两卡机器上，完整 20 epoch 的 E0/E1 公平运行方式如下；将 E0/E1 的相同硬件和日程参数一并覆盖，工作目录各不相同。**这是后续完整实验命令，不代表本次已经训练20 epoch。**

```bash
for CTRV_EXP in e0 v1; do
  PYTHONPATH=. "$CTRV_PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/train.py "projects/configs/sparsedrive_small_stage2_ctrv_${CTRV_EXP}.py" \
    --launcher pytorch --seed 3407 --deterministic \
    --work-dir "/tmp/ctrv_v1_full_${CTRV_EXP}" \
    --cfg-options data.train.ann_file=/tmp/ctrv_v1_runtime_train.pkl \
      num_gpus=2 total_batch_size=6 num_iters_per_epoch=2509 \
      data.samples_per_gpu=3 data.workers_per_gpu=3 \
      runner.max_iters=50180 checkpoint_config.interval=50180 evaluation.interval=50180
done
```

如需完全复现旧脚本零 Evo 权重，使用未改的旧配置与相同路径解析副本，不要把它标成上述 activated E0。E2 用 E1 的 `--cfg-options model.evo_cfg.lambda_dyn=0.05`（另试 `.1/.2/.5`），确保 E0/E1 的 Evo 权重、Checkpoint、随机种子、数据和日程保持一致。

**8. 尚未完成与风险**

- 未执行完整 E0/E1/E2 跨域训练或离线评估；没有 ADE/FDE、mAP/NDS、Planning L2、Collision Rate 或真实 GT 状态误差的改进结论。
- GT 仅离线的完整监督协议不成立，原因是继承的 Map/Planning/DN 行为；本次没有偷偷删除这些监督。新 Dynamics 分支本身不以目标域人工车辆状态作滤波监督。
- Top-K 和可靠 EKF 的交集较小，可能为零；需要真实数据统计覆盖率、遮挡/近邻关联错误、直行和转弯误差，以及不同 Q/R/阈值的敏感性。
- 状态仅在窗口内连续；短窗口/低速/噪声 heading/类别变化会影响可用性。位置 Hungarian 不是完整生产跟踪器。
- 原 CUDA 执行本身非确定；严格端到端回归验收尚未全部满足。原 Evo 的未来主干 BN 更新、Scene 末帧自复制及原 GT denoising 作为独立后续审议项记录，未在 V1 改写。
- 历史开销随 H×Batch 增长；未做固定教师缓存，不能将当前开销结论推广到大 Batch、多节点长时训练。
- 单步实际 DDP 和双 Worker smoke 已验证；没有执行多节点长时故障恢复或无限流式运行。
