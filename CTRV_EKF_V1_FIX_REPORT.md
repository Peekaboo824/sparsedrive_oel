**CTRV-EKF V1 修复记录（2026-10-23）**

基于 `feat/ctrv-ekf-v1` 的 `464d7ec38b98b3bd8aedd5d130cc22682ff4dadf`，开始时工作区干净。本轮只修复 V1，原始 OEL 基线仍是 `013b5c34abc3dea3215224a4c765f811d657c3f4`。本轮修复与验证材料按用户要求纳入该分支的 Git 提交，具体 Commit 以 Git 历史为准；此前验证结果保留，本轮结果单独存放在 `tests/ctrv_v1/results/fixes/`。

**1. 原因、修复和文件清单**

| 问题 | 原因 | 本轮处理 |
| --- | --- | --- |
| Dynamics 缺少 Planning Entropy Gate | 原实现只筛历史和可靠 Track，未筛低熵样本 | `dynamics_consistency.py:planning_entropy_trigger()` 为只读副本；`dynamics_loss()` 逐样本在历史模型复制、感知、滤波之前调用 |
| Bootstrap 均值与协方差不一致 | 第2/3次观测重写 `x[2:]`，仍使用原联合协方差 | `agent_track_bank.py:_bootstrap_motion()` 同时重置运动均值、运动方差和陈旧交叉协方差 |
| E0/E1 可能不同时间早停 | 继承 `WindowLossStopHook`，监控的是包含不同损失项的总 Loss | 新 E0 配置只过滤此 Hook，保留其他父 Hook；E1 继承相同固定预算和诊断配置，旧配置不改 |
| Query→Track 关系不易审查 | 原代码已按当前位置/类别关联，但缺乏专门反例测试 | 保留 `associate()`；补充未来轨迹交叉、近邻同类、一对一、类别/距离门控、坐标变换、短遮挡及过期重现测试 |
| 只有局部 Python 诊断属性 | 未进入 MMCV Logger，未归约 Rank 统计 | 新 `diagnostics.py` 的 `DynamicsDiagnosticsHook` 在优化器之后、Logger 之前归约 detached 统计，写 `runner.log_buffer` |
| 缺少梯度冲突分析 | 原工具只记录独立范数 | 新 `gradient_diagnostics.py:gradient_conflict()`，通过 `validate_real_batch.py --gradient-conflict` 显式打开；默认不运行 |

实际改动文件：

- `projects/mmdet3d_plugin/models/dynamics/dynamics_consistency.py`：新增只读门控、先门控后推理、诊断计数/标量；保留原解码、世界坐标和独立 L1。
- `projects/mmdet3d_plugin/models/dynamics/agent_track_bank.py`：增加 `_bootstrap_motion()`；`step()` 调用一致性启动；`predict_prior(..., return_covariance=False)` 可选同时返回预测先验位置协方差供诊断，不改变默认位置接口。
- `projects/mmdet3d_plugin/models/dynamics/diagnostics.py`：新增 `diagnostics_totals/metrics_from_totals/reduce_dynamics_diagnostics/DynamicsDiagnosticsHook.after_train_iter`。
- `projects/mmdet3d_plugin/models/dynamics/gradient_diagnostics.py`：新增 `_summary/gradient_conflict`，仅独立分析工具使用。
- `projects/configs/sparsedrive_small_stage2_ctrv_e0.py`：只过滤 Window 早停、保留无关父 Hook，注册只读诊断 Hook，记录共同 seed=3407。
- `projects/configs/sparsedrive_small_stage2_ctrv_v1.py`：增加启动协方差的两个可配置向量。
- `tools/ctrv_v1/validate_real_batch.py`：增加默认关闭的 `--gradient-conflict`。
- `tools/ctrv_v1/ddp_smoke.py`：读取配置的 custom imports/hooks，保存归约后的诊断。
- `tests/ctrv_v1/test_v1_fixes.py`：18 项新增测试；原21项原样保留，总39项。
- `tests/ctrv_v1/check_diagnostics_ddp.py`：独立两进程、零目标 Rank/不等 Batch 的归约测试。
- `tests/ctrv_v1/results/fixes/`：实际日志、JSON、SHA256 保护记录和 `code_changes.patch`。
- 本报告与原 `CTRV_EKF_V1_REPORT.md` 的修复入口链接。

**2. 原始 OEL 和先验的保护**

以下文件与修复前 Commit **逐字节相同**，SHA256 记录在 `results/fixes/manifest.json`：

- 整个 `models/sparsedrive.py`，包括原 `get_evo_loss()` 和 `forward_train()`；
- `models/sparsedrive_head.py`；
- `models/motion/motion_planning_head.py`；
- `models/dynamics/ctrv_ekf.py`，包括状态、sinc 转移、Jacobian、predict 和 Joseph update；
- 原 `projects/configs/sparsedrive_small_stage2.py`。

因此本轮没有改变原 Evo 的 Top-K、模态选择、cumsum、下一帧感知伪标签、最近邻距离匹配、F.l1_loss、Stop-gradient、Batch 聚合、有效权重或原 Head/Backbone 结构。`associate()` 函数正文也与修复前相同。

EKF 仍使用 Scene-local、截至 t 的模型感知历史。在当前时刻测量更新后，复制该滤波状态执行 `predict(t+1-t)`；拿到的是 t+1 **测量更新前先验**，不是使用 t+1 感知后的后验。新返回的 prior covariance 只用于已有可靠性门限和日志，不进入 Loss 权重。不引入 NN future 伪观测、自适应 Q/R 或新 Loss。

关闭开关或 `lambda_dyn=0`，原 `forward_train` 的 Guard 仍完全跳过 Dynamics，诊断 Hook 也直接返回，不执行额外归约。E0 不加载历史图像。若在 E1 Dataset 配置中只令 lambda=0，其额外历史加载 I/O 仍存在，但模型分支和数值行为已跳过；如需连 I/O 一起关闭，使用 E0 配置。

**3. 熵门控的精确语义**

新函数保留与原代码相同的：`outputs[3]['classification'][-1][b]`、`reshape(3,6)`、`gt_ego_fut_cmd[b].argmax(-1).item()`、指定 Command 的6个 Logits、`softmax`、`-(p*log(p+1e-9)).sum()`，以及 `entropy <= ENTROPY_THRESHOLD` 时跳过。没有转换 Logits 精度或替换数值稳定公式。缺失 Command 时，按原代码**不做熵过滤，直接放行**。

门控只读、不产生训练梯度；高低熵混合 Batch 中只为高熵样本创建/执行历史感知，低熵样本的 Motion/Detection Dynamics 梯度为0，不稀释其他有效 Agent 的梯度。全低熵时返回图连接零，甚至不要求取用历史 payload。

测试涵盖三种 Command、缺失 Command、等于阈值与上下边界；与真正原 `get_evo_loss()` 对输出产生的样本梯度 Mask 对照。另在实际 CUDA FP16 下验证有限低熵和精确阈值行为一致。原精度策略没有更改；原门控面对非有限 Logits/熵的行为也没有另行重新定义。

注意：Dynamics Trigger 现在和原 Evo 一样读取导航 Command；这不等于将 GT 车辆状态用于 EKF。历史感知和滤波仍不读取人工框、Instance ID、GT 速度/轨迹。

**4. Bootstrap 协方差规则**

保留原2/3帧位移启动均值规则，新增对应的不确定性维护。每次启动重写运动均值时：保留当前测量更新后的二维位置块；清空旧位置↔运动及运动内部交叉项；用正的对角运动方差建立块对角 P，再沿用原 `_stable` 的对称/PSD 维护。predict/update 数学不变。

令固定位置观测 R 的最大特征值为 r、相邻观测间隔为 dt、位移长度为 d：

- 速度方差下界：`max(v_floor, 2*r/dt²)`；
- 位移方向方差界：`h=2*r/max(d²,1e-12)`；
- 仅两个观测或前段低速时，omega 未知，保留配置的保守未知方差；
- 三个有效移动观测时，方向差估计 omega，方差界用 `(sqrt(h)+sqrt(h_previous))² / midpoint_dt²`，保守包含共用中间观测的相关影响，并取方差 floor；
- 当前时刻方向包含 `omega*dt/2` 修正，其方差界用 `(sqrt(h)+dt/2*sqrt(var_omega))²`，并取方向 floor；
- 低速时方向/omega 不作为可信零，保留未知方差下界。速度仍为原规则的0。

新增配置：

```python
ekf_bootstrap_variance_floor = [4., .25, 1.]           # v/psi/omega 方差下界
ekf_bootstrap_unobserved_variance = [25., 10., 10.]   # 未观测状态的保守初值
```

默认值是可解释的 V1 初值，不是声称最佳估计。这不是自适应 Q/R：R/Q 始终固定，仅根据已知测量噪声和有限差分几何设定启动 P。丢弃交叉项的启动近似并非严格联合统计最优滤波；更保守的 P 可能使少量短历史 Track 无法通过原位置方差门限。原门限未放宽。

静止、低速、直行、转弯、两帧未知 omega、增大 R/带噪观测、初始化后先验协方差和后续预测稳定性测试全部通过，P 对称、PSD、有限；低速方向/omega 方差保持至少10（默认配置）。

**5. Query→Track 映射**

原关联逻辑正确，本轮保留：Planning Attention Top-K → 原 Detection Query index → 当前 anchor 的世界位置和类别 → 当前 t 的 EKF Track 状态/P/R → Hungarian 一对一门控 → 同一个 Track 的 t+1 prior → 对应 Query 的首未来点 L1。

用 query index 取同一份 Motion 输出，绝不把 query index 当持久 Track ID。新反例测试将未来 NN 点大幅移动/交换到别的车附近，当前 Query→Track pairs 和 EKF targets 保持不变；只改变最终误差。原类别一致性、Mahalanobis/绝对距离门控、短遮挡保留、长时间失效、新 Track 重新 warmup 均已测试。

**6. 诊断归约和归一化**

诊断不进入 `losses` 字典，尤其避免包含字符串 `loss` 的诊断项被 BaseDetector 再次加进总损失。Hook 在正常 Optimizer Hook 后执行，创建新的 detached FP64 统计向量；所有 Rank SUM 归约后计算比率，写 `runner.log_buffer`，沿用 Text/TensorBoard Logger。无目标的 Rank 也参与归约，避免集体通信死锁。没有额外反传、Step、模型 RNG 或参数变动。

| 指标 | 每次全局训练迭代的含义 |
| --- | --- |
| `entropy_trigger_rate` | 触发样本总数 / 所有 Rank 样本总数；包括缺历史样本，缺 Command 按原逻辑算放行 |
| `topk_vehicle_count` | 熵放行样本的 Top-K 机动车总数；即使历史不足仍计入 |
| `reliable_ekf_count` | 实际完成历史滤波后、当前时刻可靠 Track 总数，未要求一定匹配某个 Query |
| `matched_dyn_count` | 最终进入 L1 的有效 Agent 总数 |
| `dyn_active_sample_ratio` | 至少一个有效 Dynamics Agent 的样本数 / 全局样本数 |
| `raw_loss_dyn` / `weighted_loss_dyn` | 按全局有效 Agent 数加权的局部 raw/加权 Loss 均值，即全局 Agent 平均 |
| `*_batch_mean` | 各 Rank 局部 Agent 均值按 Batch 大小加权；区别于全局 Agent 平均 |
| `*_rank_mean` | 各 Rank 局部 Agent 均值等权平均；这是实际 DDP 的 Loss/梯度平均口径 |
| `*_count_per_sample` | 全局 Count / 全局样本数 |
| `ekf_prior_position_variance` | 最终匹配先验 `trace(P_xy)/2` 的 Agent 平均，m² |
| `track_observation_count` | 最终匹配 Track 历史观测次数的 Agent 平均 |
| `track_age` | 当前 t 减 last_seen 的 Agent 平均；本 V1 要求当前被观察，故通常为0 |
| `association_attempted_count` | 有可靠 Track 时尝试关联的机动车 Query 数 |
| `association_rejection_count` | 上述尝试中未获得一对一当前 Track 映射的 Query 数 |
| `prior_rejection_count` | 已关联但未来 prior 不满足现有可靠性检查的数量 |
| `unavailable_dyn_target_count` | 所有触发机动车 Query 中未得到最终 Dynamics target 的数量，含历史不足等原因 |

零分母记录0。计数先归约，不能把各 Rank 触发比例直接平均。MMCV LogBuffer 在默认51步日志窗口上再平均每步指标；若需精确每步观察，设 `log_config.interval=1`。窗口平均的 Agent 均值不是跨整个窗口所有 Agent 的一次加权均值，此处不混淆这两个口径。

**聚合方式完全没改**：原 Evo 是有效样本内部坐标 L1，然后有效样本平均；Dynamics 是本 Rank 所有有效 Agent/二维坐标平均。DDP 对各 Rank 的局部 Loss/梯度再等权平均，因此 Agent 数不同，日志 `weighted_loss_dyn` 可以与训练字典中的 `loss_dyn` 不同，应与 `weighted_loss_dyn_rank_mean` 对照。例如实际一轮测试的 Agent 平均加权值约 `.028538`，Rank 平均与实际训练 Loss 约 `.026126`；这是既有归一化差异，不是额外损失缩放。

**7. E0/E1 公平设置与梯度工具**

旧脚本的 Evo 权重仍是0，未修改。新 E0/E1 都显式 `loss_weight=1.0`；不把旧零权重实验叫作启用 Evo 的对照。两者模型、Map/Planning Loss、Checkpoint、Entropy Threshold=.2、Top-K=10、AdamW/参数组/LR Schedule、数据划分、采样器和最大训练迭代预算均相同。E1 唯一算法差异是该 Dynamics 分支；只增加历史 I/O/推理开销。

父配置当前只有 Window 早停，过滤后增加共同的无梯度 Diagnostics Hook；另有测试模拟其他必需父 Hook，证明不会被删除。配置设共同 seed=3407；**原 `tools/train.py` 会用 CLI seed 覆盖 cfg.seed，且 CLI 默认0**，因此复现命令对两者都显式传 `--seed 3407 --deterministic`。没有为修复这个入口习惯而改原训练代码。实际采样器按相同 seed/Rank 连续10个 Batch 的样本和增强顺序测试相同；历史 RNG 保护的原测试也通过。

梯度冲突工具默认不加载/调用，仅 `validate_real_batch.py --gradient-conflict` 执行。使用同一组共享可训练参数向量，未使用项视为0；分别 `autograd.grad(raw_evo/raw_dyn, retain_graph=True, allow_unused=True)`。不写 `.grad`，不增加 Optimizer Step，保留后续普通反传图。输出 Backbone/Neck/Detection/Motion 的 raw cosine/norm 与指定权重后的 cosine/norm，以及 used/jointly-used 参数数。零范数/无梯度返回 cosine=null 和状态，不造假相似度。

它分析**独立、未包装模型的本地 Batch 梯度**，不是 DDP 归约后的全局梯度；不插入普通 DDP 训练 Hook。单测验证启用分析后正常梯度和 AdamW 更新逐位一致。实际工具成功运行并记录数值，不能据单个样本正余弦宣称两个目标总体相互促进。


实际固定样本的独立梯度范数与 raw 梯度余弦（单个本地 Batch）：

| 模块 | raw Evo norm | raw Dyn norm | raw cosine |
| --- | ---: | ---: | ---: |
| backbone | 25.894914 | 26.422177 | 0.161027 |
| neck | 4.349918 | 3.404553 | 0.055416 |
| detection | 20.023554 | 17.583118 | 0.039124 |
| motion | 1.142098 | 1.145286 | -0.875978 |

**8. 实际验证和日志**

- **39 项单元测试全部通过**，包括原21项及新增18项；CUDA FP16 gate 测试也实际执行，没有跳过。
- 固定真实输出 Golden Replay 四种组合再次逐位通过：disabled/lambda=0 × Evo权重0/1。检查损失字典、raw Evo、Top-K/模态、匹配位置/伪标签、输出梯度、AdamW 参数及状态。
- Replay 中 Map/Planning 值来自冻结 Golden，验证字典保留，不等同 CPU 重跑完整 Head Loss；两个 Head 源码和配置保护校验另证明未改。完整 CUDA 原有非确定性继续单独标注，没有扩大容差或声称逐位通过。
- E0/E1 真实单卡 mini-batch forward/backward，各3次配置 Optimizer 更新成功；E1 raw Dynamics 对 Backbone/Neck/Detection/Motion 与 motion regression 参数均有非零梯度，详见 `e1_batch.json`。EKF CPU 数值仍不接受梯度。
- 两卡 FP16/MMCV Runner 专门 smoke 成功，Rank 的有效 Dynamics Agent 为1/2，归约统计一致。另一个实际两进程用零目标 Rank 和不等 Batch/Agent 数验证全局比率/损失分母，结果通过。
- 真实双 Worker/Batch=2 的跨 Scene、末帧和历史不足数据加载再次通过。
- **原 `tools/train.py` 最终 E0/E1 各4次两卡 FP16 更新完成**，seed=3407、每卡1样本/1 Worker、同一 runtime PKL/Checkpoint、同一4步 LR Schedule；没有清空 custom_hooks，正式诊断 Hook 实际运行。日志没有 FP16 overflow/skip 警告，未保存/覆盖 Checkpoint。
- 最终 E1 四次 `loss_dyn`：`.0132 / .0347 / .0364 / .0392`；全局有效 Agent：`4 / 1 / 3 / 8`；Active Sample Ratio：`.5 / .5 / .5 / 1`。E0 无 `loss_dyn` 键。原始日志/结构化提取见 `training_e0.log`、`training_e1.log` 和 `training_summary.json`。

关键 Git Diff 完整保存在 `results/fixes/code_changes.patch`。核心门控改动为：

```diff
 for b in range(ref.shape[0]):
+    if not planning_entropy_trigger(outputs, data, b, cfg):
+        continue
     # 之后才可能复制 HistoricalPerception、执行历史检测和 EKF
```

启动改动为：

```diff
- ekf.x[2:] = [speed, heading, omega]  # 原来仅重写均值
+ self._bootstrap_motion(track)       # 同步运动均值和块协方差
```

**9. 可复现命令**

仓库根目录；测试 artifact 放工作盘，避免本机器20GB根文件系统被 Golden 填满。`/tmp/ctrv_v1_fixes` 现为工作盘产物目录的符号链接；本轮没有删除或覆盖此前实验数据。

```bash
CTRV_PY=/vepfs-mlp2/c20250505/240906016/jjy/Anaconda/envs/sparsedrive/bin/python
CTRV_ART=/vepfs-mlp2/c20250505/240906016/jjy/tmp_ctrv_v1_fix_artifacts
mkdir -p "$CTRV_ART/tmp"
export TMPDIR="$CTRV_ART/tmp"
PYTHONPATH=. "$CTRV_PY" -m unittest discover -s tests/ctrv_v1 -v
"$CTRV_PY" tools/ctrv_v1/replay_fixed_outputs.py \
  --golden /tmp/ctrv_v1_fixed_outputs/golden.pt --output "$CTRV_ART/replay_new.json"
"$CTRV_PY" tools/ctrv_v1/validate_real_batch.py --mode enabled \
  --config projects/configs/sparsedrive_small_stage2_ctrv_v1.py \
  --output-dir "$CTRV_ART/e1_new" --steps 3 --gradient-conflict
"$CTRV_PY" -m torch.distributed.run --nproc_per_node=2 --master_port=29745 \
  tools/ctrv_v1/ddp_smoke.py --fp16 --output-dir "$CTRV_ART/ddp_new"
"$CTRV_PY" -m torch.distributed.run --nproc_per_node=2 --master_port=29746 \
  tests/ctrv_v1/check_diagnostics_ddp.py --output-dir "$CTRV_ART/reduction_new"
```

实际原入口4步 smoke（分别将配置后缀设为 e0/v1，使用不同 work-dir 和空闲端口；没有 `custom_hooks=[]`）：

```bash
PYTHONPATH=. "$CTRV_PY" -m torch.distributed.run --nproc_per_node=2 --master_port=29747 \
  tools/train.py projects/configs/sparsedrive_small_stage2_ctrv_v1.py \
  --launcher pytorch --work-dir "$CTRV_ART/train_new_v1" \
  --no-validate --seed 3407 --deterministic \
  --cfg-options data.train.ann_file=/tmp/ctrv_v1_runtime_train.pkl \
    data.samples_per_gpu=1 data.workers_per_gpu=1 runner.max_iters=4 \
    log_config.interval=1 checkpoint_config.interval=1000000 checkpoint_config.save_last=False
```

runtime PKL 是此前修正旧绝对 LiDAR 路径的**新文件**，原标注和标签未改。若临时文件已被清理，按初版报告使用 `prepare_runtime_annotations.py` 输出一个新文件，且为 E0/E1 指定同一路径。完整正式训练保留新 E0/E1 的 Early Stop 过滤，不用 smoke 的4步覆盖；如果在本机2 GPU上按20 epoch估算，两个实验共同覆盖 `num_gpus=2 total_batch_size=6 num_iters_per_epoch=2509 data.samples_per_gpu=3 data.workers_per_gpu=3 runner.max_iters=50180 checkpoint_config.interval=50180 evaluation.interval=50180`，同 seed/数据/预训练 Checkpoint，分别输出不同 work-dir。原配置仍按3 GPU估算，不能把2卡结果错误解释为原来的 epoch 数。

**10. 范围与尚未验证项**

代码逻辑和上述 smoke 通过，不代表 Detection mAP、ADE/FDE、Planning L2 或 Collision Rate 提升。本轮不跑完整20 epoch，没有此类性能结论。

原目标域 Map/Planning 监督、Detection GT denoising 和被计算但丢弃的 Detection/Motion GT Loss 均未改变；整个训练协议仍不能称为无目标域感知标注。导航 Command 现在也按原门控方式参与 Dynamics Trigger，但不参与 EKF 状态初始化/更新。

初始化采用保守近似，仍需正式数据的关联误差、短窗口可用率、Q/R/P设置敏感性统计；这些不是本轮最佳参数结论。当前 Track 要求 t 有有效观测，遮挡时跳过 Loss；近邻同类目标仅凭位置/类别仍可能关联错误。没有引入更复杂跟踪或模型。

独立 CUDA 重复依然可能有原子操作、近似并列 Top-K 和模态导致的数值/离散差异；Fixed-output 精确回归和完整CUDA可复现性分别报告，不用放宽阈值掩盖它。新增统计没有改变原两个 Loss 的聚合分母，分析权重时必须使用对应口径。

基础设施失败没有记为通过：本轮 E1 首次保存 Golden 因根 `/tmp` 满中断，仅迁移本轮产物到工作盘并重跑；一次动态 rendezvous 启动失败，改为空闲固定端口后 E0/E1 均跑满4步。没有因此修改算法、跳过测试或删除用户数据。
