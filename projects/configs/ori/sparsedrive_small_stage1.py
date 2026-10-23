# ================ base config ===================
version = 'mini'
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

total_batch_size = 20
num_gpus = 2
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)  # 输入图像大小


# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]
num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)  # 感兴趣区域（围绕自车）

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4 # history + current

embed_dims = 256
num_groups = 8  # 注意力头
num_decoder = 6  # operation_order执行多少遍
num_single_frame_decoder = 1  # 不带时序的执行多少遍
num_single_frame_decoder_map = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]  # 下采样倍数
num_levels = len(strides)  # 特征图数量
num_depth_layers = 3  # 网络层数
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=False,
)

model = dict(
    type="SparseDrive",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",  # 使用resnet作为图像特征提取器
        depth=50,  # 选用50层的resnet，即resnet-50
        num_stages=4,  # resnet有4个主要阶段 每一个阶段都会将特征图尺寸减半，通道数加倍
        frozen_stages=-1,  # 不冻结参数 所有参数都可以训练
        norm_eval=False,  # 归一化层是否在评估模式下运行 训练模式是计算当前小批量数据的均值和方差 评估模式是停止计算当前批次的均值和方差，直接使用整个训练中累积的均值和方差 这里是训练模式，因为批量大小足够大
        style="pytorch",  # 使用pytorch官方的resnet实现来实现这里的resnet
        with_cp=True,  # 是否采用梯度检查点技术 主要是为了降低GPU显存消耗的 但是会增大计算量 以时间换空间
        out_indices=(0, 1, 2, 3),  # 表示需要输出所有4个阶段的特征图。这为后续的 FPN 提供了多尺度的输入。
        norm_cfg=dict(type="BN", requires_grad=True),  # 采用哪一种归一化方式 这里采用batch normalization
        pretrained="/root/autodl-tmp/SparseDrive/ckpt/resnet50-19c8e357.pth",  # 加载在 ImageNet 数据集上预训练好的 ResNet-50 权重。
    ),
    img_neck=dict(
        type="FPN",  # 使用特征金字塔网络 接受来自img_backbone的多尺度特征图，通过自顶向下和横向连接，将高层语义信息和底层细节信息融合
        num_outs=num_levels,  # 4 输出4个尺度的融合后的特征图 检测不同大小物体
        start_level=0,  # 指定从in_channel列表的第几个元素开始构建FPN金字塔
        out_channels=embed_dims,  # 指定所有从FPN输出的特征图的通道数 全部统一为256 方便后续检测头的处理
        add_extra_convs="on_output",  # 扩展功能 用于生成比主干网络输出的最深层还要更深的特征图（例如 P6, P7） on_input表示在 FPN 输入的最深层特征图（C5）上应用一个步长为2的卷积来生成 P6 num_level为4 所以即使写了这个框架也不会生成这个额外的特征
        relu_before_extra_convs=True,  # 与add_extra_convs配合使用 它决定了在用于生成额外特征层（如P6）的那个卷积层之前，是否要先应用一个 ReLU 激活函数。
        in_channels=[256, 512, 1024, 2048],  # 定义了每一个特征图的通道数 精确对应img_backbone resnet-50中输出的4个阶段的特征图
    ),
    depth_branch=dict(  # for auxiliary supervision only  辅助任务 用于预测深度 它利用图像特征来预测像素级的深度信息。虽然它的输出（深度图）可能不直接用于最终的规划决策，但训练这个任务可以迫使 img_backbone 学习到更丰富的3D几何信息。这是一种多任务学习的技巧，通过增加一个相关的辅助任务来提升主任务的性能。
        type="DenseDepthNet",
        embed_dims=embed_dims,  # 输出的通道数
        num_depth_layers=num_depth_layers,  # 3层小网络
        loss_weight=0.2,  # 损失占比
    ),
    head=dict(  # 接收2D图像特征，并执行所有关键的自动驾驶任务
        type="SparseDriveHead",  # 总任务头 包含三个并行子任务：3D检测、高精地图构建、动态物体运动规划
        task_config=task_config,  # 用于定义哪些任务是激活的，这里是第一阶段训练，所以检测和地图是激活的，规划是False
        det_head=dict(  # 3D物体检测头
            type="Sparse4DHead",  # 4D 空间3D+时间1D
            cls_threshold_to_reg=0.05,  # 分类阈值 只有当一个查询预测某个类别的置信度高于 0.05 时，其对应的3D边界框回归结果才被认为是有效的或参与损失计算
            decouple_attn=decouple_attn,  # 是否在注意力模块中解耦处理不同的特征，anchor_encoder会用到，如果是耦合模式，那么anchor_encoder会为每一个anchor生成统一的完整的特征向量（包含完整信息比如位置速度尺寸等）；如果是解耦模式，anchor_encoder会为每一个anchor生成多个独立的专门的特征向量，比如一个向量专门表示位置、一个向量专门表示速度等，方便后续模块对不同信息进行差异化处理增加模型灵活性，这里是后者
            instance_bank=dict(  # 实例库配置 静态查询是用来发现每一帧中的新新物体的、动态查询是用来追踪前面一帧的物体的
                type="InstanceBank",
                num_anchor=900,  # 初始查询数量 每一帧开始时，固定初始化900个查询
                embed_dims=embed_dims,  # 通道数
                anchor="/root/autodl-tmp/SparseDrive/data/kmeans/kmeans_det_900.npy",  # 静态查询初始状态（3D位置、尺寸等）的来源文件，是通过对训练集中真实物体进行k-menas聚类得到的，代表了数据集中最常见的物体先验分布
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),  # 锚点生成器，用于解析从.npy文件中加载的锚点，进行初始变换
                num_temp_instances=600 if temporal else -1,  # 追踪查询数 从前一帧的检测结果中继承而来的，用于实现跨帧追踪
                confidence_decay=0.6,  # 一个衰减系数。如果一个时序查询在当前帧没有成功匹配到任何真实物体，它的置信度分数会乘以这个系数。分数过低后，该追踪查询会被丢弃。
                feat_grad=False,  # 是否让查询特征也参与梯度计算，减少显存占用与计算量
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",  # 把anchor（3D框）编码成查询
                vel_dims=3,  # 速度向量维度(vx, vy, vz)
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,  # 嵌入维度，如果耦合那么编码器最终输出单一256维的特征向量；如果解耦，编码器最终输出4个独立的特征向量，维度分别为128, 32, 32, 64，这四个向量共同构成一个完整的查询，可以被后面模块分开来处理
                mode="cat" if decouple_attn else "add",  # 特征融合方式。如果耦合那么逐元素相加，解耦就是拼接
                output_fc=not decouple_attn,  # 编码器最后是否加全连接层 耦合的话是True，解耦是False
                in_loops=1,  # 输入处理部分循环次数
                out_loops=4 if decouple_attn else 2,  # 输出处理部分循环次数，解耦为4，耦合为2
            ),
            num_single_frame_decoder=num_single_frame_decoder,  # 指定在引入时序信息（temp_gnn）之前，要执行多少个仅处理单帧信息的解码器层。
            operation_order=(  # 定义了解码器每一层内部操作的执行顺序 先稳定，后追踪
                [
                    "gnn",  # 空间图注意力模型，同一帧内所有查询之间的信息交互
                    "norm",  # 层归一化
                    "deformable",  # 可变形注意力模型，从多视角2D图像特征图中采样视觉特征
                    "ffn",  # 前馈神经网络
                    "norm",  # 层归一化
                    "refine",  # 从当前查询特征中解码出3D框和分类结果
                ]
                * num_single_frame_decoder
                + [
                    "temp_gnn",  # 时序空间图注意力
                    "gnn",  # 空间图注意力
                    "norm", # 层归一化
                    "deformable",  # 可变形注意力
                    "ffn",  # 前馈神经网络
                    "norm",  # 层归一化
                    "refine",  # 预测模块
                ]
                * (num_decoder - num_single_frame_decoder)
            )[2:], # 第一层解码器中，跳过gnn和norm，直接从deformable开始，因为刚刚初始化的空白的查询使用gnn和norm意义不大
            temp_graph_model=dict(  # 让当前帧的查询 (Query) 与其在前一帧的对应查询进行一次注意力计算，从而实现信息的跨时间流动。
                type="MultiheadFlashAttention",  # FlashAttention是一种对标准多头注意力机制的高度优化实现。它的核心优势在于极大地减少了显存占用并提升了计算速度。
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,  # 耦合模式就是256维，解耦模式把两个维度拼接起来
                num_heads=num_groups,  # 多头注意力中头的数量
                batch_first=True,  # 表示张量的维度顺序是 (批量大小, 序列长度, 特征维度)
                dropout=drop_out,  # 随机将某一些神经元输出变为0，防止过拟合
            )
            if temporal
            else None,
            graph_model=dict(  # 和前面一样的架构，就是调用时机不一样，注意力的内容也不一样
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),  # 层归一化 输入输出都是embed_dims维
            ffn=dict(
                type="AsymmetricFFN",  # 非对称前馈网络，非对称就是它的输入维度 (in_channels) 和输出维度 (embed_dims) 是不相同的
                in_channels=embed_dims * 2,  # 输入的维度 可变形注意力模块的输出，会与其输入（原始的查询特征）进行拼接 (Concatenate) 操作
                pre_norm=dict(type="LN"),  # 在FFN内部，FC前面，先应用层归一化
                embed_dims=embed_dims,  # 输出维度
                feedforward_channels=embed_dims * 4,  # 内部隐藏层维度
                num_fcs=2,  # 内部全连接层数量。第一个FC层将维度从 in_channels 扩张到 feedforward_channels，第二个FC层将维度从 feedforward_channels 压缩回 embed_dims。
                ffn_drop=drop_out,  # drop_out比率
                act_cfg=dict(type="ReLU", inplace=True),  # 激活函数配置
            ),  # 整体流程：输入（in_channels）-> 层归一化（pre_norm）-> 第一个FC层（维度扩张变为embed_dims*4）-> ReLu激活 -> Dropout -> 第二个FC层（维度压缩变回embed_dims）-> 输出（embed_dims）
            deformable_model=dict(
                type="DeformableFeatureAggregation",  # 连接3D世界中的查询（Query）和2D世界中的图像特征,让查询能够从多视角的摄像头图像中提取与自己相关的特征，来充实和更新自己。不是盲目地看整张图片，而是让每个3D查询学会“指明”它最想看的几个关键位置，然后只从这些位置提取信息，效率极高。
                embed_dims=embed_dims,  # 定义了输入查询（Query）和最终输出的聚合后特征的维度。整个模块中的特征维度保持一致。
                num_groups=num_groups,  # 多头注意力头数
                num_levels=num_levels,  # 指定要从 FPN 输出的多少个特征层级（Level）上进行采样。
                num_cams=6,  # 摄像头数量
                attn_drop=0.15,  # dropout比率
                use_deformable_func=use_deformable_func,  # 选择底层函数的实现，True表示使用CUDA C++，速度快但是可读性差需要编译；False表示使用纯pytorch实现相同逻辑，速度慢但是灵活无需额外编译
                use_camera_embed=True,  # 是否为每个相机学习位置编码 帮助模型区分特征来源
                residual_mode="cat",  # 如何将聚合到的新特征与原始的输入查询特征进行合并（即残差连接）
                kps_generator=dict(  # 3D关键点生成器，负责确定要去图像的哪些位置“看”
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,  # 除了下面定义的固定点外，额外让模型自己学习6个关键点的三维偏移量
                    fix_scale=[  # 定义一组固定的、预设的关键点。这些点的位置是相对于查询3D框的中心和尺寸来定义的。
                        [0, 0, 0],  # 3D框中心点
                        [0.45, 0, 0],  # 沿框X轴正方向移动45%长度的位置
                        [-0.45, 0, 0],  # 沿框X轴负方向移动45%长度的位置
                        [0, 0.45, 0],  # 沿框Y轴正方向移动45%长度的位置
                        [0, -0.45, 0],  # 沿框Y轴负方向移动45%长度的位置
                        [0, 0, 0.45],  # 沿框Z轴正方向移动45%长度的位置
                        [0, 0, -0.45],  # 沿框Z轴负方向移动45%长度的位置
                    ],  # 这7个点构成了3D框中心和三个主轴上的采样点，提供了一个稳定的基础结构。
                ),
            ),  # 整体流程：接受3D查询 -> 为3D框生成3D关键点（kps_generator，7个固定的+6个可学习的）-> 将这些3D关键点，利用相机内外参（在forward时传入），投影到所有6个相机的、所有num_levels个特征图上，得到一系列2D采样坐标 -> 模块在这些2D坐标上，通过双线性插值等方式，采样出对应的视觉特征 -> 采样到的特征被加权聚合（注意力权重也是网络学习的）成一个 embed_dims 维的特征向量 -> 聚合后的特征与原始查询特征按 "cat" 方式进行拼接（残差）-> 返回这个维度为 embed_dims * 2 的、包含了丰富视觉信息的新查询特征
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,  # 输入维度
                num_cls=num_classes,  # 总共有多少个物体类别
                refine_yaw=True,  # 是否对3D框的偏航角 (Yaw) 进行专门的、更精细的优化处理
                with_quality_estimation=with_quality_estimation,  # 是否要额外预测一个定位质量分数。除了标准的分类置信度（“这个框里是汽车的概率有多大？”），这个功能让模型额外预测一个分数来回答：“我预测的这个框，位置到底准不准？ 有助于抑制那些分类正确但定位极差的检测结果
            ),  # 流程：一般包含几个并行的头 分类头：输入 embed_dims -> 输出 num_classes (各类别分数)；回归头：输入 embed_dims -> 输出 N (3D框的位置、尺寸等参数的修正量)；质量头：输入 embed_dims -> 输出 1 (预测的IoU分数)
            sampler=dict(  # 在训练阶段，为所有的查询（Query）分配“学习目标”（即真值，Ground Truth），并生成用于计算最终损失函数的监督信号
                type="SparseBox3DTarget",
                num_dn_groups=0,  # 静态查询设置去噪组的数量
                num_temp_dn_groups=0,  # 时序查询设置去噪组的数量 都是0说明没有启用去噪任务
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,  # 添加噪声的尺度
                max_dn_gt=32,  # 一张图中最多使用32个真实物体来进行去噪任务
                add_neg_dn=True,  # 在去噪任务中，除了给真实物体加噪声，还会额外创造一些标签错误的“负样本”，让模型学会不仅要回归到正确的位置，还要分对正确的类别。
                cls_weight=2.0,  # 追踪任务的匹配成本 在计算总匹配成本时，分类预测的错误程度（例如用Focal Loss计算）所占的比重是2.0。
                box_weight=0.25,  # 在计算总匹配成本时，3D框预测的错误程度（例如用L1 Loss计算）所占的比重是0.25。
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,  # 对3D框的每个回归分量设置更精细的权重
                cls_wise_reg_weights={  # 按类别设定专属的回归权重 它意味着：“对于‘交通锥（traffic_cone）’这个类别，不要使用上面那个通用的 reg_weights，请使用我为它专门定义的这个新的权重列表。
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),
            loss_cls=dict(  # 分类误差
                type="FocalLoss",
                use_sigmoid=True,  # sigmoid处理分类输出logits，每一个类别的分数
                gamma=2.0,  # 聚焦参数 控制着对易分类样本权重抑制的强度 gamma 越大，对易分样本的抑制就越强，模型就越专注于难分样本
                alpha=0.25,  # 平衡参数 可以用来平衡正负样本本身的重要性
                loss_weight=2.0,  # 分类损失这一项在总损失中的权重
            ),
            loss_reg=dict(  # 预测的3D框与真实3D框之间的几何差距
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),  # 使用 L1 Loss (Mean Absolute Error, 平均绝对误差) 来计算预测的3D框参数（如中心点x,y,z，尺寸w,l,h）与真实值之间的差距
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),  # 辅助回归损失。预测的点离物体的真实中心有多近
                loss_yawness=dict(type="GaussianFocalLoss"),  # 辅助回归损失。预测的偏航角（Yaw）有多可靠
                cls_allow_reverse=[class_names.index("barrier")],  # 为特定类别设置一个特殊的、非常实用的规则 对于像“栅栏 (barrier)”这样的物体，它的物理形态是对称的。一个朝向为 yaw 的栅栏和一个朝向为 yaw + 180° 的栅栏在现实世界中是完全一样的
            ),
            decoder=dict(type="SparseBox3DDecoder"),  # 最终的、用于坐标变换的输出模块 负责将网络内部的、相对的、迭代式的预测，最终转换为我们需要的、绝对的、易于理解的3D边界框坐标
            reg_weights=[2.0] * 3 + [1.0] * 7,  # 回归权重，每一项不同
        ),
        map_head=dict(  # 输出一组矢量化的地图元素，通常是由一系列3D点构成的折线（Polyline），来精确地描述道路的几何结构
            type="Sparse4DHead",  # 4D 空间3D+时间1D
            cls_threshold_to_reg=0.05,  # 一个查询所预测的地图元素（例如一条车道线）只有在其对应类别的置信度分数高于 0.05 时，其回归结果（即预测的线段坐标）才被认为是有效的，或才参与到回归损失的计算中
            decouple_attn=decouple_attn_map,  # 是否解耦注意力
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,  # 只有静态查询 每一帧都会固定初始化100个用于寻找地图元素的查询。
                embed_dims=embed_dims,
                anchor="/root/autodl-tmp/SparseDrive/data/kmeans/kmeans_map_100.npy",  # 存储的是3D空间中的点或短线段的几何先验
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),  # 处理点线
                num_temp_instances=0 if temporal_map else -1,  # 不采用追踪 没有动态查询
                confidence_decay=0.6,  # 不会产生实际作用 因为不采用动态查询
                feat_grad=True,  # 是否允许梯度流过查询的特征
            ),
            anchor_encoder=dict(  # 从.npy文件中加载anchor（一个anchor由num_sample个点组成），然后经过网络输出特征（embed_dim维）
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,  # 定义该编码器输出的特征向量维度
                num_sample=num_sample,  # 每一个anchor由num_sample个点组成
            ),
            num_single_frame_decoder=num_single_frame_decoder_map, # 不考虑时间只考虑空间的重复次数
            operation_order=(
                [
                    "gnn",  # 和det_head一样
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder_map
                + [
                    "temp_gnn",  # 和det_head一样
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder_map)
            )[:],  # 这里没有去掉gnn和norm这两层！
            temp_graph_model=dict(  # 和det_head一样
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal_map
            else None,
            graph_model=dict(  # 和det_head一样
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),  # 和det_head一样
            ffn=dict(  # 和det_head一样
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(  # 这里和det_head不一样
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,  # 额外的3个可学习的偏移量
                    fix_height=(0, 0.5, -0.5, 1, -1),  # 固定偏移量
                    ground_height=-1.84023, # ground height in lidar frame
                ),
            ),  # 有一个查询->沿着折线每一个点在垂直方向上有一个偏移->之后3D位置投影到2D图像中提取相关特征->特征聚合起来和原始输入特征拼接起来给后面的模块
            refine_layer=dict(  # 和det_head不一样
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,  # 输入维度
                num_sample=num_sample,  # 回归头 一个地图元素包含多少个点（物体检测输出的是物体的位置长宽高速度转向，那么类比过来这里输出的就是numsample个点）
                num_cls=num_map_classes,  # 分类头 类别
            ),
            sampler=dict(  # 和det_head不一样 将代表3D折线的查询，与真实的3D折线进行匹配
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),  # 衡量预测的类别（如“车道线”）和真值的类别是否一致
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),  # 衡量预测的线的几何形状与真值的线的几何形状的相似度
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,  # 定义了一个感兴趣区域（Region of Interest）的大小。匹配和损失计算可能只在这个区域内进行
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(
                    type='LinesL1Loss',
                    loss_weight=10.0,
                    beta=0.01,
                ),
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id",
            with_instance_id=False,
            task_prefix='map',
        ),
        motion_plan_head=dict(  # 预测与规划
            type='MotionPlanningHead',
            fut_ts=fut_ts,  # 要预测的未来时间步数量（例如，预测未来6秒，每秒2个点，则为12）
            fut_mode=fut_mode,  # 预测fut_mode种最可能的轨迹，一个物体未来会有多种可能轨迹，挑选出fut_mode种可能轨迹里面概率最高的
            ego_fut_ts=ego_fut_ts,  # 自车要预测的未来时间步数量
            ego_fut_mode=ego_fut_mode,  # 自车要预测的最可能的轨迹数量
            motion_anchor=f'/root/autodl-tmp/SparseDrive/data/kmeans/kmeans_motion_{fut_mode}.npy',  # 为预测任务提供初始锚点 .npy 文件里存储的是通过K-Means聚类从数据集中提取的最常见的几种运动轨迹
            plan_anchor=f'/root/autodl-tmp/SparseDrive/data/kmeans/kmeans_plan_{ego_fut_mode}.npy',  # 为规划任务提供初始锚点 .npy 文件里存储的是通过K-Means聚类从数据集中提取的最常见的几种规划路径
            embed_dims=embed_dims,  # 通道数
            decouple_attn=decouple_attn_motion,  # 是否解耦注意力
            instance_queue=dict(  # 历史状态队列
                type="InstanceQueue",
                embed_dims=embed_dims,  # 存储的特征向量的维度
                queue_length=queue_length,  # 要保存多少个历史时间步的物体状态（包括位置、特征等）
                tracking_threshold=0.2,  # 将新一帧的检测结果与队列中已存在的追踪轨迹进行匹配 每一新帧，det_head会输出新的检测结果，同时instance_queue也会根据历史预测出当前帧可能位置，之后会计算新的检测结果与每个轨迹预测的位置之间的相似度，如果相似度大于这个tracking_threshold，就认为成功匹配，新的检测结果加入到队列中，否则认为新的检测结果没有匹配任何现有轨迹，用来初始化新的追踪轨迹
                feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),  # 最深层特征图的宽度和高度 主要用于坐标系转换
            ),
            operation_order=(
                [  # 重复三次
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "cross_gnn",  # 新增的跨模态图注意力
                    "norm",
                    "ffn",                    
                    "norm",
                ] * 3 +
                [
                    "refine",  # 预测与规划两个头
                ]
            ),
            temp_graph_model=dict(  # 和前面一样 是当前帧的查询和自己过去状态的关注，用于理解自身的动态变化
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            graph_model=dict(  # 和前面一样 是当前帧的查询和当前帧的所有其他查询的关注，用于理解当前场景中各个实体之间的相互关系
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            cross_graph_model=dict(  # 新增的内容 是当前帧和前面地图特征/物体特征的关注 用于融合感知信息以供决策
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),  # 和前面一样
            ffn=dict(  # 和前面一样
                type="AsymmetricFFN",
                in_channels=embed_dims,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 2,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            refine_layer=dict(  # 用于预测+规划
                type="MotionPlanningRefinementModule",
                embed_dims=embed_dims,  # 输入的向量维度
                fut_ts=fut_ts,  # 预测多少个未来时间步
                fut_mode=fut_mode,  # 预测多少种未来可能轨迹
                ego_fut_ts=ego_fut_ts,  # 预测ego车辆多少个未来时间步
                ego_fut_mode=ego_fut_mode,  # 预测ego车辆多少种未来可能轨迹
            ),
            motion_sampler=dict(  # 接受模型预测出的所有fut_mode种预测轨迹，接受ground truth的轨迹，根据这条ground truth轨迹，在模型预测出的轨迹里面选择一条和这个ground truth最接近的，认定为正样本，后续计算损失就是基于这条轨迹计算的
                type="MotionTarget",
            ),
            motion_loss_cls=dict(  # 输出fut_mode种轨迹每一条轨迹的概率
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2
            ),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),  # 选择出的轨迹和真实轨迹的误差
            planning_sampler=dict(  # 一样的
                type="PlanningTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
            ),
            plan_loss_cls=dict(  # 一样的
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,
            ),
            plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),  # 选择出的轨迹和真实轨迹的误差
            plan_loss_status=dict(type='L1Loss', loss_weight=1.0),  # 状态预测 比如速度加速度等
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),  # 输出最终的各个轨迹坐标点以及置信度
            planning_decoder=dict(  # 分层规划解码器 根据输出的轨迹坐标以及概率，参考high-level的导航信息+碰撞检测过滤掉不合适的轨迹 在最终轨迹中选择出概率最高的作为最终输出
                type="HierarchicalPlanningDecoder",
                ego_fut_ts=ego_fut_ts,
                ego_fut_mode=ego_fut_mode,
                use_rescore=True,
            ),
            num_det=50,  # cross_gnn中每一个查询关注50个物体特征
            num_map=10,  # cross_gnn中每一个查询关注10个地图特征
        ),
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDataset"  # 数据集类型
data_root = "/root/autodl-tmp/SparseDrive/data/nuscenes/"  # 数据集根目录
anno_root = "/root/autodl-tmp/SparseDrive/data/infos/" if version == 'trainval' else "data/infos/mini/"  # 预处理后的标注文件所在的目录
file_client_args = dict(backend="disk")  # 配置文件读取方式 disk表示本地磁盘 也即存储在服务器硬盘上

img_norm_cfg = dict(  # 图像归一化配置定义 原始图像减去均值除以方差 对像素值进行归一化使其符合标准分布
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True  # 这里的均值和方差是在ImageNet数据集上计算出的所有图像的RGB通道均值
)
train_pipeline = [  # 处理原始图像
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),  # 从文件中加载多视角图像 根据标注信息中提供的文件路径，读取6个摄像头的图像文件，并将像素值从整数（0-255）转换为32位浮点数
    dict(  # 从文件中加载点云数据 读取对应的激光雷达（LiDAR）点云文件
        type="LoadPointsFromFile",
        coord_type="LIDAR",  # 指定坐标系是LIDAR坐标系
        load_dim=5,  # 加载点云的前5个维度(x, y, z, intensity, ring_index)并全部使用
        use_dim=5,
        file_client_args=file_client_args, # 文件读取方式
    ),
    dict(type="ResizeCropFlipImage"),  # 对图像进行随机的尺寸缩放、裁剪和水平翻转
    dict(
        type="MultiScaleDepthMapGenerator",  # 生成多尺度深度图 为模型的 depth_branch（深度预测辅助任务）准备真值。它将加载的点云投影到（经过增强的）图像上，来生成像素级的深度图
        downsample=strides[:num_depth_layers],  # 指定了要生成哪些下采样尺度的深度图，以匹配 img_neck 输出的多层特征图
    ),
    dict(type="BBoxRotation"),  # 对3D场景进行随机旋转 将整个场景（包括点云和所有的3D边界框）围绕Z轴进行一个随机角度的旋转。这能教会模型识别不同朝向的物体
    dict(type="PhotoMetricDistortionMultiViewImage"),  # 对图像进行光度学扭曲 随机地改变图像的亮度、对比度、饱和度和色调，以模拟不同的光照和天气条件，增强模型的鲁棒性
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),  # 归一化多视角图像 对图像像素值进行标准化处理
    dict(  # 基于距离的物体范围过滤器 移除那些距离自车太远的标注物体
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),  # 对于所有类别的物体，只保留55米范围内的进行训练。这有助于模型专注于最相关的感知区域。
    ),
    dict(type="InstanceNameFilter", classes=class_names),  # 基于类别的实例过滤器 只保留那些类别在 class_names 列表中的物体。例如，如果 class_names 不包含“拖车”，那么数据集中所有的“拖车”标注都会被过滤掉
    dict(
        type='VectorizeMap',  # 矢量化地图 为 map_head 准备真值的关键步骤。它将原始的地图标注（可能格式复杂）转换成统一的、由点构成的折线（Polyline）列表
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,  # 每一条折线都被重采样到固定点数
        permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),  # 针对 Sparse4D 架构的专用数据适配器 一些格式对齐上的工作
    dict(  # 最后一步 负责打包
        type="Collect",  # 从这个大字典里挑出模型训练真正需要的那些项目，把它们打包成最终的输出
        keys=[  # 这个列表里的键对应的值，是模型训练所需的核心数据 它们会被取出，堆叠（stack）成一个批次（batch），并转换为PyTorch的Tensor，最终被送到GPU上
            "img", # 图像
            "timestamp",  # 时间戳
            "projection_mat",  # 投影矩阵 将在3D空间中的点，投影到2D的相机图像平面上 deformable_model需要这个操作
            "image_wh",  # 图像长宽
            "gt_depth",  # 深度真值
            "focal",  # 焦距 相机内参的一个关键参数，决定了相机的“视野”大小
            "gt_bboxes_3d",  # 检测框真值
            "gt_labels_3d",  # 检测类别标签真值
            'gt_map_labels',  # 地图类别真值
            'gt_map_pts',  # 地图点真值
            'gt_agent_fut_trajs',  # 物体未来轨迹真值
            'gt_agent_fut_masks',  # 标记未来轨迹的每一个时间步，其对应的真值是否有效 无效的话计算损失的时候这个时间步会被忽略
            'gt_ego_fut_trajs',  # 自车未来轨迹真值
            'gt_ego_fut_masks',  # 标记自车未来轨迹的每一个时间步
            'gt_ego_fut_cmd',  # 自车导航信息真值
            'ego_status',  # 自车状态真值
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],  # 元数据 也会被保留在最终的数据包里，但不会被转换成Tensor或送到GPU。它们通常用于调试、可视化或一些需要CPU进行的计算
    ),
]
test_pipeline = [  # 测试流水线不会包含数据增强模块 也不会加载真实值
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),  # 相同 加载多视角图像
    dict(type="ResizeCropFlipImage"),  # 只执行确定性的尺寸调整（Resize）和裁剪（Crop），以确保输入尺寸符合模型要求，而会禁用随机翻转（Flip）
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),  # 相同 归一化多视角图像 模型在哪个分布上训练的，就必须在哪个分布上测试。
    dict(type="NuScenesSparse4DAdaptor"),  # 相同 整理数据格式以适配模型架构，这个步骤在测试时同样需要
    dict(  # 最终打包 真值都没有了
        type="Collect",
        keys=[
            "img",  # 图像
            "timestamp",  # 时间戳
            "projection_mat",  # 投影
            "image_wh",  # 长宽
            'ego_status',  # 自车状态
            'gt_ego_fut_cmd',  # 导航信息
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],  # 相同 打包一些无需送入GPU的元数据
    ),
]
eval_pipeline = [  # 评估 所以都是真值
    dict(  # 根据物体距离自车的位置，对**真值（Ground Truth）**列表进行过滤。
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),  # 再次对**真值（Ground Truth）**列表进行过滤，只保留那些类别在项目定义的 class_names 列表中的物体
    dict(  # 矢量化处理地图
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=True,
        normalize=False,
    ),
    dict(  # 最终打包
        type='Collect',   # 这个 keys 列表里没有任何传感器数据（如 'img'），全是 gt_（Ground Truth）或处理后的真值，这再次证明了它的作用是准备“标准答案
        keys=[
            'vectors',  # 矢量化地图真值
            "gt_bboxes_3d",  # 检测框真值
            "gt_labels_3d",  # 检测类别真值
            'gt_agent_fut_trajs',  # 未来轨迹真值
            'gt_agent_fut_masks',  # 未来轨迹有效性
            'gt_ego_fut_trajs',  # 未来自车轨迹真值
            'gt_ego_fut_masks',   # 未来自车轨迹有效性
            'gt_ego_fut_cmd',  # 导航信息
            'fut_boxes'  # 其他物体在未来时间步上的3D边界框真值，用于评估运动预测的精度。
        ],
        meta_keys=['token', 'timestamp']  # 元数据 评估脚本会用这个 token 来确保模型对某一帧的预测结果与 eval_pipeline 处理出的该帧的标准答案能够准确地匹配上，然后进行一对一的比较。
    ),
]

input_modality = dict(  # 是纯视觉输入 没有雷达数据
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(  # 数据集基本配置 数据根目录 类别 输入模态等
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    version="v1.0-trainval",
)
eval_config = dict(  # 在进行模型评估（Validation）时，传给数据加载器（DataLoader）的总配置文件
    **data_basic_config,  # 继承基础数据配置
    ann_file=anno_root + 'nuscenes_infos_val.pkl',  # 指定验证集标注文件
    pipeline=eval_pipeline,  # 指定使用的处理流水线
    test_mode=True,  # 开启测试模式 比如禁用所有随机操作，只进行确定性的尺寸调整
)
data_aug_conf = {  # 数据增强参数配置
    "resize_lim": (0.40, 0.47),  # 对输入图像进行随机缩放的比例范围
    "final_dim": input_shape[::-1],  # 送入网络的图像的最终分辨率 input_shape 可能是一个 (H, W) 的元组，[::-1] 的操作会将其反转为 (W, H)
    "bot_pct_lim": (0.0, 0.0),  # 定义了随机裁剪时，允许从图像底部裁掉部分的百分比范围 不允许从图像底部进行任何裁剪。这在自动驾驶场景中非常合理，因为靠近车头的路面信息（图像底部）通常是最重要的
    "rot_lim": (-5.4, 5.4),  # 定义了对图像进行2D平面内随机旋转的角度范围（单位是度）。图像会被随机旋转 -5.4 到 +5.4 度之间的一个角度
    "H": 900,  # 定义输入图像的原始长宽
    "W": 1600,
    "rand_flip": True,  # 启用了随机水平翻转这一数据增强策略
    "rot3d_range": [0, 0],  # 定义了对整个3D场景（点云和3D框）进行随机旋转（通常是绕Z轴）的角度范围 被禁用了
}

data = dict(  # 定义了所有与数据加载相关的配置，框架会根据这个字典来创建相应的数据加载器（DataLoader）实例。
    samples_per_gpu=batch_size,  # 每个GPU处理的样本数量
    workers_per_gpu=batch_size,  # 每个GPU分配的数据加载工作进程数
    train=dict(
        **data_basic_config,  # 通用基础配置
        ann_file=anno_root + "nuscenes_infos_train.pkl",  # 标注文件
        pipeline=train_pipeline,  # 数据处理管线
        test_mode=False,  # 随机性数据增强
        data_aug_conf=data_aug_conf,  # 传递数据增强超参数
        with_seq_flag=True,  # 启用序列数据加载模式。这是一个关键的时序参数。设置为 True 后，数据加载器不会再随机抽取单帧，而是会加载连续的视频帧序列。
        sequences_split_num=2,  # 定义了如何从数据集中切分或采样序列
        keep_consistent_seq_aug=True,  # 保持序列增强的一致性。这个参数至关重要。它确保了对于一个加载进来的帧序列，所有的数据增强操作（如随机翻转、旋转）对序列中的每一帧都是完全相同的。如果每一帧的增强不一致（例如第一帧翻转了而第二帧没有），那么整个场景的几何关系就会错乱，模型将无法学习到正确的时序信息
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,  # 这里也用的test_pipeline!
        data_aug_conf=data_aug_conf,
        test_mode=True,  # 禁用随机性
        eval_config=eval_config,
    ),
    test=dict(  # 和val一模一样
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
)  # 注意，不是eval_pipeline就对应eval，test_pipeline就对应test，eval_pipeline相当于是正确答案，test_pipeline是给模型准备考卷（测试数据），这两个功能不一样 模型预测出来结果之后会和eval_pipeline中的真值比较 实现评估

# ================== training ========================
optimizer = dict(
    type="AdamW",  # 优化器类型
    lr=4e-4,  # 学习率
    weight_decay=0.001,  # 权重衰减系数
    paramwise_cfg=dict(  # 精细化学习率配置
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))  # 梯度裁剪 防止梯度爆炸
lr_config = dict(  # 学习率调度器
    policy="CosineAnnealing",  # 余弦退火
    warmup="linear",
    warmup_iters=500,  # 从一个很小的值线性增加到基础学习率
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,  # 学习率最小值
)
runner = dict(  # 一整个训练流程的主循环
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,  # 设置总的训练迭代次数
)

# ================== eval ========================
eval_mode = dict(
    with_det=True,
    with_tracking=True,
    with_map=False,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,  # 在计算追踪指标时，用于判断一个预测框和一个真值框是否成功匹配的IoU（交并比）阈值。只有IoU大于0.2才算匹配成功。
    motion_threshhold=0.2,  # 在评估预测轨迹时使用的一个阈值，例如，可能指预测的轨迹终点与真实终点之间的距离要小于0.2米才算一个“命中”。
)
evaluation = dict(  # 什么时候评估
    interval=num_iters_per_epoch*checkpoint_epoch_interval,  # 每多少次迭代就评估一次
    eval_mode=eval_mode,  # 评估的具体内容
)
