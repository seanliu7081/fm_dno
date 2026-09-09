# FM-DNO 项目长期上下文

最后核对日期：**2026-09-09（UTC）**。本文用于跨对话交接、实验规划和长期记录，按 **First：heading predictor → Second：三类主策略 → Third：DNO** 的顺序解释设计演进。

本文区分设计、已实现功能、已验证结果和后续假设。运行状态及 checkpoint 可用性是有日期的快照；开启新实验前应再次查看实际文件。历史结果保留原来的模型身份，不随新权重替换。

## 0. 研究目标与固定实验背景

核心问题：**从观测预测未来动作方向，并将其用于条件或噪声先验，能否改善条件动作生成？在执行阶段进一步优化初始噪声，能否提高任务成功率与泛化能力？**

主策略是连续动作 **flow matching：Transformer 速度场 + Euler 采样**。本研究不依赖动作 tokenizer。仓库还包含其他策略和 flow backbone；对方向机制的第一轮比较固定 backbone，避免同时改变多个因素。其他 backbone 见 [flow_backbones.md](flow_backbones.md)。

| 项目 | 固定设置 / 约定 |
| --- | --- |
| 仓库 | `/home/haotian/code/fm_dno`；命令从仓库根目录执行 |
| Python | `conda activate oat`；解释器 `/home/haotian/miniforge3/envs/oat/bin/python` |
| GPU | 两张 RTX 4090，系统编号只有 0、1 |
| GPU 映射 | `CUDA_VISIBLE_DEVICES=1` 选择系统 GPU 1；程序内部为 `cuda:0` |
| 数据 | `data/libero/libero10_N500.zarr`，10 个任务、500 条示范 episode |
| 划分 | 450 条训练、50 条验证；固定 split seed 42；方向预测器和主策略使用相同划分 |
| 观测 | 外部相机、腕部相机、机器人状态；最近 2 帧 |
| 状态 | 末端位置 3、四元数 4、夹爪状态 2、task_uid 1，共 10 维 |
| 动作 | 每步 7 维；预测 16 步，通常执行前 8 步，再接收新观测 |
| 生成器 | 默认 Transformer：embed_dim 256、4 层、4 heads；Euler 10 步 |
| 主策略训练 | 默认 batch size 32、5001 epochs、EMA；默认值不代表已完成训练 |
| 归一化 | 动作 `normalizer_mode=so2_block`，`normalizer_vector_mode=rms` |

优化 seed 与数据划分 seed 是不同变量。多 seed 实验可以改变训练 seed，但保持参考训练的 `split_seed=42` 与策略的 `task.policy.dataset.seed=42`。归一化统计只拟合训练 episodes。

## 1. First：Heading predictor

### 1.1 为什么引入学习的方向参考

早期 [flow_policy_canon.py](../oat/policy/flow_policy_canon.py) 直接从最近两帧末端位置差计算运动方向。它没有学习模块；静止启动时方向可能无定义，转弯时近期运动也不一定代表未来动作。

因此引入观测到未来动作块方向的预测器。方向是由现有观测推导出的结构化特征，并未增加新的传感器信息。是否帮助生成，需要后续闭环实验验证。

### 1.2 网络与输入

实现：[heading_predictor.py](../oat/perception/heading_predictor.py)。默认 `reference_mode=image_state`：

```text
外部相机每帧 → 独立 ResNet18Conv + SpatialSoftmax → 64 维
腕部相机每帧 → 独立 ResNet18Conv + SpatialSoftmax → 64 维
机器人状态   → 归一化                         → 10 维
                        每帧拼接：138 维
                        两帧展平：276 维
                  Linear(276,128) → SiLU → Linear(128,3)
```

前两个输出归一化得到单位方向，`atan2` 得到角度；第三个输出经过 sigmoid：

\[
\hat d(o)=(\cos\hat\theta,\sin\hat\theta),\qquad
\hat p_v(o)=\sigma(\ell_v).
\]

`reference_mode=state` 去掉两个图像分支，方向头输入变成两帧状态：`20 → 128 → 3`。这只改变参考预测器；主策略仍然接收图像和状态。

### 1.3 标签、损失与 confidence 的含义

监督标签来自未来示范动作的 `chunk_heading(action)`，具体在 SO(2) 兼容归一化之后计算。默认 `action_spec` 为：

```yaml
vector_blocks: [[0, 1], [3, 4]]
scalar_dims: [2, 5, 6]
block_weights: [1.0, 0.0]
```

因此方向标签默认只使用平移 XY 的合成方向；不是物体朝向，也不是单步末端朝向。标签覆盖预测的 16 步动作块，尚未改为仅执行的 8 步。

合成方向强度低于 `min_target_confidence=0.05` 时标签无效。损失为有效标签上的 masked cosine loss，加上所有样本上的有效性 BCE：

\[
L_{\rm ref}=
\frac{\sum_i m_i(1-\hat d_i^\top d_i)}{\max(\sum_i m_i,1)}
+\operatorname{BCEWithLogits}(\ell_v,m).
\]

**`confidence` 表示方向标签有效的概率，不是角度预测正确的概率，也不是角度不确定性。** 标签方向很明确但模型预测错角度时，confidence 仍可能很高。不能用其门限通过率宣称筛除了角度错误。

未来动作只用于离线监督。构造策略的方向 condition 或 source 时，训练和推理均使用预测结果，不能用真实动作方向替代。

### 1.4 已完成的参考预训练

实际输出目录是 `output/heading_reference/image_state/`，**路径中没有 `seed42` 子目录**。其他文档中的带 seed 子目录路径只是命令模板，不能套用到此 artifact。

| 记录 | 数值 |
| --- | --- |
| 训练轮数 | 30 epochs，编号 0–29 |
| 已保存最佳权重 | `output/heading_reference/image_state/checkpoints/best.pt` |
| best.pt 选择规则 | 最小验证总 loss；选中 epoch 18 |
| epoch 18 验证方向 MAE / 平均余弦 | 24.8477° / 0.82278 |
| epoch 18 有效性门限通过率 | 99.7163% |
| 全程最低方向 MAE | epoch 23：24.4863°；没有单独保存该轮权重 |
| 最后一轮方向 MAE | epoch 29：25.1852° |
| 固定 episode 划分证据 | `output/heading_reference/image_state/split.json` |
| 完整逐轮指标 | `output/heading_reference/image_state/metrics.json` |

后十轮角度指标基本进入平台。当前先推进主策略和 DNO，未仅为增加轮数延长预训练。**尚无确认的匹配 state-only 结果，不能断言图像优于仅状态。**

## 2. Second：三类方向引导主策略

### 2.1 共同生成过程与冻结边界

三类策略共用 [flow_policy_learned_canon.py](../oat/policy/flow_policy_learned_canon.py)，类名 `LearnedCanonicalPhaseFlowPolicy`。设策略观测特征为 \(h_\phi(o)\)，生成器为 \(G_\phi(c,z)\)。

训练时，用相应 source \(z\) 与归一化示范动作 \(a\) 构造 flow matching 插值及速度目标；推理时，从相同规则构造的 source 开始，进行 10 步 Euler 积分得到动作。使用 condition / source 的方式在训练和推理间保持一致。

| 模块 | 主策略离线训练 | DNO / 部署 |
| --- | --- | --- |
| 参考图像/状态 encoder、方向头、参考归一化 | 全部冻结、eval 模式 | 全部冻结 |
| 主策略自己的图像/状态 encoder、flow 网络 | 正常训练 | 全部冻结 |
| 每轮噪声 z | 根据对应 source 构造 | DNO 可优化 |
| 新增噪声初始化器 | 与基础策略训练分开 | 当前通过 teacher 蒸馏训练；部署时冻结 |

参考预处理在主策略训练和部署期间保持一致，使用固定的参考评估预处理。`prepare_for_training()` 在 workspace 恢复 checkpoint **之后**、DDP 初始化之前，加载尚未初始化的参考模块。完整参考模型及其归一化内嵌于策略 checkpoint；已初始化的 checkpoint 恢复不依赖原独立 `best.pt`。

### 2.2 版本 1：Condition only — E

配置：`train_flowpolicy_heading_condition`；`policy.reference_use=condition`。

\[
c_E(o)=\left[h_\phi(o),\cos\hat\theta(o),\sin\hat\theta(o),\hat p_v(o)\right],
\qquad z_E\sim\mathcal N(0,I).
\]

三个参考量拼接到每帧观测特征，两个观测 token 共用该窗口预测出的参考。当前 encoder 每帧由 138 维变成 141 维。

虽然属于 canonical policy 研究系列，**E 实际只增加方向 condition，噪声仍为 IID，没有 source canonicalization**。它检验显式方向特征能否帮助网络使用观测中的几何信息，是当前优先训练的主方案。

### 2.3 版本 2：Noise prior only — D

配置：`train_flowpolicy_learned_canon`；`policy.reference_use=source`。

主策略 condition 保持 \(h_\phi(o)\)，不额外拼接参考方向。预测方向用于构造观测相关的 source：

\[
\epsilon\sim\mathcal N(0,I),\quad \delta\sim\operatorname{vonMises}(0,\kappa),
\qquad
z_D=\rho\!\left(\hat\theta(o)+\delta-\theta(\epsilon)\right)\epsilon.
\]

默认 `prior_noise_scale=1`、`kappa=4`、`canon_prob=1`。\(\rho\) 旋转配置的二维 action blocks，其余 scalar 维不变。有限 kappa 保留角度随机性；无穷大是精确对齐的消融。

必须减去噪声自身方向 \(\theta(\epsilon)\)：仅按观测角度旋转各向同性高斯，不会改变它的分布。SO(2) 分块归一化确保这些角度与旋转操作相容。

方向/有效性预测必须有限，且有效性概率达到默认门限 0.5 才应用 source 对齐，否则跳过。这个门控不提供角度准确度保证。

D 使用 image+state 参考。匹配的 state-only prior 版本是 **C**，配置 `train_flowpolicy_learned_canon_state`。D/C 的主策略自身都继续接收图像和状态。

### 2.4 版本 3：Condition + noise prior — F

配置：`train_flowpolicy_heading_both`；`policy.reference_use=both`。

\[
c_F(o)=c_E(o),\qquad z_F=z_D.
\]

F 同时提供显式方向 condition 和观测相关 source。**E vs F** 是主要比较：在网络已经获得方向 condition 后，改变初始噪声是否仍带来收益？

F 必须按其 source 构造规则独立训练。不能把训练好的 E checkpoint 改一个开关就当成已训练 F。Canonicalization 也不保证“噪声旋转多少，输出动作就旋转多少”，该响应需要实际测量。

### 2.5 保留 A–F 实验命名

本文按用户要求先讲 E、D、F，但沿用历史字母，避免后续日志含义改变。

| 组别 | config-name | 方向参考 | 额外方向 condition | 对齐 source |
| --- | --- | --- | --- | --- |
| A | `train_flowpolicy_canon_iid` | 无 | 否 | 否 |
| B | `train_flowpolicy_canon_soft` | 近期末端运动 | 否 | 是 |
| C | `train_flowpolicy_learned_canon_state` | 冻结 state-only 预测器 | 否 | 是 |
| D | `train_flowpolicy_learned_canon` | 冻结 image+state 预测器 | 否 | 是 |
| E | `train_flowpolicy_heading_condition` | 同 D | 是 | 否 |
| F | `train_flowpolicy_heading_both` | 同 D | 是 | 是 |

A 保持相同的 SO(2) 动作归一化。B/C/D/F 第一轮比较固定相同 kappa。C vs D 检验参考图像价值；E vs F 检验额外 source 的价值；D 是区分 condition 与 prior 效应的重要消融。

“E 是稳妥主方案、F 是主要 canonicalization 候选”是实验优先级，并非已经证明的 E/F 成功率排序。

## 3. Third：将 DNO 接入 E/F

### 3.1 为什么需要执行阶段的新设计

每执行最多 8 步动作，环境会产生新观测。目标物体可能处于不同位置，也可能发生抓取失败、滑动或状态偏差。方向参考提供归纳偏置，但本身并不评价候选动作能否完成当前任务。

当前 DNO 的定义是：**在冻结 E/F 生成器下，依据当前任务评分选择或优化完整初始噪声，从而改变这一轮生成的动作。** 本实现独立于旧 OrbitDNO，核心文件为 [task_noise_policy.py](../oat/policy/task_noise_policy.py)。

OOD 目标位姿和新物体类别是研究目标，当前普通 reset-seed 试验尚未实现这种评估。训练苹果、测试球是否算“域内”，必须由训练物体集合和测试协议定义；未见过的类别不能仅因都是 pick-place 就标为域内。

### 3.2 每一轮如何运行

1. 接收真实执行后的最新两帧观测，重新计算参考方向和主策略 condition。
2. 为各候选构造一次原始 source：E 为 IID；F 为参考方向 canonical source，包括本轮固定的 angular dither。
3. 缓存本轮 condition 和任务上下文；在同一次优化中固定它们。
4. 通过完整 Euler 采样器生成候选动作，评分并选择，或对完整噪声做梯度更新。
5. 执行获选动作的前缀，逐步检查环境成功和结束状态；通常最多 8 步。
6. 接收新观测后重新规划。未执行的动作后缀不能当作已发生的 transition。

F 的 source 只在优化前 canonicalize 一次。优化过程中不重复对齐、不重采角度扰动。当前优化所有 `16 × 7` 噪声维度，不限于一个整体旋转角；它不直接优化 heading condition。

### 3.3 梯度 DNO 的目标与约束

设本轮原始候选噪声为 \(z_0\)，task context 为 \(u_t\)，归一化反变换后的动作序列为 \(A(z)\)：

\[
\min_z J_{\rm task}(A(z),u_t)
+\lambda\operatorname{mean}\left[\left(\frac{z-z_0}{\sigma}\right)^2\right],
\qquad
\operatorname{RMS}(z-z_0)\leq r\sigma.
\]

当前实现为 Adam、默认学习率 0.05、trust_weight 0.01、trust_radius 0.5、逐样本梯度裁剪；默认比较预设使用 4 个候选、3 次更新。保留原始候选和搜索过程中任务 proxy 最优的有限候选，避免返回非有限动作。

梯度穿过全部 Euler 步，但基础网络权重冻结。DNO 可在外层 `no_grad` 下局部启用梯度；梯度模式不能包在 `torch.inference_mode()` 内。

保留原始候选只保证返回结果的 **proxy 分数不更差**，不保证真实任务成功率更高。噪声约束限制偏离训练 source 的程度，但也不构成分布内或稳定性的保证。

### 3.4 五种可运行模式

| `--modes` | 方法 | 更新对象 | 是否从跨回合数据学习 |
| --- | --- | --- | --- |
| `baseline` | 原 source 单次采样 | 无 | 否 |
| `best_of_k` | 独立采样多个候选后重排 | 当前候选选择 | 否 |
| `dno` | 候选噪声梯度优化 | 当前 z | 默认否；可导出 teacher 数据 |
| `amortized` | 学习初始化器提出噪声修正，与原始候选比较 | 离线训练 initializer 参数 | 是，teacher 蒸馏 |
| `amortized_dno` | 学习初始化后继续 DNO | 离线 initializer 参数 + 当前 z | 是，teacher 蒸馏 |

`amortized` 当前仍生成原始和修正两个候选，并用 scorer 选择；不是完全摆脱 scorer 的单次 actor 推理。

三种不同的“更新策略”必须分开记录：当前 z 的优化改变本轮行为；训练 initializer 将经验保存在新模块；微调基础 flow 权重则是另一种策略训练。**当前没有实现 actor–critic RL，也没有已训练的成功率 critic。**

### 3.5 第一版任务评分：模拟器几何 oracle

实现：[task_objectives.py](../oat/dno/task_objectives.py)、[dno_context.py](../oat/env/libero/dno_context.py)。

从 live simulator 读取 BDDL 目标、物体/目标区域位置、真实双指抓取接触和 OSC 控制器缩放。依次使用 reach、grasp、lift、transport、release 阶段；阶段依据真实环境反馈更新，同次噪声优化期间不变化。

评分使用配置的 `n_action_steps` 前缀（默认 8 步，而非完整 16 步），将动作反归一化、按 OSC 输入范围裁剪和缩放后积分得到末端位置近似：

\[
\tilde p_j=p_t+\sum_{i=1}^{j}s\odot\operatorname{clip}(a_{i,xyz}).
\]

默认代价包括末步目标平方距离、前缀平均目标平方距离（权重 0.1），以及阶段指定夹爪命令的均方误差（权重 0.001）。成功或时间上限可能使真实执行少于 8 步；当前评分长度不随提前终止动态缩短，archive 中的真实长度以 `executed_steps` 为准。这是局部运动近似，不预测接触动力学、抓取姿态、碰撞、滚动或滑落。

当前支持六个独立 On/In 放置任务，完整列表见 [task_dno.md](task_dno.md)。Close、Turnon、依赖式堆叠和未验证任务会明确拒绝。多物体任务优先继续处理当前已抓住的物体，否则选择未满足的放置目标。

**这是 privileged-state 几何实验，不是图像独立部署方案。** 原策略与 oracle DNO 的对比还包含额外任务几何信息的影响；在同一 oracle 下比较 best-of-K 与梯度 DNO，才能进一步区分搜索方法。真实 success 始终来自执行动作后的 LIBERO 检查，不能用 proxy 改善替代。

### 3.6 学习噪声初始化器：当前采用 teacher 蒸馏

实现：[noise_initializer.py](../oat/dno/noise_initializer.py)、[train_noise_initializer.py](../scripts/train_noise_initializer.py)。

\[
z_{\psi}=z_0+\Delta_\psi(c(o),z_0,f(u_t)).
\]

输入包括缓存的观测 condition、该示例原始 source 噪声和 9 维任务特征：相对目标位置 3、夹爪命令 1、阶段 one-hot 5。默认 MLP hidden_dims 为 `[128,128]`；输出为有 RMS 上界的噪声残差，末层零初始化使初始模型等于恒等映射。

监督目标是同一 source 经 DNO 改进后的噪声，训练使用 MSE。保留 source 输入可避免把不同随机 source 的最优噪声直接混成一个平均标签。训练/验证按整条 episode 划分，归一化只拟合训练数据；验证不优于恒等模型时，best.pt 保留恒等模型。

格式 v2 只归一化观测特征和相对位置，设置方差下界和输入裁剪；原始噪声与离散阶段/夹爪特征保持原尺度。这修复了最早极小样本 smoke 中低方差特征放大的问题。旧 v1 initializer 不再兼容。

checkpoint 保存模型结构、归一化、E/F 身份与**基础 checkpoint 文件的 SHA-256**。即使结构相同，换成另一轮基础策略权重也会拒绝加载。恢复一个嵌入了参考模型的 flow checkpoint，不等于能够恢复与另一个基础模型绑定的 initializer。

teacher archive 记录获选候选对应的原始噪声、优化噪声、condition、task features、episode ID、真实执行前缀及长度、t/next_t、reward 和 episode success。`accepted` 只代表对其自身 source 的 proxy 改善。`--successful-only` 进一步筛选真实成功 episodes。best-of-K 没有残差优化标签，不能作为当前蒸馏脚本的 teacher mode。

### 3.7 公平比较与日志语义

相同任务和 seed 下，候选 0 的 source 在不同模式间一致，F 的 angular dither 也在其中；这指同一观测和控制周期下的随机构造一致，不代表策略执行分叉后仍处于相同状态。每轮观测 encoder 只计算一次，候选共享缓存条件。

评估器显式给底层 simulator 设 seed，在物体落稳后取观测，并比较初始 simulator state 与 fixture 位姿哈希。不匹配会终止比较。串行执行环境，不复用旧 OrbitDNO 或训练 workspace。

令候选数 K、梯度更新数 S、Euler 步数 N：

| 模式 | 每轮候选加权前向 NFE |
| --- | --- |
| baseline | N |
| best-of-M | MN |
| dno | K(S+2)N |
| amortized | 2N |
| amortized_dno | K(S+3)N |

DNO 还包含反向传播；匹配前向 NFE 不是匹配总 FLOPs 或延迟。默认 best-of-K 比较预设自动匹配 dno 前向量，若与 amortized_dno 比较需显式调整候选数。

底层 LIBERO 会将超时合并到 done。2026-09-08 的审查已修复新评估记录：到上限且失败为 `terminated=false,truncated=true`；最后一步成功仍为正常终止。另存原始 `env_reported_terminated` 与 `time_limit_reached`。**早期 pilot/smoke 文件未改写，使用其终止字段训练 critic 前必须修正。** 现有 initializer 训练不使用这些字段。

## 4. 已验证进展与实验记录

### 4.1 时间线

| 日期 / 阶段 | 进展与解释 |
| --- | --- |
| 2026-09-07 及此前交接 | image+state 参考完成 30 epochs；E 开始训练；D/F 设计已实现 |
| 2026-09-07 | E 训练 rollout 曾出现 BrokenPipeError；worker 生命周期和错误诊断得到修复 |
| 2026-09-08 | 五种 DNO 模式、几何评分、蒸馏与独立评估脚本完成；针对性测试 46 项通过 |
| 2026-09-08 | 固定 E epoch-200 checkpoint 完成三模式完整回合 pilot，并训练 v2 initializer |
| 2026-09-09 | 本文核对实际文件和训练日志；当前训练与历史 DNO 基础模型的可用性见下文 |

46 项测试覆盖梯度链、冻结状态、E/F source、固定候选初始化、噪声约束、非有限动作、episode 划分、模型身份、真实执行前缀、相同环境初态及超时边界。此处记录已有验证，不代表本次只写文档又重跑了训练或测试。

### 4.2 E 主策略训练快照

截至 **2026-09-09 06:23:28 UTC**，`output/learned_canon/condition/seed42/logs.json` 已进入 epoch 501，读取末尾的 global_step 为 1952610。epoch 501 当时仍在进行；最近完成的 rollout 是 epoch 500。

| 已完成 rollout epoch | LIBERO-10 mean success rate | 验证 loss |
| --- | --- | --- |
| 200 | 22.8% | 0.3100 |
| 250 | 24.6% | 0.3253 |
| 300 | **27.8%** | 0.3422 |
| 350 | 27.6% | 0.3653 |
| 400 | 27.2% | 0.3813 |
| 450 | 25.6% | 0.3984 |
| 500 | 22.4% | 0.4172 |

这是训练期间的多任务 rollout 指标，不能与下节单任务四回合 pilot 的百分比直接等同。已有记录中最高成功率来自 epoch 300，而非最后一轮；尚未完成默认 5001 epochs，也尚未获得匹配的 E/F 对比结论。

当时实际存在的已保存 E 权重包括 epochs 0、50、100、150、250、300、350、400、450，以及 `latest.ckpt`。可确认的最佳已记录 rollout 对应文件为 `output/learned_canon/condition/seed42/checkpoints/ep-0300_sr-0.278.ckpt`。本次没有加载正在更新的 latest，不能仅凭其更新时间判断内部 epoch。

**历史 DNO 基础 checkpoint 已缺失。** `ep-0200_sr-0.228.ckpt` 不在原目录，包含忽略文件的全仓库文件名搜索也未找到对应命名副本。现有文件状态与训练 top-k 权重保留机制相符，但没有独立的删除事件记录，也未排除仓库外或重命名备份。

因此，4.3/4.4 的历史结果仍然成立，但其原始部署命令目前不能直接重跑。initializer 文件仍在，却不能改用 epoch 300 或 latest 作为底座。恢复方式是找到同一 SHA-256 的基础文件，或归档一个新基础 checkpoint 并重新采集 teacher、训练和评估。**不要把历史 ep0200 initializer 当成任何 E checkpoint 通用的模块。**

### 4.3 第一轮 DNO 完整回合 pilot

固定任务：`LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`。

固定基础策略：E epoch 200，读取 `ema_model`；原文件 `output/learned_canon/condition/seed42/checkpoints/ep-0200_sr-0.228.ckpt`。SHA-256：

```text
347d73a36d3d2e28c1f8ddb755b642938cf721355998d8f30923251ae6c755c0
```

4 个初始状态，seed 1000–1003，最多 550 环境步；各模式初态哈希匹配。CPU 2 线程、OSMesa；K=2、S=2，best-of-K 自动为 8。

| 方法 | 成功回合 | CPU 每段动作推理 p50 | 前向 NFE / 周期 | 反向步骤 / 周期 |
| --- | --- | --- | --- | --- |
| 原 E | 2/4 | 63.7 ms | 10 | 0 |
| E + best-of-8 | 4/4 | 105.4 ms | 80 | 0 |
| E + DNO | 3/4 | 247.7 ms | 80 | 2 |

延迟不含 simulator 执行和渲染。结果见 [results.json](../output/task_dno/e_ep0200_basket_pilot_seed1000/results.json)。

当前结论只限于该 pilot：候选重排表现最好，梯度 DNO 尚未超过增加采样。4/4 不能解释为稳定的 100% 成功率。这不是完整 LIBERO-10 分数，也不是 OOD 位姿、新物体或 F 的实验结果。

### 4.4 初始化器训练与新 seed 检查

DNO pilot 共导出 169 条实际控制周期记录；成功且被接受的记录保留 100 条。训练 episodes `[0,3]` 共 64 条，验证 episode `[2]` 共 36 条。最多训练 50 epochs，早停于 epoch 42，最佳 epoch 32。

| 指标 | 数值 |
| --- | --- |
| 恒等初始化验证噪声 MSE | 0.009145776 |
| 学习初始化验证噪声 MSE | 0.008381958 |
| 相对下降 | 8.35% |

产物：[initializer best.pt](../output/task_dno/e_ep0200_initializer_pilot_seed1000/best.pt)、[run.json](../output/task_dno/e_ep0200_initializer_pilot_seed1000/run.json)、[summary.json](../output/task_dno/e_ep0200_initializer_pilot_seed1000/summary.json)。它只绑定上述 E epoch-200 基础 checkpoint。

新 seed 2000/2001 上已执行 baseline、amortized、amortized_dno，每回合仅 16 步，初态匹配且学习噪声修正非零：[smoke results](../output/task_dno/e_ep0200_initializer_smoke_seed2000/results.json)。这些回合均未成功，短长度只验证加载和执行路径，不能评价成功率收益。8.35% 是 teacher 拟合改善，也不能等同于任务收益。

## 5. 当前 artifact 与代码入口

所有 `output/` 目录都是本地实验产物，通常不纳入 Git。要跨机器恢复上下文，必须同步对应数据与 checkpoint，不能只复制本文。

| 入口 | 用途 / 状态 |
| --- | --- |
| [heading_predictor.py](../oat/perception/heading_predictor.py) | 参考网络、监督、冻结与 artifact 格式 |
| [train_heading_reference.py](../scripts/train_heading_reference.py) | 离线参考训练 |
| [flow_policy_learned_canon.py](../oat/policy/flow_policy_learned_canon.py) | D/E/F 的参考 condition/source 接入 |
| [task_noise_policy.py](../oat/policy/task_noise_policy.py) | 五模式冻结策略包装器 |
| [task_objectives.py](../oat/dno/task_objectives.py) | 可微前缀动作评分 |
| [dno_context.py](../oat/env/libero/dno_context.py) | 当前 privileged task adapter |
| [noise_initializer.py](../oat/dno/noise_initializer.py) | v2 学习噪声残差 |
| [eval_task_dno_libero.py](../scripts/eval_task_dno_libero.py) | 串行、同初态的闭环比较与 teacher 采集 |
| [train_noise_initializer.py](../scripts/train_noise_initializer.py) | 按 episode 划分的蒸馏训练 |
| [task_dno 配置目录](../oat/config/task_dno) | `e_compare`、`f_compare`、`pilot`、`amortized_compare` |
| [参考和主策略说明](learned_canonicalization.md) | 完整配置、冻结/恢复细节；部分恢复状态为历史快照 |
| [DNO 说明](task_dno.md) | 评分限制、模式、参数与历史复现命令 |

参考可用权重：`output/heading_reference/image_state/checkpoints/best.pt`。主策略的当前可用权重见 4.2；DNO 的历史基础身份见 4.3。任何实验名称中的 `ep0200` 都保留历史含义，不能指向 ep0300 或 latest。

## 6. 下一轮实验的执行约定

### 6.1 启动实验前

1. 选择已经完成保存的基础 checkpoint，记录 E/F、epoch、model/EMA、文件 SHA-256；将其保存到不受训练 top-k 清理影响的独立目录。不要让可变的 `latest.ckpt` 承担长期实验身份。
2. 同一组 baseline、best-of-K、DNO 固定基础权重、参考模型、scorer、环境初态与生成器设置。
3. 替换基础 checkpoint 后，重新采集匹配 teacher 并训练 initializer；旧 initializer 的身份检查不能绕过。
4. 训练/调参 seed 与最终评估 seed 分开；旧 smoke 使用过 2000/2001，下一轮可预先固定例如 3000 起的独立评估集合。普通新 seed 不自动构成 OOD。

### 6.2 主策略训练模板

以下 E 命令创建一个**新 run**，不是恢复当前 seed42 目录。若训练 F 或 D，分别替换 config-name，并使用新的 `both` 或 `source` 输出目录。

```bash
conda activate oat
CUDA_VISIBLE_DEVICES=1 python scripts/run_workspace.py \
  --config-name=train_flowpolicy_heading_condition \
  reference_checkpoint=output/heading_reference/image_state/checkpoints/best.pt \
  seed=42 training.seed=42 task.policy.dataset.seed=42 \
  hydra.run.dir=output/learned_canon/condition/seed42_new
```

默认 `task.policy.lazy_eval=true` 仍计算验证 loss，但不做环境 rollout。要测闭环成功率，加 `task.policy.lazy_eval=false training.rollout_every=50`。本机 EGL 可在启动前设置 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1`；先检查 GPU 是否可用。

已有 E run 续训应保持原输出目录并使用 `training.resume=true`。恢复时刻由实际 checkpoint 决定，不能依据终端残留进度条或旧交接中 epoch190/200 的描述。

### 6.3 对已归档基础权重执行 DNO 比较

这是新实验模板；先将选定基础 E 权重归档到 `output/frozen_policies/e_base.ckpt`，不要在该路径覆盖另一轮权重。输出目录必须尚不存在。

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python scripts/eval_task_dno_libero.py \
  --config oat/config/task_dno/e_compare.yaml \
  --checkpoint output/frozen_policies/e_base.ckpt \
  --task-name LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket \
  --seed 3000 --episodes 20 --num-candidates 2 --num-grad-steps 2 \
  --device cpu --cpu-threads 2 \
  --output-dir output/task_dno/e_base_basket_seed3000
```

F 使用 `f_compare.yaml` 和独立训练、归档的 F 权重。以上参数与历史 pilot 使用同样的 K/S，但新基础权重或新 seed 的结果必须登记为新实验。GPU 空闲时可使用 CUDA；CPU + OSMesa 路径已实际验证。

教师采集、蒸馏及 learned 模式评估的命令见 [task_dno.md](task_dno.md)。其历史 ep0200 路径目前不能直接作为可用基础权重；改用一组新归档权重及其匹配 teacher/initializer，或恢复相同哈希的旧基础文件。

### 6.4 按优先级继续验证

| 优先级 | 实验 | 回答的问题 |
| --- | --- | --- |
| P0 | 归档可用基础模型；更多独立 seeds、多种受支持任务上比较 baseline / best-of-K / DNO | pilot 的提升是否稳定，梯度是否优于增加采样 |
| P1 | 获得匹配训练设置和参考的 F checkpoint，开展配对 E/F 比较；保留 D 消融 | condition 已存在时 source canonicalization 是否还有价值 |
| P1 | 采集更多成功 teacher episodes；按 episode 划分训练 initializer；在未见 seeds 完整评估 | 跨回合蒸馏能否减少搜索成本并保留任务收益 |
| P2 | 预先定义训练范围外的物体/目标位姿偏移、强度与测试 seeds | 受控 pose OOD，而非普通随机 reset |
| P2 | 明确定义 seen/unseen 物体实例或类别，扩展 BDDL/目标 adapter，记录抓取与动力学变化 | 物体泛化；苹果到球尚未实现或验证 |
| P3 | 用观测估计目标几何，或训练任务 critic；需要时再引入噪声 actor–critic | 缩小 privileged-state 假设，学习长期回报而非局部 proxy |

DNO 能否弥补物体外观识别失败或生成器缺少相应抓取行为，当前没有保证。需要区分感知错误、候选动作覆盖不足、scorer 误导和搜索能力不足，不能将所有 OOD 失败归结为噪声方向。

## 7. 工程历史与维护规则

早期 `BrokenPipeError` 表示 rollout 子进程已经退出，并不直接说明其退出原因。W&B/PyTorch deprecation warnings 不是已确认的原因。现有 runner 在 rollout 开始时用 spawn 建立 workers，结束或异常时关闭，并补充进程退出/traceback 诊断。环境数可从 20 降为 10 来降低同时占用；旧 runner 和新 runner 的历史 rollout 不应自动视为完全配对。

新 DNO 评估器串行创建和关闭环境，不启动 W&B 或训练 workspace。超时数据语义已在 3.7 记录。当前文档任务不启动新训练、不修改基础策略权重、不清理历史实验。

后续更新本文时：

- 给新结果写日期、任务与域定义、基础 checkpoint 哈希、参考身份、EMA/model、代码版本、seed 集合、回合数、成功数、预算与延迟、scorer 是否 privileged。
- 新结论附 `manifest.json` / `results.json` 等证据路径；将烟雾测试、验证 loss、代理评分、噪声 MSE 与闭环成功率分别记录。
- 训练日志已进入某 epoch 不等于该轮已完成，也不等于对应 checkpoint 已保存。新快照更新“当前状态”，保留历史实验的固定身份。
- 若 checkpoint 被 top-k 清理或迁移，更新可用性和恢复路径；不悄悄改写旧实验为新模型结果。
- 不将尚未完成的 F、state-only、OOD 或 actor–critic 设想写成已实现收益。

建议新增实验记录：

```yaml
experiment_id: <unique_id>
date_utc: <YYYY-MM-DD>
status: planned | running | completed | blocked
hypothesis: <one question>
base_policy: E | D | F
base_checkpoint: <immutable path>
base_checkpoint_sha256: <hash>
weights: ema_model | model
reference_checkpoint_or_identity: <reference identity>
code_revision: <commit and relevant uncommitted changes>
train_split_seed: 42
teacher_seeds: []
eval_seeds: []
tasks_and_domain: <task list, ID/OOD definition, seen/unseen objects>
methods_and_budget: <candidate count, gradient steps, Euler steps, forward NFE>
scorer: <version, privileged fields or image-derived inputs>
results: <successful/total episodes, uncertainty, p50/p95 latency>
artifacts: <manifest/results/checkpoint paths>
conclusion_and_limitations: <supported conclusion only>
```
