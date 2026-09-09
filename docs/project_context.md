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
| 归一化 | 历史 D/E/F：动作 SO(2) block RMS；新增 shared-heading mini / 全量零扰动版本：训练集逐维 Min–Max |

优化 seed 与数据划分 seed 是不同变量。多 seed 实验可以改变训练 seed，但保持参考训练的 `split_seed=42` 与策略的 `task.policy.dataset.seed=42`。方向参考、新增 shared-heading mini 和全量零扰动版本的归一化统计只拟合训练 episodes。2026-09-09 代码核查发现：原通用 `ZarrDataset.get_normalizer()` 使用整个 replay buffer，因此历史主策略不能一概称为仅拟合训练统计；本次未修改该通用路径。

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

### 4.5 Shared ObsEncoder + heading 的离线 mini test（2026-09-09）

为检验“同一观测是否已经足以从主策略特征中读出方向，以及显式 heading 是否仍有优化价值”，新增 [SharedHeadingFlowPolicy](../oat/policy/flow_policy_shared_heading.py) 和 [mini runner](../scripts/mini_shared_heading.py)。这三个分支是新的共享编码器实验，**不沿用原冻结 E/D/F 的模型身份**。

共同协议：原 450/50 episode 划分、split seed 42；10 个任务均衡采样 4,000 个训练窗口和 1,000 个验证窗口；训练 seed 42/43；每分支从零训练 5,000 updates，batch 32；相同骨干、初始权重、批次、增强、flow noise/time；普通 Min–Max 动作归一化，统计仅来自训练 episodes；IID source；EMA、10 步 Euler。方向标签保持原始 XY 的 16 步合成方向，采样动作误差使用前 8 步。两个辅助损失权重均为 0.1。

| 分支 | 方向读出 MAE | Flow 验证 MSE | 前 8 步原始动作 MSE | 生成动作块方向 MAE |
| --- | ---: | ---: | ---: | ---: |
| Basic Flow + detached 诊断读出头 | 39.93° | 0.181017 | 0.151808 | 69.43° |
| 共享 encoder + 方向辅助监督 | 33.56° | 0.181084 | 0.159220 | 65.52° |
| 共享 encoder + 辅助监督 + heading condition | 34.85° | 0.174208 | 0.139015 | 50.78° |

表中为两个 seed 均值；方向指标使用有效标签，生成动作指标使用均衡覆盖验证 episodes 的 200 个窗口。Basic Flow 的读出头只接收 detached 特征，且与策略分开裁剪梯度，不改变策略更新。仅由训练窗口拟合的全局/任务平均方向参照误差分别为 82.93°/78.19°。

结果支持普通共享特征可读出方向；但特征包含机器人状态，不能单独证明视觉几何学习。辅助监督使方向更准，却没有改善整体动作目标。新增 condition 相比 baseline 的 flow MSE 平均下降 3.76%，生成动作方向误差下降 26.86%，两个 seed 均改善；原始动作 MSE 平均下降 8.43%，但分别是 **16.28% 和 0.04%**，对训练 seed 的稳定性尚未建立。

**没有进行环境 rollout，也没有测 SR。不能将这些离线误差转换为成功率，更不能宣称恢复了用户所述 39% baseline。** 原 baseline 39% 的精确 checkpoint 尚未定位。本次固定预算约 928.4 秒（不含数据准备/开发）；600-update pilot 仅用于吞吐和早期学习检查，正式结论使用预设 5,000-update 终点。

完整证据：[协议与复现](shared_heading_mini.md)、[结果](../output/mini_shared_heading/mini_5000_20260909/results.md)、[配对 episode 统计](../output/mini_shared_heading/mini_5000_20260909/comparison.json)、[权重哈希](../output/mini_shared_heading/mini_5000_20260909/artifact_hashes.json)。17 项 CPU 测试、三分支 GPU smoke、真实观测的保存/恢复一致性检查通过。权重修复仅将版本元数据转为普通字符串，tensor 值未变。

### 4.6 Shared heading condition + noise prior 的离线 mini test（2026-09-09）

在 4.5 的共享、联合训练 heading condition 上增加两种方向 source。三组保持同一普通 Min–Max normalizer、数据划分、初始权重、批次及原始高斯噪声；每组 5,000 updates，seed 42/43。head 继续通过 condition 和辅助损失训练，**仅构造 source 时 detach 当前预测**，不测试通过 source 端点反传的方案。

先验在原始坐标中计算并对齐这次噪声的平移 XY 合成方向，再归一化：`N(R_phi N^-1(z))`；`phi = predicted_heading + VonMises(0,4) - raw_noise_heading`，预测 validity 至少 0.5 才启用。`condition_prior_xy` 只旋转平移 XY；`condition_prior_both` 同时旋转旋转动作 XY。这份数据的平移 XY Min–Max 恰好相同缩放且零偏移，旋转 XY 则不是，因此双块先验还改变旋转噪声的均值/协方差，**不是原始 SO(2)-normalizer F 的精确复现**。

| 分支 | 主评估动作 MSE | 主评估生成方向 MAE | 额外 4 组噪声动作 MSE | 额外 4 组噪声生成方向 MAE |
| --- | ---: | ---: | ---: | ---: |
| Condition + IID | 0.139015 | 50.78° | 0.140088 | 51.42° |
| Condition + 平移 XY prior | 0.150203 | 49.00° | 0.141372 | 47.54° |
| Condition + 双 XY 块 prior | 0.144900 | 49.16° | 0.140320 | 48.67° |

所有值为两个训练 seed 均值，动作指标为相同 200 个验证窗口的前 8 步原始动作 MSE。主评估每个窗口一个固定 latent；额外 4 组独立、跨模型配对的 latent 只用于推理复核，不重训、不选最优采样，且保留原主结果。此补充在第一个 seed 显示“方向略好、动作更差”、第二个 seed 尚在训练时确定。

主评估中 XY-only / 双块 prior 的平均动作 MSE 分别上升 8.05% / 4.23%。但额外噪声均值只高 0.92% / 0.17%，且两个训练 seed 的差值方向相反；配对 episode 区间都包含 0，故**不能宣称稳定退化**。额外噪声下生成方向误差分别下降 3.88° / 2.75°，两个 seed 都改善。

当前证据是：方向 prior 确实有引导作用，但在已有共享 heading condition 后，**尚未显示整体动作质量的稳定额外收益**。下一步可保留联合训练的 head + condition，以 IID source 为默认对照。不同 source 的 flow 插值与速度标签不同，不能拿 native flow MSE 直接排名。此实验没有 rollout/SR，不能判断历史 39% vs 28% 差距或推广为所有 D/E/F 的必然结论。

三组训练合计约 946.7 秒，补充推理约 6.5 秒（均不含数据准备）。31 项 CPU 测试通过；六个最终模型安全加载、配置恢复和真实观测重复采样检查通过。两个 IID 对照组逐步评估与各 487 个最终参数张量完整复现 4.5 的 condition 结果。

证据：[协议与复现](heading_prior_mini.md)、[主结果](../output/mini_heading_prior/mini_5000_20260909/results.md)、[额外噪声验证](../output/mini_heading_prior/sampling_robustness_20260909/report.md)、[独立检查与权重哈希](../output/mini_heading_prior/mini_5000_20260909/verification.json)。代码：[训练 runner](../scripts/mini_heading_prior.py)、[额外采样 evaluator](../scripts/evaluate_mini_heading_prior_sampling.py)。

### 4.7 固定模型的 prior 方向精度诊断（2026-09-09）

为检验 heading 精度是否限制现有 XY prior，固定 4.6 的两个 `condition_prior_xy` EMA 模型、原预测 heading condition、Min–Max normalizer、validity gate、10 步 Euler，以及补充评估保存的 200 个验证窗口 × 4 组高斯噪声和 κ=4 角度扰动。仅将 source 使用的角度替换为 GT 或 GT 加 ±15°/±30°/±60°；误差正负号按窗口/噪声固定，跨角度级别和训练 seed 共享。GT 使用原始未来 16 步动作的 XY 合成方向，199/200 个标签有效，无效的 1 个窗口在所有组中精确保留原结果。没有重训。

| Source 角度 | 前 8 步动作 MSE | XYZ MSE | 生成 16 步方向 MAE |
| --- | ---: | ---: | ---: |
| 原预测 | 0.141372 | 0.139278 | 47.54° |
| GT | 0.137352 | 0.129041 | 31.14° |
| GT ±15° | 0.138785 | 0.132548 | 34.03° |
| GT ±30° | 0.143387 | 0.142675 | 42.09° |
| GT ±60° | 0.159963 | 0.179812 | 66.53° |

GT 替换使动作 MSE 降低 **2.84%**、XYZ MSE 降低 **7.35%**，两个 seed 均改善。平均动作差值为 −0.004019，按任务内 episode 配对 bootstrap 的 95% 区间为 [−0.006516, −0.001295]；此区间条件于两个固定模型和四组固定噪声，不表示跨训练 seed 的不确定性。两个模型的动作、XYZ 和生成方向误差都随注入误差 0/15/30/60° 单调上升。按原预测误差分组，>60° 窗口的改善最大，<15° 组基本持平；分组结果为描述性诊断。

**此结果支持继续改进 source 的 heading 精度，但尚未证明改进后重训能超过 IID 或提高 SR。** GT 来自未来标签，只能用于诊断；模型训练时使用预测方向 source，而本次保留的 condition 仍可能预测错误。GT 组也保留 κ=4 随机扰动，因此初始噪声实际方向误差仍为 23.56°，并非精确对齐或严格性能上界。

推理约 6.2 秒（不含数据准备）；原预测组的 source 和五项动作指标逐位复现上一轮补充评估。13 项新增诊断 CPU 测试与 31 项原策略测试全部通过。证据：[完整协议与复现](heading_prior_mini.md#fixed-model-source-angle-diagnostic-2026-09-09)、[报告](../output/mini_heading_prior/angle_sensitivity_20260909/report.md)、[敏感性曲线](../output/mini_heading_prior/angle_sensitivity_20260909/angle_sensitivity.png)、[分组图](../output/mini_heading_prior/angle_sensitivity_20260909/accuracy_strata.png)。代码：[evaluator](../scripts/evaluate_heading_angle_sensitivity.py)、[reporter](../scripts/summarize_heading_angle_sensitivity.py)。

### 4.8 GT prior 的 condition × jitter 2×2 诊断（2026-09-09）

为解释 4.7 的 GT prior 为何仍有约 31° 生成方向误差，固定同一模型、200 个验证窗口、4 组噪声和 GT source 中心角，交叉比较预测/GT heading condition 与原 κ=4 角度扰动/零扰动。GT condition 仅替换显式 cos/sin 两维，保留全部观测特征和预测 validity；原 source gate、normalizer 与采样器均不变。199 个有效标签参与干预，1 个无效窗口在四组中精确保留原预测 source、原扰动及原 condition。

| Heading condition | Prior 角度扰动 | 前 8 步动作 MSE | XYZ MSE | 生成 16 步方向 MAE |
| --- | --- | ---: | ---: | ---: |
| 预测 | 原 κ=4 | 0.137352 | 0.129041 | 31.14° |
| 预测 | 零扰动 | 0.131390 | 0.115072 | 15.60° |
| GT | 原 κ=4 | 0.142568 | 0.128623 | 32.64° |
| GT | 零扰动 | 0.136482 | 0.114723 | 15.94° |

保持预测 condition 时，去掉扰动使动作 MSE 降低 **4.34%**、XYZ MSE 降低 **10.83%**，方向误差由 31.14° 降至 15.60°。两个 seed 都改善，GT condition 下去掉扰动也有改善。预测 condition 下动作差值为 −0.005962，条件于固定模型/噪声的配对 episode-bootstrap 95% 区间为 [−0.007285, −0.004527]。

把显式 condition 换成 GT 没有改善平均整体动作误差；主要变化是夹爪误差增加，XYZ 基本不变。两个 seed 的动作差值均为正，但对应 episode 区间包含零，不能宣称普遍退化。整体动作的交互差值为 −0.000124，区间 [−0.001020, +0.000723]。这些区间不估计跨训练运行的不确定性。

零扰动时初始 source 方向误差约为 0.000006°，但经过 flow 后仍为 15.60°/15.94°，说明精确对齐初始噪声不能强制最终方向正确；本次也未支持“把显式 condition 改 GT 就能消除剩余误差”的假设。不能将 source 与生成动作的 MAE 相减来分摊原因。所有组使用未来 GT 标签，GT condition 与零扰动都改变训练输入分布；**尚不能推断真实预测 prior 应直接取消扰动，也没有测 SR**。

推理约 5.4 秒（不含准备），57 项 CPU 测试通过；原 GT source + 预测 condition + 原扰动组逐位复现 4.7。独立审计从原始 Zarr 重算全部 6,400 份生成动作指标并检查配对、condition 两维替换、source 几何与无效标签回退。证据：[协议与复现](heading_prior_mini.md#gt-source-condition--jitter-factorial-2026-09-09)、[报告](../output/mini_heading_prior/factorial_20260909/report.md)、[四组对比图](../output/mini_heading_prior/factorial_20260909/factorial_cells.png)、[独立审计](../output/mini_heading_prior/factorial_20260909/verification.json)。代码：[evaluator](../scripts/evaluate_heading_factorial.py)、[reporter](../scripts/summarize_heading_factorial.py)。

### 4.9 零角度扰动的可训练入口（2026-09-09）

新增 `scripts/mini_heading_prior.py --modes condition_prior_xy --source-jitter zero`：使用联合训练的预测 heading 同时作为 condition 与 XY prior 中心方向，训练、验证、默认推理和权重恢复均采用精确零角度扰动。仍保留高斯噪声、validity gate 和构造 source 时的 detach。GT 只用于监督，**不等同于 4.8 使用 GT prior 得到的 0.131390 / 15.60° 结果**。

新 checkpoint 保存 `source_jitter`；旧 checkpoint 缺省恢复为 `vonmises`。显式传入角度扰动仍可覆盖默认值供诊断使用；`kappa=0` 表示均匀角度分布，不能当成零扰动。69 项 CPU 测试、CLI 和真实数据单步 GPU smoke 通过；该版本的正式 mini 已完成，结果见 4.10。默认命令仍为 4,000 固定训练窗口、两个 seed 各 5,000 updates 的 mini 协议；命令见 [训练说明](heading_prior_mini.md#training-with-predicted-heading-and-zero-angular-jitter)。

### 4.10 预测 heading + 零扰动 prior 的正式 mini 结果（2026-09-09）

按 4.9 命令从头训练两个 seed（42/43），各 5,000 updates；4,000 个训练窗口、1,000 个验证窗口、初始化、Min–Max normalizer 和其他训练参数保持与原实验一致。训练约 314.0 秒，随后用相同 200 个窗口 × 4 组已保存高斯噪声比较新模型与原 IID/κ=4 模型，推理约 6.6 秒。三组 condition 均来自预测，方向 prior 的中心角也均来自预测，未使用 GT 作为 source 输入。

| Source | 前 8 步动作 MSE | XYZ MSE | 生成方向 MAE | 预测器 MAE |
| --- | ---: | ---: | ---: | ---: |
| IID | 0.140088 | 0.140357 | 51.42° | 34.85° |
| 预测 XY prior，κ=4 | 0.141372 | 0.139278 | 47.54° | 35.15° |
| 预测 XY prior，零扰动 | 0.137806 | 0.131320 | 44.81° | 34.98° |

生成指标平均两个 seed 和四组噪声；预测器 MAE 使用原完整 1,000 个验证窗口中的有效标签。新模型平均动作 MSE 比 IID 低 1.63%、比 κ=4 低 2.52%，但**seed 42 对两个对照都变差，seed 43 对两个对照都改善**。零扰动的动作 MSE 分别为 0.143878/0.131734，IID 为 0.137924/0.142252，κ=4 为 0.142140/0.140604。XYZ 和生成方向误差在两个 seed 中均改善；预测器精度仍约 35°。

零扰动减 IID 的配对动作差值为 −0.002282，固定模型/噪声条件下 episode-bootstrap 95% 区间为 [−0.006953, +0.002418]；减 κ=4 为 −0.003566，区间 [−0.006716, −0.000253]。这些区间不度量跨训练 seed 的不确定性，不能因后一项不跨零就声称 seed 间稳定。原单噪声主评估中，零扰动平均动作 MSE 为 0.140133，略高于 IID 的 0.139015，也保留在报告中。

**当前支持零扰动对 XYZ 和生成方向的改善，但未建立整体动作质量对 IID 或 κ=4 的稳定优势，没有测 SR。** 4.8 的 0.131390 / 15.60° 仍是旧模型使用 GT prior 的离线参照，不能归为新模型成绩。

新旧 split、normalizer 和初始预测逐项核对一致；当前代码下四个旧模型的 source 与动作指标逐位复现，κ=4 模型的完整生成动作也逐位复现。训练前已通过 69 项 CPU 测试。证据：[完整协议与复现](heading_prior_mini.md#completed-predicted-heading-zero-jitter-mini-2026-09-09)、[汇总报告](../output/mini_heading_prior/zero_jitter_comparison_20260909/report.md)、[对比图](../output/mini_heading_prior/zero_jitter_comparison_20260909/comparison.png)、[训练产物](../output/mini_heading_prior/predicted_zero_5000_seed42_43/summary.json)。代码：[跨训练 evaluator](../scripts/compare_zero_jitter_training.py)。

### 4.11 预测 heading + 零扰动 prior 的全量训练入口（2026-09-09）

新增独立 [policy](../oat/policy/flow_policy_heading_zero.py) 与 [配置](../oat/config/train_flowpolicy_heading_zero.yaml)，将 4.10 的 `condition_prior_xy + source_jitter=zero` 接入正式 `TrainPolicyWorkspace`。共享 encoder/head 联合训练，预测方向同时用于 condition 和 XY prior；GT 仅作监督。保留 Min–Max、训练集 RMS、独立 head LR `1e-3`、core/head 分别裁剪及 100 步 warmup。新 [dataset](../oat/dataset/heading_zarr_dataset.py) 只用训练 episodes 拟合统计；workspace 的可选 hook 支持 RMS 初始化和分组裁剪，旧 policy 路径保持原行为。

默认固定 split seed 42，450/50 episodes；候选窗口从 mini 的 4,000/1,000 扩展到 124,342/13,748，使用正式 DataLoader。batch 32、5001 epochs 是完整训练预算，**不是 5000 updates**；默认 `drop_last=true` 的尾 batch 行为和 scheduler 首步差异见说明。checkpoint 保存完整配置、normalizer 和 RMS，标准 `BasePolicy.from_checkpoint` 可恢复，无需 heading reference 或推理时访问 dataset。

此配置默认 `task.policy.lazy_eval=false`、`training.rollout_every=50`，开启每 50 epochs 的闭环 SR 评估；默认已改为 `task.policy.env_runner.n_parallel_envs=10`，每次仍评估 500 回合。workspace 从 0 开始编号，实际在 epoch 0、50、100…训练结束后 rollout；验证 loss、动作重建和 checkpoint 保存仍每 10 epochs 执行，即 epoch 0、10、20…。top-k 仍按完整 16 步 raw reconstruction MSE 选取，该指标与 mini 的前 8 步 MSE 不是同一协议。GPU 1 的启动命令已包含 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1`。

相关 110 项测试通过；真实全量数据的正式 workspace 已完成 3 个训练 batch、验证、动作重建与 checkpoint 保存/加载检查。该历史 smoke 使用 `task.policy.lazy_eval=true`，没有运行模拟器 rollout。**该历史 smoke 没有运行完整训练或 SR 评估。** 命令与说明：[heading_zero_full.md](heading_zero_full.md)；证据：[验证记录](../output/heading_zero/full_training_validation_20260909/verification.json)。

后续实际启动在首次 rollout 创建 20 个 Libero worker 时出现 `Killed`。Gym 停止维护提示仅来自导入，不能据此认定退出原因。排查时当前会话 cgroup 累计记录 4 次历史 OOM kill，历史内存峰值约 52.8 GiB；因无法访问带时间戳的内核记录，现有证据支持 RAM 压力，但尚不能把累计事件与该进程逐项对应。模拟器闭包未捕获训练 dataset/model。默认并行数降至 10 后，有界真实重试已通过：完整 RGB 数据集、3 个训练 batch、20 个截短回合（每任务 2 个、每回合 8 个动作）、2 个视频、验证/重建/checkpoint 均完成，exit 0，耗时 38.05 秒。监测期间主机可用 RAM 最低 19.441 GiB，会话 cgroup 占用峰值 41.410 GiB（含缓存），OOM kill 计数保持 4→4；相关配置与 worker 测试 14 项通过。证据：[guard_result.json](../output/heading_zero/eval_memory_debug_20260909_retry1/guard_result.json)、[workspace.log](../output/heading_zero/eval_memory_debug_20260909_retry1/workspace.log)。这不等于完整训练或默认 500 回合、最长 550 步的 SR 评估。

20→10 保留名义任务/seed 排列表，但已有 runner 对同任务重复回合未重新应用记录 seed，且 `LiberoEnv.reset` 忽略传入 seed，因此不能保证不同并行数的真实初始状态精确配对。本次仅调整该配置的并行数，未修改 seed 行为。

首次受监控验证由用户手动停止（SIGTERM/exit 143，无新增 OOM），见 [中断记录](../output/heading_zero/eval_memory_debug_20260909/interruption.json)。失败的 `output/heading_zero/seed42_full` 在首次 rollout 前尚未保存 checkpoint；文档中的重新启动命令改用 `output/heading_zero/seed42_full_env10` 和 `training.resume=false`，保留旧日志。本次排查没有进一步启动完整训练。

## 5. 当前 artifact 与代码入口

所有 `output/` 目录都是本地实验产物，通常不纳入 Git。要跨机器恢复上下文，必须同步对应数据与 checkpoint，不能只复制本文。

| 入口 | 用途 / 状态 |
| --- | --- |
| [heading_zero_full.md](heading_zero_full.md) | 共享 heading condition + 预测 XY prior 零扰动的独立 policy、完整训练配置与命令 |
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
