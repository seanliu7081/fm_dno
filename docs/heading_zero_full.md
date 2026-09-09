# Shared heading condition + predicted XY prior：零扰动全量训练

这是 `Condition + XY prior，零扰动，新训练` mini 的全量训练入口。对应两个 seed、四组固定噪声的离线结果为动作 MSE **0.137806**、XYZ MSE **0.131320**、生成方向 MAE **44.81°**、预测器 MAE **34.98°**；这些是 mini 结果，不是新全量模型的成绩。详见 [mini 记录](heading_prior_mini.md#completed-predicted-heading-zero-jitter-mini-2026-09-09)。

## 实现与设置

- 独立 policy：[HeadingZeroFlowPolicy](../oat/policy/flow_policy_heading_zero.py)，继承已验证的 shared-heading 机制，并固定 `condition + heading source + zero jitter + XY-only`。
- 独立 Hydra 配置：[train_flowpolicy_heading_zero.yaml](../oat/config/train_flowpolicy_heading_zero.yaml)。
- 数据：[HeadingZarrDataset](../oat/dataset/heading_zarr_dataset.py)，只用训练 episodes 拟合 Min–Max 与方向标签的 XY RMS；RGB 使用固定 `[0,255]` 范围。
- 正式 workspace 接入两个可选 hook：设置训练集 RMS、分别裁剪 core/head 梯度。旧 policy 没有这些 hook 时保持原训练路径。

| 项目 | 设置 |
| --- | --- |
| 输入 | 两个相机与机器人状态，最近 2 帧；共享 ObsEncoder |
| heading head | 128 维隐藏层；方向与有效性预测；与 encoder、flow 联合训练 |
| condition / prior 方向 | 都来自当前观测的预测 heading；GT 仅用于训练监督 |
| source | 标准高斯经 `N(R_phi(N^-1(z)))` 变换；只旋转动作 XY 通道 `[0,1]` |
| 角度扰动 | 训练、验证和默认推理都为 0；基础高斯噪声仍随机 |
| gate / 梯度 | 有效性概率 ≥ 0.5 才旋转；source 使用 detached heading；condition 和辅助监督保留梯度 |
| loss | flow loss + 0.1 × 方向 cosine loss + 0.1 × validity BCE |
| normalizer | 训练集逐维 Min–Max；没有切换到 SO(2) / SE(2) |
| 学习率 | flow `5e-5`，ObsEncoder `1e-5`，heading head `1e-3` |
| 优化 | AdamW；core/head 分别按 norm 1 裁剪；100 步 warmup 后恒定 LR；EMA |
| 动作 | 预测 16 步，执行前 8 步；Euler 10 步 |
| 默认预算 | batch 32，5001 **epochs**；没有 mini 的 5000-update 截断 |
| 闭环 SR | 默认 `task.policy.lazy_eval=false`，`training.rollout_every=50`；epoch 0、50、100…训练结束后评估 |
| rollout 并行数 / 总回合 | `n_parallel_envs=10`；每次仍评估 500 回合 |
| 数据划分 | 500 条示范中 450 条训练、50 条验证；split seed 固定 42 |

全量数据有 **124,342 个训练帧窗口、13,748 个验证帧窗口**，替代 mini 的任务均衡 4,000/1,000 固定窗口。默认沿用基础配置的 `drop_last=true`：训练每轮随机打乱后丢弃最后不足 32 的 batch；验证也丢弃末尾不足一个 batch。需要每轮包含尾 batch，可增加 `dataloader.drop_last=false val_dataloader.drop_last=false`。

优化 seed 可以改变，数据 split seed 仍固定为 42。全量训练使用正式 DataLoader / 随机数流；不会复现 mini 的逐步随机序列。正式 scheduler 的首更新 LR 为 0，而 mini 为基础 LR 的 1/100；两者都有 100 步 warmup。RTX 4090 上两条训练路径均使用 BF16 autocast。

## 启动新训练

从仓库根目录运行；GPU 1 在进程内映射为 `cuda:0`。下面启动训练、离线验证和默认每 50 epochs 的闭环 SR 评估；模拟器使用 GPU 1 的 EGL，W&B 记录保存在本地。

```bash
cd /home/haotian/code/fm_dno
conda activate oat
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/run_workspace.py \
  --config-name=train_flowpolicy_heading_zero \
  seed=42 \
  training.resume=false \
  logging.mode=offline \
  hydra.run.dir=output/heading_zero/seed42_full_env10
```

`training.num_epochs=5001` 是整个训练集的 epoch 数，与 mini 的 `--steps 5000` 不同。需要其他预算时直接覆盖，例如 `training.num_epochs=1001`。独立 seed 43 使用 `seed=43` 与 `hydra.run.dir=output/heading_zero/seed43_full_env10`。不需要预训练 heading reference，也不需要 mini checkpoint；默认从头联合训练。

本次失败的 `output/heading_zero/seed42_full` 尚无 checkpoint：首次 rollout 发生在该轮保存 checkpoint 之前。上面的命令使用新目录 `output/heading_zero/seed42_full_env10` 从头启动，以保留失败运行的日志。

续训同一 run 时保持相同配置和输出目录，将 `training.resume=false` 改成 `training.resume=true`；正式 workspace 从该目录的 `checkpoints/latest.ckpt` 恢复模型、EMA、优化器和训练进度。避免把已有训练目录用于另一次从头训练。

如需在线 W&B，把 `logging.mode=offline` 改成 `logging.mode=online`。此配置默认已设置 `task.policy.lazy_eval=false training.rollout_every=50`，训练命令无需额外覆盖。workspace 从 epoch 0 开始编号，因此闭环 SR 在 epoch 0、50、100…训练结束后评估；默认 Libero runner 使用 10 个并行环境，每次仍评估 500 回合。

## Checkpoint 与指标

验证 loss、动作重建和 checkpoint 保存仍默认每 10 epochs 执行一次，即 epoch 0、10、20…训练结束后；闭环 SR 按独立的 50-epoch 周期执行。保存 `checkpoints/latest.ckpt`，并按 `test_reconst_mse` 越低越好保留三个 checkpoint。

`test_reconst_mse` 是标准 workspace 的 **完整 16 步、7 维原始动作 MSE**，使用当前验证窗口和新采样噪声；mini 表格是固定窗口/噪声下的 **前 8 步** 指标。两者数值不能直接当作同一评估协议比较。默认 rollout 会记录 SR；此配置仍按重建 MSE 选 top-k。

checkpoint 保存独立类的配置、模型/EMA、normalizer 和 `heading_xy_rms`。标准入口可直接恢复用于推理，不需要访问训练数据或独立 heading reference：

```python
from oat.policy.base_policy import BasePolicy

policy = BasePolicy.from_checkpoint(
    "output/heading_zero/seed42_full_env10/checkpoints/latest.ckpt"
).to("cuda:0").eval()
```

## 首次 rollout 的内存排查（2026-09-09）

一次实际训练运行在首次 rollout 创建 20 个 Libero worker 时被系统以 `Killed` 终止。Gym 停止维护的提示来自依赖导入，本身没有给出此次退出的异常原因。排查时当前会话 cgroup 已累计记录 4 次历史 OOM kill，历史内存峰值约为 **52.8 GiB**；无法读取带时间戳的内核 OOM 记录，因此不能将累计计数精确对应到这次进程。现有证据支持主机 RAM 压力的判断。

此配置的默认并行环境数已从 20 降为 **10**，每次评估仍为 **500 回合**，每 **50 epochs** 进行一次。模拟器 worker 的闭包只携带环境参数，没有捕获训练 dataset 或 policy；新增内存来自独立模拟器、依赖和渲染资源叠加在训练进程已有内存上。真实 10 并行的有界重试已通过，细节见下方验证记录。

20→10 保持当前 LIBERO-10 的名义任务/seed 排列表，但不能保证真实初始状态逐项配对：已有 runner 在重复相同任务时没有按每回合的记录 seed 重新播种，`LiberoEnv.reset(seed=...)` 也没有使用传入 seed。不同并行数会改变 worker 的随机状态历史；本次内存设置修改没有同时改变这套 seed 行为。

## 验证状态（2026-09-09）

相关 110 项测试通过，覆盖新配置、训练集统计、联合梯度、优化器与独立裁剪、零扰动 source，以及 checkpoint 恢复。正式 workspace 已在真实完整数据上完成 3 个训练 batch、1 个验证 batch、1 个动作重建 batch，并保存模型和 EMA；该历史 smoke 使用 `task.policy.lazy_eval=true`，没有运行模拟器 rollout；这只是入口验证，不是完整训练或性能评估。

真实 checkpoint 的恢复检查与可复现脚本见 [验证记录](../output/heading_zero/full_training_validation_20260909/verification.json) 和 [验证脚本](../output/heading_zero/full_training_validation_20260909/verify_checkpoint.py)。

内存排查后的真实 10 并行重试以 **exit 0** 完成，耗时 **38.05 秒**：保留完整 RGB 数据集，执行 3 个训练 batch、20 个截短回合（每任务 2 个，每回合 8 个动作）、2 个视频，以及验证、动作重建和 checkpoint 保存。主机可用 RAM 最低 **19.441 GiB**；本次监测的会话 cgroup 占用峰值 **41.410 GiB**（包含缓存），OOM kill 累计计数保持 **4→4**。相关配置、worker 生命周期和错误处理的 14 项测试通过。证据：[内存监控结果](../output/heading_zero/eval_memory_debug_20260909_retry1/guard_result.json)、[workspace 日志](../output/heading_zero/eval_memory_debug_20260909_retry1/workspace.log)。

此前首次受监控验证由用户手动停止，收到 SIGTERM、exit 143，OOM 计数没有增加；记录见 [中断说明](../output/heading_zero/eval_memory_debug_20260909/interruption.json)。上述成功重试仅验证 10 并行的训练/rollout 路径，**不等于完成默认 500 回合、最长 550 步的 SR 评估，也不是完整训练成绩**。本次排查没有进一步启动完整训练。
