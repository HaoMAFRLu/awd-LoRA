# MoE 匈牙利硬匹配：先更新 consensus

使用 `--cfg_version ns97m_hungarian` 选择新模式。每轮按
**W_e → X → P_e → X_e → Y_hat_e** 更新，保留自由的 expert-specific 矩阵 X_e。
三个投影联合匹配、共用同一份 P_e；所有专家都参与，包括专家 0。
原 `ns97m_sinkhorn`、`ns97m_aligned*`、未对齐和 vanilla 配置保持原有行为。

正式配置继承现有 97M 的模型、数据、AdamW、学习率、weight decay、rho 和训练预算。
与 `ns97m_sinkhorn` 相比，改动是 P_e 的硬置换表示、匹配求解器和 X/P_e 更新顺序。
初始化仍在第 0 步完成：P_e=I、X=专家均值、X_e=W_e−X、Y_hat_e=0。
这不修改网络权重或额外消耗随机数，也不执行初始化匹配、Sinkhorn 或 SVD。
从第 10 步起每 10 步更新一次结构；结束时需要补更新，也执行完整的 X/P_e/X_e/Y_hat_e 更新。

## 更新公式

沿用问答文档的记号：X 为 consensus，X_e 为 expert-specific 矩阵，W_e 为完整专家权重，
P_hat_e=diag(P_e,P_e,P_e)。代码的 `dual` 保存 U_e=Y_hat_e/rho。
先按原任务训练流程更新 W_e，再执行以下一次结构更新：

1. 固定旧 X_e 和旧 U_e，形成 R_e=W_e(new)−X_e(old)+U_e(old)。X 与 P_e 共用这份目标。
2. 用旧 P_e 对齐 R_e，取均值得到新 X：
   `X(new) = mean_e R_e @ P_hat_e(old).T`。
3. 固定新 X，对每个专家求使 `||R_e − X(new) @ P_hat_e||_F²` 最小的硬置换。
   配对代价同时包含 gate、up 和 down 的通道差异。
4. 使用新 X 和新 P_e 更新 `X_e(new) = W_e(new) − X(new) @ P_hat_e(new) + U_e(old)`。
5. 更新 `U_e(new) = U_e(old) + W_e(new) − X(new) @ P_hat_e(new) − X_e(new)`，刷新训练目标。

步骤 3 后不再重新平均 X。步骤 4 中不能直接以 R_e 替换 W_e(new)+U_e(old)，
否则会多减一次旧 X_e。自由残差下，U_e 在精确算术中归零，代码保留实际浮点计算结果。

P_e 不使用梯度、伪逆、截断或 Sinkhorn。复用 SciPy 的 `linear_sum_assignment`
求解线性指派（具体实现为 modified Jonker–Volgenant）。新配置的
`improvement_tolerance: 0.0` 接受任何严格降低计算代价的排列，相同代价保留旧排列。
代价采用 FP32、直接距离，指派在 CPU 上求解；这是固定本轮 R_e 和 X 时的子问题最优解。

P_e 在 checkpoint 中保存为 int64 `[expert, native_channel]` 索引：
`permutation[e, a] = b` 表示原通道 a 对应共享通道 b。
PyTorch 的权重布局与文档转置，因此 gate/up 用 `shared[permutation]`，
down 用 `shared.T[permutation].mT`。严格置换保持映射前后的 Frobenius 范数。

## 配置、恢复与运行

新增配置为 [ns97m_hungarian.yaml](../configs/ns97m_hungarian.yaml)，核心设置如下：

```yaml
salaad:
  residual_mode: dense
  low_rank_enabled: false
  sparse_enabled: false
  controller: null
  initialization: identity_shared_mean_residual_dual_zero
  structure_order: [shared, permutation, residual, dual]
  channel_alignment:
    method: hungarian
    fix_reference: false
    match_every_optimizer_steps: 10
    improvement_tolerance: 0.0
```

此模式要求匹配间隔与结构间隔相等。继承的 `reference_expert` 与
`initialization_max_iterations` 在当前单位矩阵初始化下不参与计算。
同层三个投影恢复后共用 P_e，加载时验证排列为双射及三个投影一致。
导出沿用 dense residual 的 v5 格式，P_e 的约定为 `native_to_shared`，
重构权重不包含 U_e。原有 checkpoint 可继续用原配置恢复；新模式需另开实验，
不支持将已有软 P checkpoint 直接当作硬置换续训。

检查正式配置：

```bash
myenv/bin/python scripts/train_salad.py --cfg_version ns97m_hungarian --dry-run
```

CPU 小模型在第 3 步初始化、第 5/7/8 步更新结构，检查中断和恢复：

```bash
myenv/bin/python scripts/train_salad.py --cfg_version smoke_hungarian \
  --device cpu --output /tmp/new_hungarian_smoke --stop-after 5
myenv/bin/python scripts/train_salad.py --cfg_version smoke_hungarian \
  --device cpu --resume /tmp/new_hungarian_smoke/checkpoints/step_00000005
```

测试包括穷举小矩阵排列核对求解结果、更新顺序、非单位专家 0、初始化、
最终补更新、失败时不提交状态、训练恢复和导出：

```bash
myenv/bin/python -m unittest discover -s tests/moe -p 'test_*.py' -q
myenv/bin/torchrun --standalone --nproc-per-node=2 \
  tests/moe/alignment_distributed_worker.py /tmp/new_hungarian_dp2 --hungarian-dense
```

正式训练沿用原入口和语料，在分配到 GPU 后运行：

```bash
myenv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_salad.py \
  --cfg_version ns97m_hungarian --data-manifest /path/to/manifest.json \
  --output /path/to/new_run
```

2026-10-07 本地验证：新增 8 项测试与原有测试共 117 项均已验证通过，
正式配置 dry-run、CPU 双进程 8 步训练及从第 2/3/7 步逐位恢复通过。
全套复跑中一次训练入口测试遇到 Gloo 本机回环地址解析失败；
在允许本机网络的环境中重跑入口测试组，15 项全部通过。
