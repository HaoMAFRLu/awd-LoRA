# Sinkhorn 软通道对齐训练

配置入口是 `--cfg_version ns97m_sinkhorn`。同层 gate/up/down 共用一份软 P，
完整 expert 权重仍用于任务前向；L/S/U、Adam 和 streaming SVD 基保持专家原通道顺序。

正式配置在训练开始前初始化分解和 P，不保留 vanilla 前缀。
从第 10 步开始，每 10 步同时更新 P、shared、L、S、U；每轮 P 更新做 **8 步 SGD**。
结束时若需要补一次结构更新，P 也一起更新。总预算仍为 2100 步。

## 新增设置

这些是首版运行默认值，尚未经过正式 MoE 训练调优。原任务训练参数、
rho=1e-5、L/S 控制器和结构更新间隔沿用现有 97M 配置。

| 配置项 | 值 | 含义 |
|---|---:|---|
| `inner_steps` | 8 | 每轮优化分数表 A 的梯度步数 |
| `temperature` | 1.0 | Sinkhorn 温度，全程固定 |
| `learning_rate` | 1.0 | P 子问题的 SGD 步长 |
| `move_penalty_over_rho` | 1.0 | lambda_move / rho |
| `initial_softening` | 0.1 | 初始硬 P 与均匀矩阵混合的比例 |
| `max_iterations` | 100 | 每次 Sinkhorn 行列归一化的迭代上限 |
| `marginal_tolerance` | 1e-5 | 最大行/列和误差容限 |

P 的 SGD 步长与模型 AdamW 的学习率分别配置。它优化的是文档目标 **F/rho**：
重构平方误差的一半，加 `move_penalty_over_rho / 2 * ||P-P_old||²`。
整体除以 rho 保持子问题的最优解和两项的相对权重，避免小 rho 让梯度更新消失；
模型梯度中的 rho 不变。8 个内部步骤始终使用同一份外层旧 P、旧 shared 和残差。
第一步的变化惩罚梯度为零，后续步骤才开始受到它的影响。

## 执行顺序

1. 训练开始前，用原来的三个投影联合硬匹配生成初始排列。对非参考专家取
   `P = 0.9 * P_hard + 0.1 / channels`，令 `A = temperature * log(P)`。
   专家 0 保持 P=I、A=0；它的固定关系也使 shared 的法方程矩阵可逆。
2. 用软 P 解 shared 的最小二乘问题，初始化 L=0、S=E-native_shared、U=0，
   保留原 streaming 基初始化。此时 Q=E，不改变完整 expert 权重。
3. 每步任务梯度经 DP 平均后，加入一次 `rho * (E-Q)`，再裁剪并执行原 AdamW。
4. 每轮结构更新先固定 `R = E-L_old-S_old+U_old`。P 子问题优化三投影真实重构误差
   加变化惩罚；使用 log 域 Sinkhorn，并通过其梯度更新 A，连续做 8 步。
   Gram/cross 矩阵只是同一平方误差梯度的计算优化，没有替换成线性 OT 距离目标。
5. 固定新的 P，求 `M = sum(P @ P.T)`。gate/up 解
   `M @ shared = sum(P @ R)`；down 解 `shared @ M = sum(R @ P.T)`。
   使用一次 Cholesky 分解求解三个投影，保留 P_0=I，不加新的 ridge。
6. 顺序更新 L、S、U、控制器及 Q。所有层候选更新成功并经 DP 协商后才提交状态；
   非有限数或 Sinkhorn 未满足行列和容限会使本轮失败，保留原辅助状态和 Q。

数值操作与辅助状态使用 FP32，关闭这些矩阵乘法中的 TF32。P 的约定为
`P[expert, shared_channel, native_channel]`，gate/up 使用 `P.T @ shared`，
down 使用 `shared @ P`。P_0=I 在软对应中是一项建模约束，不只是重新编号。

## 保存、恢复和导出

Checkpoint 保存分数表 A、实际 P、原有全部辅助状态和最新更新时间；
同层三个投影恢复后共用同一份 A/P。加载时核对形状、精度、非负性、行列和、
参考专家、三投影一致性，以及保存的 A 能否生成保存的 P。

既有未对齐与硬匹配 checkpoint 仍按原模式读取。软 P 使用
`salaad_moe.export.v4`，保存 FP32 P，重构为 `native_shared + L + S`，不含 U；
不会把软 P 四舍五入为硬置换。v2/v3 导出仍可读取。

## 本地检查和启动

只读取正式配置：

~~~bash
myenv/bin/python scripts/train_salad.py --cfg_version ns97m_sinkhorn --dry-run
~~~

CPU 小模型（第 3 步初始化，第 5、7、8 步同时更新 P/L/S，包含最终补更新）：

~~~bash
myenv/bin/python scripts/train_salad.py --cfg_version smoke_sinkhorn \
  --device cpu --output /tmp/new_sinkhorn_smoke --stop-after 5
myenv/bin/python scripts/train_salad.py --cfg_version smoke_sinkhorn \
  --device cpu --resume /tmp/new_sinkhorn_smoke/checkpoints/step_00000005
~~~

单元测试与实际双进程训练、恢复检查：

~~~bash
myenv/bin/python -m unittest discover -s tests/moe -p 'test_*.py' -q
myenv/bin/torchrun --standalone --nproc-per-node=2 \
  tests/moe/alignment_distributed_worker.py /tmp/new_sinkhorn_dp2 --sinkhorn
~~~

正式训练入口（需先同步代码并由集群调度分配 GPU）：

~~~bash
myenv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_salad.py \
  --cfg_version ns97m_sinkhorn --data-manifest /path/to/manifest.json \
  --output /path/to/new_run
~~~

与此前先做 100 步 vanilla、之后每 200 步更新硬 P 的实验相比，本配置同时改变了
P 的表示、初始化时机与更新频率。
若要单独判断软对应的作用，需另做 P 更新时机相同的硬匹配对照。
已有三个正式作业的配置不随本实现改变。

通俗解释和推导见
[PDF](../discussion_notes/moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.pdf)。

2026-09-22 验证：88 项单元测试、DP2 训练及逐位恢复、CUDA BF16 CLI 暂停恢复均通过；
另检查了 64 专家、176 通道、256 hidden size 的单层 CUDA 数值更新。
详细记录见 [验证结果](moe_sinkhorn_validation.json)。这些检查验证实现，不代表正式训练质量。
