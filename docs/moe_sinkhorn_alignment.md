# MoE 自由残差 ADMM 与 Sinkhorn 对齐

配置入口是 `--cfg_version ns97m_sinkhorn`。同层 gate/up/down 共用一份软 P，
完整 expert 权重 W 用于任务前向。当前模式对应
[ADMM 文档](../discussion_notes/moe_salaad_admm_20261005/moe_salaad_admm.tex)的五步公式：
W → P → 共享矩阵 X → 自由残差 X_e → 乘子 Y。
代码保存缩放乘子 `U = Y / rho`，与文中未缩放的乘子更新等价。
PyTorch 权重使用 `[expert, output, input]` 布局，是文中权重矩阵的转置布局。

正式配置在训练开始前初始化分解和 P，不保留 vanilla 前缀。
从第 10 步开始，每 10 步依次更新 P、shared、residual、U。
2026-10-05 固定 P 的更新规则为 **无约束闭式解 → 正下限截断 → Sinkhorn**，每轮执行一次。
结束时若需要补一次结构更新，P 也一起更新。总预算仍为 2100 步。

## 新增设置

以下为当前正式配置。任务训练参数、rho=1e-5 和结构更新间隔沿用现有 97M 配置。
训练器按 optimizer step 调度；这里每 10 个权重优化步骤对应一次 ADMM 外层更新，
没有新增按完整数据集遍历计数的 epoch 调度。

| 配置项 | 值 | 含义 |
|---|---:|---|
| `residual_mode` | `dense` | 使用一个完整的专家残差 X_e |
| `low_rank_enabled` / `sparse_enabled` | false / false | 关闭 L/S 分解 |
| `controller` | null | 不建立阈值控制器或 streaming SVD 基 |
| `channel_alignment.fix_reference` | false | 所有专家的 P 均参与更新 |
| `initialization` | `identity_shared_mean_residual_dual_zero` | P 为单位矩阵、X 为均值、X_e 为差值、U 为零 |
| `update_rule` | `closed_form_clip_sinkhorn` | 固定的三步更新规则 |
| `clip_min` | 1e-8 | 闭式解逐元素截断的正下限 |
| `temperature` | 1.0 | 保存 A=T log(Q)，使 exp(A/T)=Q |
| `max_iterations` | 150 | 后续 P 更新中 Sinkhorn 行列归一化的迭代上限，达标即停止 |
| `marginal_tolerance` | 1e-5 | 最大行/列和误差容限 |

P 更新固定本轮旧 shared 和残差，直接计算新的结果。
已移除 P 的梯度步数、学习率和旧 P 移动惩罚配置；配置校验会拒绝这些旧字段。
A 只保存截断结果的对数，不再作为梯度优化变量，也不从旧 A 热启动。

令 X 表示共享权重，R 表示本轮拟合目标，gate/up/down 是三个投影。
闭式法方程为 `M @ P_LS = C`，其中：

~~~text
M = X_gate @ X_gate.T + X_up @ X_up.T + X_down.T @ X_down
C = X_gate @ R_gate.T + X_up @ R_up.T + X_down.T @ R_down
Q = clamp_min(P_LS, 1e-8)
P_new = Sinkhorn(Q)
~~~

实现将 `H = [X_gate, X_up, X_down.T]` 和
`D = [R_gate, R_up, R_down.T]` 沿列拼接，计算
`P_LS = pinv(H.T) @ D.T`。H 满行秩时等价于解上述法方程；
秩不足时取最小范数解。直接对拼接矩阵求伪逆，避免先形成 Gram 矩阵放大条件数。
同层专家共用一次 SVD 分解；使用 PyTorch FP32 伪逆的默认奇异值截断规则，不增加 ridge。
得到 Q 后令 `A = temperature * log(Q)`，一次调用 log 域 Sinkhorn 生成并保存 P。
结果直接接收，不附加重构误差下降筛选。

## 执行顺序

W 使用基础 MoE 模型原有的初始化。每层所有专家直接取 `P_e = I`，
每个投影的 `shared = mean(W, dim=0)`，`residual = W - shared`，`U = 0`。
这时训练目标 `anchor` 与 W 一致（允许浮点舍入误差）。
初始化不执行硬匹配、软化、Sinkhorn 或共享矩阵最小二乘求解，也不额外抽取随机数。
继承配置中的硬匹配参考专家等参数只供旧初始化使用，当前模式不会固定某个专家的 P。

精确单位矩阵直接保存为 P，初始化阶段的 `alignment_logits` 为 None，checkpoint 不写该字段。
第一次结构更新仍执行闭式解、正下限截断和 Sinkhorn，并从截断结果生成有限的 A。
从单位矩阵开始时，CPU 训练检查中的一次归一化需要 120 次迭代；
按当前设定，迭代上限为 150，行列和误差容限仍为 1e-5。
150 次不是收敛保证：正式矩阵尺寸的另一次 CPU 短序列检查在第 10 步、
第 3 层达到上限时，最大行/列和误差约为 3.0e-5，因此按当前规则停止。
最小二乘和归一化全部改用 FP64 作独立复核后，该输入在第 175 次检查才达到 1e-5。
生产配置保持 150；到达上限仍未达标时，报错会列出实际误差、容限和迭代次数。

1. **W**：固定辅助状态，每个权重优化步骤在 DP 平均后加入一次
   `rho * (W - anchor)`，其中 `anchor = native_shared + residual - U`，然后执行优化器。
2. **P**：固定 `R = W_new - residual_old + U_old` 和旧共享矩阵，对每个专家联合求
   无约束最小二乘解，截断到 1e-8，再执行 Sinkhorn。三个投影共用同一个 P。
3. **X**：固定新的 P，计算 `B = sum(P @ P.T)`。gate/up 使用
   `shared = pinv(B) @ sum(P @ R)`；down 使用
   `shared = sum(R @ P.T) @ pinv(B)`。允许 B 奇异，取最小范数解。
4. **X_e**：`residual_new = W_new - native_shared_new + U_old`。
5. **Y**：在缩放单位下计算
   `U_new = U_old + (W_new - native_shared_new - residual_new)`，然后刷新 anchor。

自由残差使第五步的 U 在精确算术下等于零；代码仍执行上述公式并保留 FP32 舍入结果。
此模式不对残差施加低秩或稀疏约束。所有层的候选状态通过 DP 协商后才提交；
非有限数或 Sinkhorn 未满足行列和容限会保留原辅助状态和 anchor。
候选训练目标 `anchor` 也检查有限性，防止各分量有限但相加后溢出。
结构更新发生在优化器更新之后；该阶段报错时训练立即终止，需要从完整 checkpoint 恢复，
不能在同一内存状态上直接重试一个训练步。

数值操作与辅助状态使用 FP32，关闭这些矩阵乘法中的 TF32。P 的约定为
`P[expert, shared_channel, native_channel]`，gate/up 使用 `P.T @ shared`，
down 使用 `shared @ P`。

## 保存、恢复和导出

Checkpoint 标记 `residual_mode: dense`，保存 shared、residual、U、实际 P 和最新更新时间；
执行过 P 更新后还保存分数表 A。同层三个投影恢复后共用同一份 P 和可选的 A。
加载时核对形状、精度、非负性、行列和及三投影一致性。
只有采用单位矩阵初始化且尚未执行结构更新的 checkpoint 才允许省略 A，此时所有 P 必须严格等于 I。
其他 Sinkhorn checkpoint 必须保存 A，并通过 A 重建 P 的一致性检查。
推理和导出读取 checkpoint 时，同时核对完成标记、训练步数、配置哈希和实际配置；
加载导出文件时检查共享矩阵及最终重构权重的有限性。

新导出使用 `salaad_moe.export.v5`，保存完整 FP32 残差，重构为
`native_shared + residual`，不含 U。不会把软 P 四舍五入为硬置换。
日志报告重构误差、残差范数、未缩放乘子范数和 rho。

其他配置默认仍使用 `residual_mode: low_rank_sparse`。没有 residual_mode 元数据的旧辅助状态
按 L/S 模式读取，既有 v2/v3/v4 导出保持兼容。新旧残差状态不能混用；
旧 L/S checkpoint 不能直接用当前 dense 配置续训。

## 本地检查和启动

只读取正式配置：

~~~bash
myenv/bin/python scripts/train_salad.py --cfg_version ns97m_sinkhorn --dry-run
~~~

CPU 小模型（第 3 步初始化，第 5、7、8 步更新 P/X/X_e/Y，包含最终补更新）：

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
P 的表示、初始化时机、更新频率以及专家残差模型。
已有三个正式作业的配置不随本实现改变。

当前三步规则见问答文档的问题 7：
[PDF](../discussion_notes/moe_salaad_qa_20261004/moe_salaad_qa.pdf) /
[LaTeX](../discussion_notes/moe_salaad_qa_20261004/moe_salaad_qa.tex)。
旧梯度版本保留在 [历史 PDF](../discussion_notes/moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.pdf)。

2026-10-05 上限 150 的复查：101 项单元测试、正式配置 dry-run，以及 DP2 八步训练和逐位恢复通过。
新增回归用例复现并验证三处修复：推理/导出缺失 checkpoint 元数据一致性检查、
导出共享矩阵/重构权重的非有限值检查，以及更新/恢复时训练目标相加溢出的拒绝。
另用实际不能在 150 次内达标的矩阵验证严格迭代上限及失败后辅助状态保持不变。

额外检查保持正式模型的 8 层、64 专家、176 中间通道、256 hidden size 和 BF16 计算，
使用 CPU、合成数据、batch=1、sequence length=8，以减小验证成本。
该检查在第 10 个优化器步的第 3 层触发上述不收敛错误；它不是一次通过的正式训练验证。
本次本机 CUDA 驱动不可用，未重新验证 GPU 路径。

2026-10-05 较早的单位矩阵初始化版本：98 项单元测试、正式配置 dry-run，以及 DP2 训练和逐位恢复通过。
新增检查确认 P 严格等于 I、X 为原权重均值、X_e 为差值、U 为零、W 和随机数状态不受辅助初始化影响，
以及第一次结构更新之前的 checkpoint 恢复和 BF16 导出；恢复也覆盖第一次 P 更新之后的状态。

2026-10-05 较早的自由残差版本：96 项单元测试、正式配置 dry-run，以及 DP2 训练和逐位恢复通过。
新增检查覆盖文档五步公式、所有专家的 P 更新、奇异共识矩阵、非零初始乘子、
完整残差保存/恢复、BF16 导出，以及新旧残差状态混用的拒绝。

2026-10-05 较早的 L/S 三步 P 更新版本：90 项单元测试、正式配置 dry-run，以及 DP2 训练和逐位恢复通过，
覆盖闭式解、负元素截断、秩不足及全零共享矩阵、保存恢复、BF16 导出和失败时的状态保留。

2026-09-22 的旧梯度版本验证：88 项单元测试、DP2 训练及逐位恢复、CUDA BF16 CLI 暂停恢复均通过；
另检查了 64 专家、176 通道、256 hidden size 的单层 CUDA 数值更新。
详细记录见 [验证结果](moe_sinkhorn_validation.json)。这些检查验证实现，不代表正式训练质量。
