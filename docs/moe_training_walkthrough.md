# MoE + SALAAD 代码阅读指南

当前主入口只训练 MoE。SALAAD 是结构约束框架，负责让同层同投影的专家权重接近
`X_i = X_hat + L_i + S_i`。模型任务训练、数据和分解状态分别组织，不使用旧 LLM/Vision 的注册器或训练器。

## 先看这条调用路径

```text
scripts/train_salad.py
  __main__: 显式读取参数、构造配置路径、调用 main(...)
  main():   配置 -> 设备/DP -> 数据 -> Trainer -> run()
                                |
salaad_moe/trainer.py            |
  Trainer.run() <----------------+
    恢复 checkpoint（可选）
    _run_steps(): 训练 -> 验证 -> 保存 -> 日志
      train_step(): 一个 global batch 的完整更新
        _backward_task(): 累积任务梯度
        average_gradients(): DP 平均
        manager.inject_gradients(): 加 SALAAD 约束
        clip_grad_norm_() -> AdamW.step()
        manager.after_step(): 按周期更新结构
                                |
salaad_moe/solver.py             |
  ConsensusManager <------------+
    initialize(): 初始化 shared/L/S/U
    structure_sweep(): shared -> L -> S -> U
    refresh_anchors(): 将 Q 广播给所有 ranks
```

`scripts/train_moe.py` 只是同一 `main()` 的命令别名，没有第二套训练流程。

| 文件 | 负责什么 |
|---|---|
| `scripts/train_salad.py` | 唯一主流程、CLI 和公共调试断点 |
| `salaad_moe/runtime.py` | 设备/DP 初始化、统一输出目录、准备语料 |
| `salaad_moe/model.py` | Attention、router、SwiGLU experts、任务损失 |
| `salaad_moe/data.py` | token manifest、next-token 标签、全局样本顺序 |
| `salaad_moe/trainer.py` | AdamW、梯度顺序、训练循环与评估 |
| `salaad_moe/groups.py` | 将每层 gate/up/down 权重暴露为专家矩阵组 |
| `salaad_moe/solver.py` | SALAAD 辅助状态、共识 ADMM、阈值控制器 |
| `salaad_moe/checkpoint.py` | 完整保存/恢复模型、Adam、SALAAD、数据游标和 RNG |
| `salaad_moe/tracking.py` | rank 0 的 W&B 日志和 run 恢复 |

可选 Megatron 适配文件不属于这条主流程，阅读原生 DP 训练时无需查看。

## 一个训练 step 为什么这样执行

开始训练前，`Trainer.__init__()` 默认调用 `manager.initialize(0)`。
这里的 0 表示尚未完成任何 optimizer 更新：用初始专家权重设置
`shared=mean(X)`、`L=0`、`S=X-shared`、`U=0`，并准备约束目标 Q。
默认 `state_initialization_step=0`；smoke 随后在第 2、4、6、8 步更新结构。

1. 每卡取自己的 micro-batches，计算 LM NLL、load-balancing loss 和 router z-loss。
   每次 backward 的 loss 除以累积次数，累积完成后对应一个 batch mean。
2. 在所有 DP ranks 间平均任务梯度。这里显式调用 all-reduce，不使用自动同步梯度的 DDP wrapper。
3. SALAAD 加入一次 `rho * (X - Q)`，其中 `Q = X_hat + L + S - U`，`U = Y/rho`。
   所有专家都参与这一约束，未被路由访问的专家也不例外。
4. 裁剪任务与结构相加后的梯度，然后执行 AdamW。
5. 达到结构更新周期时，固定刚更新的 X，对辅助变量执行一次或多次 ADMM。
6. 全部成功后推进 step 和全局数据游标，记录指标。失败时中止，从完整 checkpoint 恢复。

在梯度累积的每个 micro-batch 都加 penalty，会重复计入约束；在结构更新时直接覆盖 X，
则会改变当前算法。代码在这些位置有对应注释。

## 哪些参数在训练

模型前向始终使用完整的 dense X，AdamW 更新它。Embedding、attention、router 和 norm
也参与任务训练，但当前 SALAAD 只作用于 expert 的 gate/up/down。

`decay` 收集二维矩阵及三维 expert 参数，使用配置中的 weight decay。
`no_decay` 收集 norm 缩放等一维参数，weight decay 为 0。两组参数都会更新。
BF16 配置使用 autocast；参数、梯度、Adam 状态和 SALAAD 辅助状态仍为 FP32。

辅助变量不参与 autograd：`shared` 为 `[out, in]`，L/S/U 为 `[experts, out, in]`。
结构更新分别执行共识均值、奇异值阈值、逐元素阈值和对偶累积；控制器根据有效秩/非零密度
调整下一轮阈值。target 是优化目标，不等于保证最终达到的硬 rank/density。
`tau_l/tau_s` 使用有符号积分更新，允许变为负数；奇异值收缩和稀疏幅度阈值算子
中的 `.clamp_min(0)` 仍保留。checkpoint 会保存并恢复这些有符号阈值。

低秩更新使用 SALAAD++ 的 `streaming_svd_step()`：复用上次的 `svd_basis`，
做一次矩阵乘法和 QR，投影列范数作为近似奇异值，再执行阈值操作。
初始化时用一次普通 SVD 准备基；每次结构更新返回新基，供下一轮使用。
`svd_basis` 是 `[experts, k, k]`，`k=min(out,in)`，它和约束目标 Q 是两个不同变量。
基随辅助状态保存和恢复，因此暂停后能继续相同的 streaming 迭代。
训练没有 driver/fallback 选项；`svd_chunk_size` 只控制每批处理多少个 expert。
导出直接保存训练得到的 shared/L，以及 S 的全部非零值，保留原精度。
导出不再执行 SVD、rank/density 截断或额外精度转换。

## DP 如何分工

所有 ranks 都保存完整模型并处理不同样本。第 g 个矩阵组的辅助状态由 `g % world_size`
对应的 rank 保存和更新。owner 更新后广播 Q，因此模型副本加入相同的结构梯度。
这分摊了 SVD 和辅助状态，不会把模型前向按层分到不同 GPU。

## 本地运行和调试

```bash
# Default: local DCLM slice, small model, single-GPU BF16, eight steps.
myenv/bin/python scripts/train_salad.py

# Synthetic CPU smoke test.
myenv/bin/python scripts/train_salad.py --cfg_version smoke --device cpu

# Two CPU processes to verify data parallelism.
myenv/bin/torchrun --standalone --nproc-per-node=2 scripts/train_salad.py \
  --cfg_version smoke_dp2 --device cpu

# Inspect the configuration and parameter counts without creating a model.
myenv/bin/python scripts/train_salad.py --cfg_version ns97m --dry-run
```

VS Code 选择 `MoE: local DCLM smoke` 后直接按 F5；调试配置的 `args` 为空，
固定使用 `myenv/bin/python` 和 `scripts/train_salad.py`。其他 IDE 也只需选择这两个路径，
不需要输入参数。在 `main()` 第一条语句设置断点。
想检查每一步训练，进入 `Trainer.train_step()`；想检查 ADMM，进入 `structure_sweep()`。
本地默认真实 DCLM 切片、8 步、BF16、`rho=0.05`、4 个 CPU 线程，优先使用 CUDA，
没有 CUDA 时使用 CPU。W&B 离线记录到 `SALAAD-MoE`，输出自动放入新的时间戳目录。
即使单进程也初始化真实的进程组：`rank=0`、`world_size=1`，CUDA 使用 NCCL，
CPU 使用 Gloo。单进程的初始化使用进程内 `HashStore`，无需填写 `MASTER_ADDR/MASTER_PORT`；
`dist.is_initialized()` 在训练中为 `True`，广播和归约分支会实际执行。
训练退出时由 `main()` 销毁进程组。多进程仍使用 `torchrun` 提供的初始化环境。
`rho=None` 和 `num_total_iters=None` 表示使用 YAML 中的值；`output=None` 表示自动选目录，
`resume=None` 和 `branch_from=None` 表示新建实验，这些都不需要手动输入。

配置仍放在 `configs/` 根目录。默认继承链是
`smoke_dclm_bf16.yaml -> smoke_bf16.yaml -> smoke.yaml -> ns97m.yaml`。
正式数据需要先准备 token manifest；完整命令见 [运行说明](moe_salaad.md)。
`--stop-after N` 与 `--num_total_iters N` 都表示到第 N 步暂停，完整 LR/结构调度始终来自 YAML。
`--resume` 恢复全部状态；`--branch-from` 从共享的 vanilla 前缀开始 SALAAD 分支。

## 比较 vanilla 和 SALAAD

保持模型、初始化、数据顺序、batch、优化器和预算一致，仅把 `salaad.enabled` 设为 false
即可得到 vanilla MoE。已有 `ns97m_vanilla.yaml` 继承 `ns97m.yaml` 实现这一设置。
训练和 raw 验证使用 X；reconstructed 验证在模型副本中使用 `X_hat + L + S`，不影响训练状态。
