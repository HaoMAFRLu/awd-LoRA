# MoE 通道对齐训练

入口仍是 `scripts/train_salad.py`。选择 `--cfg_version ns97m_aligned` 启用已讨论的算法。
`ns97m` 保留为不对齐的 consensus 对照，`ns97m_independent_ls` 保留为 independent L+S 对照。

前向一直使用完整 X。排列表 `permutation[i, a] = b` 表示：专家 i 的原通道 a 对应
shared 的位置 b。gate/up 的行与 down 的列共用这张表。X、L、S、U、Adam 状态和
streaming SVD 的迭代基都保持原顺序；只有 H 使用共同编号。

## 执行顺序

1. 初始化时，以专家 0 为模板，最多做 10 轮联合匹配和对齐平均。差异同时包含
   gate 行、up 行和 down 列的平方差，用 SciPy 的 `linear_sum_assignment` 求一对一指派。
   专家 0 的排列固定。设置 L=0、S=X−对应后的 H、U=0，用一次 SVD 准备迭代基。
2. 每步累积任务梯度、DP 平均，加一次 `rho * (X-Q)`，裁剪后执行 AdamW。
   Q 是对应回专家原顺序后的 `H + L + S - U`。
3. 每 10 步计算 `R=X-L_old-S_old+U_old`。若也是 100 的倍数，将 R 的三个投影
   联合匹配当前 H。只有差异降低超过 `1e-6 * max(1, 原差异)` 才接受候选排列。
4. 对齐 R 后平均得到新 H；将 H 对应回各专家，依次更新 L、S、U、阈值和 Q。
   L 使用既有 streaming 更新，新阈值在下一轮生效，允许负阈值。目标秩比/密度
   固定为 0.15/0.10，不提前冻结控制器。
5. `layer_index % DP` 决定该层的结构负责人，三个投影一起更新并广播各自的 Q。
   每 100 步和结束时保存完整 checkpoint；最终结构更新只执行一次。

训练期间不新增 validation 或 W&B 指标。匹配或结构更新失败时，候选状态不会提交，
失败会传到全部 rank。主要代码为 [alignment.py](../salaad_moe/alignment.py)、
[solver.py](../salaad_moe/solver.py) 和 [export.py](../salaad_moe/export.py)。

## 配置与运行

[ns97m_aligned.yaml](../configs/ns97m_aligned.yaml) 继承原 97M 的模型、数据、AdamW
与预算：DP4、2100 步、rho=1e-5、结构间隔 10。新增设置为：

```yaml
salaad:
  auxiliary_owner: deterministic_layer_id_mod_dp
  structure_order: [permutation, shared, low_rank, sparse, dual]
  channel_alignment:
    enabled: true
    match_every_optimizer_steps: 100
    initialization_max_iterations: 10
    improvement_tolerance: 1.0e-6
    reference_expert: 0
```

把 `match_every_optimizer_steps` 设为 0 可只做初始化对齐，随后固定 P；正数必须是
结构间隔的整数倍。对齐要求全部三个投影和 learned shared，目前从 step 0 初始化，
每次做一轮结构更新。SciPy 已加入 `requirements.txt`。距离使用 FP32 计算，指派在 CPU 求解。

检查正式配置，不启动训练：

```bash
myenv/bin/python scripts/train_salad.py --cfg_version ns97m_aligned --dry-run
```

在准备好语料的四卡机器上启动：

```bash
myenv/bin/torchrun --standalone --nproc-per-node=4 scripts/train_salad.py \
  --cfg_version ns97m_aligned --data-manifest /path/to/manifest.json \
  --output /path/to/new_run
```

[smoke_aligned.yaml](../configs/smoke_aligned.yaml) 用于本地软件检查：每 2 步更新结构，
每 4 步检查配对，总共 8 步，关闭 W&B。以下输出目录需尚不存在：

```bash
myenv/bin/python scripts/train_salad.py --cfg_version smoke_aligned \
  --device cpu --output /tmp/moe_aligned_example --stop-after 3
myenv/bin/python scripts/train_salad.py --cfg_version smoke_aligned \
  --device cpu --resume /tmp/moe_aligned_example/checkpoints/step_00000003
```

双进程配置为 [smoke_aligned_dp2.yaml](../configs/smoke_aligned_dp2.yaml)。Python API 恢复时，
构造 `Trainer(..., initialize_auxiliary=False)`，随后调用 `load_checkpoint` 或
`run(..., resume=...)`。CLI 会自动设置，直接恢复 P 和迭代基，不重做初始匹配或 SVD。

## 保存、导出与验证

Checkpoint 保存完整 X、Adam、数据游标、RNG、H/L/S/U、阈值、迭代基、P，及最近
结构更新和匹配步数。每个投影记录相同的排列表和自身通道轴；加载时检查排列是一对一、
三个投影一致、参考专家保持恒等排列，随后在内存中共用一份 P。旧的不对齐 checkpoint
按原语义加载，不会自动转换为对齐算法。

对齐导出使用 `salaad_moe.export.v3`，保留 H、P、完整 L 和原精度的全部 S 非零值，
记录 `native_to_shared` 排列方向。不对齐导出仍为 v2，两种格式均可读取。
重构为“对应回原顺序的 H + L + S”，**不含 U**。现有导出和评估命令用法不变。

```bash
myenv/bin/python -m unittest discover -s tests/moe -p 'test_*.py' -q
myenv/bin/torchrun --standalone --nproc-per-node=2 \
  tests/moe/alignment_distributed_worker.py /tmp/new_alignment_dp_test
```

测试覆盖已知排列恢复、穷举指派与显式置换矩阵、完整结构更新、BF16 任务计算下的
FP32 状态、恢复时不重新初始化、checkpoint/导出中的排列检查和原顺序重构。
CPU 单进程和双进程 8 步训练中，中途保存后继续训练与恢复训练的模型、Adam、
辅助状态、数据游标和 RNG 逐位相同。双进程还覆盖空 owner、匹配失败传播和 Q 广播。

当前环境没有可用 CUDA，尚未完成 H100 检查或正式 DCLM 训练；这些小模型检查用于验证实现。
