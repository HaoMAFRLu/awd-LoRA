# MoE-SALAAD

本分支提供独立的 MoE 训练框架，以 SALAAD 约束专家权重的共享、低秩和稀疏结构：

```text
X_i = X_hat + L_i + S_i
```

每一层的 gate/up/down 分别建立专家组。前向使用完整权重 X，AdamW 更新模型；
SALAAD 独立更新 X_hat/L/S/dual。关闭 salaad.enabled 即得到同训练器下的 vanilla MoE。

## 从哪里读代码

先读 [中文代码阅读指南](docs/moe_training_walkthrough.md)，再按下面顺序查看：

1. [scripts/train_salad.py](scripts/train_salad.py)：MoE 专用入口，显式 __main__ → main()。
2. [salaad_moe/trainer.py](salaad_moe/trainer.py)：任务梯度、DP 同步、优化器和训练循环。
3. [salaad_moe/solver.py](salaad_moe/solver.py)：SALAAD 约束梯度、ADMM 和阈值控制器。
4. [salaad_moe/model.py](salaad_moe/model.py)：attention、router、top-k experts 和任务损失。
5. [salaad_moe/runtime.py](salaad_moe/runtime.py) / [data.py](salaad_moe/data.py)：设备、运行目录与 token 数据。
6. [checkpoint.py](salaad_moe/checkpoint.py) / [tracking.py](salaad_moe/tracking.py)：恢复训练和 W&B。

scripts/train_moe.py 只提供同一 main() 的命令别名。MoE 训练不经过旧 salad/、
models/、dataloaders/ 或 salaad_vision/ 的训练入口；旧研究文件仍保留在仓库中。

## 本地启动

使用现有 myenv；核心训练依赖 PyTorch、NumPy、PyYAML 和 wandb。
正式语料准备和下游任务另需 tokenizer/数据处理依赖，见 [完整运行说明](docs/moe_salaad.md)。

VS Code 已提供 `MoE: local DCLM smoke` 调试配置：直接按 F5，无需输入参数。
固定使用 `myenv/bin/python` 启动 `scripts/train_salad.py`，可在 `main()` 内设置断点。
默认读取真实 DCLM 本地切片，运行 8 步、BF16、`rho=0.05`，W&B 离线；
优先使用 CUDA，否则使用 CPU。每次运行自动创建独立的时间戳输出目录。
单进程也会初始化真实的分布式进程组（rank 0、world size 1），因此调试会进入
`dist.broadcast` / `dist.all_reduce` 等通信分支，无需额外设置地址或端口。

```bash
# Default: local DCLM slice, single-GPU BF16, eight optimizer steps.
myenv/bin/python scripts/train_salad.py

# Synthetic single-GPU BF16 smoke.
myenv/bin/python scripts/train_salad.py --cfg_version smoke_bf16

# CPU smoke。
myenv/bin/python scripts/train_salad.py --cfg_version smoke --device cpu

# 双进程 CPU DP。
myenv/bin/torchrun --standalone --nproc-per-node=2 scripts/train_salad.py \
  --cfg_version smoke_dp2 --device cpu

# 只查看配置和参数量。
myenv/bin/python scripts/train_salad.py --cfg_version ns97m --dry-run
```

W&B 默认启用，项目名 SALAAD-MoE，只由 rank 0 创建 run。
默认 smoke 配置使用离线模式；正式在线运行使用已有登录或 WANDB_API_KEY。

## 配置

所有运行配置直接位于 configs/，模型结构也在 YAML 内，无需额外的模型 JSON。
当前入口仅接受 MoE 配置。

| 配置名 | 用途 |
|---|---|
| smoke_bf16 | 随机 token 单 GPU BF16 调试 |
| smoke_dclm_bf16 | 默认：真实 DCLM 小切片，Pythia 词表，单 GPU BF16，W&B 离线 |
| smoke | 单进程 FP32 小模型 |
| smoke_dp2 | 双进程 DP 小模型 |
| ns97m | 97M SALAAD，DP2 |
| ns97m_vanilla | 同条件 97M vanilla |
| ns97m_independent_ls | 不使用共享矩阵的 L+S 消融 |
| ns480m | 480M SALAAD，DP4 |

默认继承链是 smoke_dclm_bf16.yaml → smoke_bf16.yaml → smoke.yaml → ns97m.yaml。
SALAAD 默认在第 0 步、首次参数更新之前初始化：shared 为专家权重均值，
L 为零，S 为各专家相对均值的残差，U 为零。smoke 在第 2、4、6、8 步更新结构。
YAML 按 model、data、training、parallel、salaad、export 划分职责。
正式训练需要先用 scripts/prepare_moe_data.py 准备 token manifest。

默认启动会使用仓库中的 `data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json`。
也可以显式指定配置和数据路径：

```bash
myenv/bin/python scripts/train_salad.py --cfg_version smoke_dclm_bf16 \
  --data-manifest data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json
```

该语料包含 1,048,576 个训练预测 tokens，配置仍只运行 8 步。
数据准备位置与下载记录见 [DCLM 数据记录](docs/moe_dclm_data.md)。

## 每一步训练

```text
任务 loss backward（梯度累积）
  → DP 平均任务梯度
  → 加一次 rho * (X - X_hat - L - S + U)
  → 全局梯度裁剪
  → AdamW
  → 到周期时更新 X_hat / L / S / U
  → 推进数据游标，验证、保存、记录
```

DP 每个 rank 都有完整模型。SALAAD 的组 owner 只分摊辅助状态和 SVD，并广播约束目标 Q。
BF16 配置使用 autocast；参数、梯度、Adam 和结构辅助状态保持 FP32。

## 输出和恢复

默认输出在 data/<folder>/<cfg_version>/<timestamp>/：

- config.resolved.json：完整继承配置。
- run_metadata.json：数据身份、运行环境和恢复来源。
- metrics.jsonl、wandb_run.json：本地指标与 W&B run 身份。
- checkpoints/step_xxxxxxxx/：模型、Adam、全局数据游标、各 rank 的 SALAAD 状态和 RNG。

--resume 恢复完整 checkpoint；--branch-from 从相同 vanilla 前缀建立 SALAAD 分支。
--stop-after N 与 --num_total_iters N 表示到第 N 步暂停，完整调度仍由 YAML 决定。

模型导出使用 scripts/export_moe.py；NLL/PPL 和下游任务分别使用
scripts/evaluate_moe.py 与 scripts/evaluate_moe_tasks.py。完整命令和验证边界见
[运行说明](docs/moe_salaad.md)。
导出保留训练得到的 shared/L/S 原值和精度，S 仅无损转换为 CSR；没有额外的秩或密度上限。

## 测试

```bash
myenv/bin/python -m unittest discover -s tests/moe -v
```

覆盖模型、SALAAD、数据、checkpoint、导出、日志和入口。smoke 结果用于软件验证，
不代表完整预训练质量或 H100 性能。

## License

见 LICENSE。
