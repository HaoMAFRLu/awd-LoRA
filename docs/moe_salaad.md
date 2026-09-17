# MoE-SALAAD 代码与运行方法

本实现对应同一层、同一 projection 内的
`X_i = X_hat + L_i + S_i`。模型训练完整的 dense expert 权重 `X_i`，
`X_hat/L/S/scaled_dual` 是辅助状态，模型中没有 always-on shared expert。
当前入口只服务 MoE，已移除旧 dense LLM/Vision 的配置分流和训练器依赖。
阅读代码建议先看 [MoE 训练流程与注释指南](moe_training_walkthrough.md)。

## 当前默认与对照原则（2026-09-17）

后续讨论和实验默认普通 DP，`EP=TP=PP=CP=1`。vanilla MoE 和 SALAAD 都使用
`scripts/train_salad.py --cfg_version <名称>`，从 `configs/<名称>.yaml` 读取配置，
先进入 `scripts/train_salad.py` 的 `main()`，准备运行环境和语料后直接创建
`salaad_moe.trainer.Trainer`。`scripts/train_moe.py` 只导入同一个 `main()` 作为命令别名，
没有独立后端或跨脚本回调。调试时可在 `main()` 的首条语句设置公共断点。
Megatron 是保留的可选适配路径，不是当前训练的前置条件。
当前目标是与同训练器的 vanilla 对齐，参考论文只提供模型规模等背景。

不传启动参数时，`parse_args()` 默认选择 `smoke_dclm_bf16`，数据路径为仓库下的
`data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json`。显式数据路径可覆盖默认值；
选择其他配置或恢复 checkpoint 时，保留配置或原实验的数据选择。

配对实验固定模型初始化、数据顺序、tokenizer/packing、DP 数、micro batch、梯度累积、
全局 batch、精度、重计算、AdamW、LR 调度、梯度裁剪、token 预算和评估协议。
vanilla 保留相同的 router、balance loss 和 z-loss；SALAAD 只增加分解约束与辅助更新。
单独报告由结构诊断、SVD 和同步带来的时间、峰值显存及存储开销。

先验证 97M，再扩展至 480M；614M 尚无独立配置，2.5B 不进入当前初步验证。
在本地工作区可查阅 [完整讨论记录](../discussion_notes/MOE_SALAAD_DISCUSSION_LOG.md)。
`discussion_notes/` 按仓库规则不跟踪，当前运行要求同时保存在本节。

## 入口与验证范围

| 入口 | 用途 | 本次验证 |
|---|---|---|
| `scripts/train_salad.py` | MoE 专用 `main()`，按配置直接创建 MoE Trainer | 根目录配置读取、双进程 CPU、单 GPU BF16、真实 W&B SDK 离线日志 |
| `scripts/train_moe.py` | 同一 `main()` 的命令别名 | 公共断点、配置读取和暂停/恢复 |
| `scripts/train_moe_megatron.py` | 可选的固定 FLAME/Megatron DP 适配入口 | 原生参数映射、FP32 master hook、QKV/SwiGLU/norm 转换协议测试；尚未运行完整 Megatron CUDA 环境 |
| `scripts/prepare_moe_data.py` | 去重、划分、token packing、可选 Megatron indexed 输出 | 本地真实文本、tokenizer JSON、重复文档、标签位移、文件完整性 |
| `scripts/export_moe.py` | 原精度 shared/L + 全部 S 非零值的 int32 CSR 文件 | 零 L、小分量保留、实际文件读取与无损重构 |
| `scripts/evaluate_moe.py` | raw/reconstructed/exported 的 token NLL/PPL | 三种路径均有小模型测试 |
| `scripts/evaluate_moe_tasks.py` | 六个 zero-shot 下游任务 | likelihood 与手工 token score 对齐；未运行正式任务数据集 |

2026-09-09 的 44 项测试和双进程通信检查通过；使用原有入口完成单 GPU BF16、
双进程 CPU 各 8 步训练，真实 W&B SDK 离线记录通过，未测试上传到云端。
历史验证及本次结果见 [moe_salaad_validation.json](moe_salaad_validation.json)。随机 token
只用于软件检查；这些结果不代表 DCLM 预训练质量、论文复现或 H100 吞吐量。
进入完整预训练前，先用默认原生 DP 入口在目标 GPU 上完成 vanilla/SALAAD 配对短跑，
覆盖辅助状态初始化和多次 SVD 更新，检查质量指标、峰值显存及每步耗时。

2026-09-16 重写为 MoE 专用框架后，49 项测试通过，包括公共断点、显式参数、
暂停/恢复、拒绝旧配置和禁止旧 LLM/Vision 模块导入的检查。固定 seed 的 8 步 CPU
SALAAD/vanilla 对比中，重写前后的模型、Adam、辅助状态、数据游标和 RNG 逐项完全一致。
通过新入口完成单 GPU BF16 和双进程 CPU DP 各 8 步短跑，真实 W&B SDK 离线记录正常。
本次验证详情见 [moe_framework_validation.json](moe_framework_validation.json)；
此前的 `moe_salaad_validation.json` 保留为 2026-09-09 的历史验证快照。

## 配置

运行配置全部放在 `configs/` 根目录，支持相对路径 `inherits` 深合并。
`--cfg_version smoke_bf16` 读取 `configs/smoke_bf16.yaml`，依次继承
`smoke.yaml`、`ns97m.yaml`。模型结构直接写在 YAML 中，无需额外的 `_model.json`。

| 配置 | 层数 / hidden / expert FFN | experts / top-k | 全局 batch | 步数 | 参数总数 |
|---|---|---|---|---|---|
| `ns97m.yaml` | 8 / 256 / 176 | 64 / 8 | 1024，DP2 × micro16 × accum32 | 2100 | 97,194,240 |
| `ns480m.yaml` | 12 / 512 / 352 | 64 / 8 | 1024，DP4 × micro8 × accum32 | 3624 | 479,736,320 |
| `smoke.yaml` | 2 / 32 / 24 | 8 / 2 | 4，DP1 × micro2 × accum2 | 8 | 53,920 |

`smoke_dp2.yaml` 是 DP2 的同模型配置，`smoke_bf16.yaml` 是单 GPU BF16 配置。
`ns480m.yaml` 尚无独立的 vanilla 配置文件；该尺度的对照应继承它，并只修改实验名、
设置 `salaad.enabled: false`。

所有层都是 MoE；MHA、RoPE、RMSNorm、SwiGLU，无 bias、无 dropout、embedding/head
不共享。`ns97m_vanilla.yaml` 关闭 SALAAD；`ns97m_independent_ls.yaml` 固定 shared
为零。其他消融可以在继承配置中设定：

```yaml
inherits: ns97m.yaml
experiment: moe_ns97m_down_only
salaad:
  projections: [down]
  # shared_mode: learned | fixed | none
  # low_rank_enabled: true
  # sparse_enabled: true
```

`fixed` 固定初始化均值；`none` 为 independent L+S。只保留 L 时，初始化
`L=X-X_hat,S=0`，随后逐渐收缩；只保留 S 时设 `low_rank_enabled: false`。
不同时关闭两种 residual。支持的首版并行策略为 DP，EP=TP=PP=CP=1。

```bash
myenv/bin/python scripts/train_salad.py --cfg_version ns97m --dry-run
myenv/bin/python scripts/train_salad.py --cfg_version ns480m --dry-run
```

Dry run 不分配 97M/480M 模型，不读取语料。

## 数学与梯度顺序

路由先选 top-k logits，再在被选专家上 softmax。balance loss 使用完整 expert
softmax；计数除以 `tokens * topk`，单层均匀路由的 balance loss 为 1。
balance/z-loss 跨层求和，跨 microbatch 和 DP 取均值。

任务目标是 `CE + 0.1 * balance + 0.001 * z_loss`。NLL/PPL 只计算 CE。

1. 累积 `task_loss / accumulation_steps` 的梯度。
2. 完成一次 DP 平均；Megatron 同时完成 `prepare_grads()` 的 master 梯度准备。
3. 对所有 expert 加 `rho * (X - Q)`，其中 `Q = X_hat + L + S - scaled_dual`。
4. 对包含约束项的完整梯度做 global norm clipping，再执行 AdamW。
5. 成功更新后，在指定边界执行结构 sweep、广播 Q、推进成功步计数与数据游标。

第 3 步不除以 K、元素数、DP 或 accumulation，不乘 routing weight，也不遗漏
未被选中的 expert。native 参数本身是 FP32 masters，autocast 提供 BF16 计算；
Megatron 则直接访问 optimizer 创建的 `main_param`，保持其 BF16 model copy。
两个后端有不同的混合精度实现，跨后端不声称逐位相同。
原生 `selective_attention` 重计算整个 attention 模块（含 QKV/output projection）；
Megatron 对应其原生 selective attention 策略，两者的重计算开销不同。

默认在第 0 步、首次参数更新之前初始化 `X_hat=mean(X), L=0, S=X-X_hat, scaled_dual=0`，
使初始约束残差接近零。97M/480M 每 10 个成功 optimizer steps 按 shared → SVT(L) → soft(S) → scaled dual
更新一次。第 480M 方案结束时补一次 structure flush；`--stop-after` 的中途暂停不
提前 flush，也不缩短 LR 或 target ramp。
本地 smoke 同样在第 0 步初始化，每 2 步更新一次结构，即第 2、4、6、8 步。

每个 expert 独立存储阈值 `tau_l=alpha/rho`、`tau_s=beta/rho`。rho 对所有组固定统一：
97M 为 `1e-5`，480M 为 `3.3333333333333333e-6`。controller 使用**奇异值之和**
覆盖率（gamma=0.999），不是平方和；零 L 的统计为零。秩比和 sparse 非零密度
从 1 逐渐降到 0.15/0.10，decay 开始时冻结 controller 系数，仍继续结构更新。
控制器采用有符号积分更新，`nonnegative_projection: false`，允许 `tau_l/tau_s`
超调到负值。奇异值和逐元素幅度的截零仍保留；负阈值可产生扩张。
checkpoint 恢复接受有限的负阈值，并继续检查状态中的 NaN/Inf。

`group_id % DP` 决定整组 owner。owner 保留 shared/L/S/dual/阈值和 streaming SVD 的基，
所有 rank 缓存 FP32 约束目标 Q。结构更新固定使用 SALAAD++ 的 streaming SVD：
对每个 expert 复用上一轮基，执行一次正交迭代（矩阵乘法 + QR），用投影列范数估计奇异值。
这是近似分解；秩控制器先排序估计值，再按原有的奇异值之和计算覆盖率。
`actual_rank_mean` 记录正的 streaming 分量数，也属于近似秩诊断。
初始化按 SALAAD++ 用一次普通 SVD 准备基，不改变 `L=0, S=X-X_hat`，
直接调用 PyTorch 默认实现。导出直接保存训练得到的 L，不再执行 SVD。
没有 driver 配置、后备算法或训练时的分解路径切换。每块 8 个矩阵（smoke 为 4），
streaming 更新与重构使用 FP32 并禁用 TF32。每个 expert 新增 `k*k` 个 FP32 基元素，
`k=min(out,in)`；基保存在 checkpoint 中，旧版缺少该基的 checkpoint 不能直接续训。
所有 owner 的候选状态通过有限值检查后才提交。
失败会传播到全部 rank 并中止该次运行；从最后完整 checkpoint 恢复，不从失败后的
内存对象继续调用训练。事务性候选状态会产生额外临时内存，纸面静态预算未包含它。

WSD 采用 **1-based optimizer update**：第 1 步 LR 为 `max_lr/warmup_steps`，
第 M 步达到 `min_lr`。这是对前文以 completed steps 记号表达的调度做出的明确
索引约定；两个后端统一使用同一个函数。前 500 步 pilot 保留完整 2100 步曲线的前缀。

## 本地完整 smoke

使用现有 `myenv` 即可。基础路径依赖 PyTorch、NumPy、PyYAML、wandb；文本准备需要
tokenizers/transformers，Parquet 需要 pyarrow，DCLM 的 Zstandard 压缩切片需要
zstandard，下游评估需要 `lm-eval==0.4.9.1`。
本次测试环境为 PyTorch 2.7.1+cu126、NumPy 1.26.4、PyYAML 6.0.2。

本地单 GPU BF16 测试配置为 `configs/smoke_bf16.yaml`，使用你原来的入口：

```bash
myenv/bin/torchrun --standalone --nproc-per-node=1 scripts/train_salad.py \
  --cfg_version smoke_bf16
```

该配置运行 2 层、hidden 32、8 experts / top-2 的模型，共 8 个 optimizer steps。
`data.synthetic_smoke: true` 会自动准备小型随机 token 语料。
默认输出在 `data/salaad_moe/smoke_bf16/<timestamp>/`，语料在该次输出的 `data/` 下。
`--folder` 可修改输出分组，`--output` 可直接指定运行目录。
正式配置没有自动生成随机语料的标记，必须提供真实的 `--data-manifest`。

也可显式准备语料、指定输出，再导出和评估。下面的目录须尚未存在；输出目录防覆盖。

```bash
myenv/bin/python scripts/prepare_moe_data.py \
  --config configs/smoke.yaml --synthetic-smoke \
  --output /tmp/moe_demo_data

myenv/bin/python scripts/train_moe.py \
  --config configs/smoke.yaml \
  --data-manifest /tmp/moe_demo_data/manifest.json \
  --output /tmp/moe_demo_run --device cpu --allow-synthetic

myenv/bin/python scripts/export_moe.py \
  --checkpoint /tmp/moe_demo_run/checkpoints/step_00000008 \
  --output /tmp/moe_demo_export.pt

myenv/bin/python scripts/evaluate_moe.py \
  --model /tmp/moe_demo_export.pt --mode exported \
  --data-manifest /tmp/moe_demo_data/manifest.json --device cpu
```

双进程 CPU 使用 `smoke_dp2.yaml`，显式指定 CPU，避免单卡机器将 rank 1 指向不存在
的 GPU。单 GPU BF16 使用 `smoke_bf16.yaml`。

```bash
myenv/bin/torchrun --standalone --nproc-per-node=2 scripts/train_salad.py \
  --cfg_version smoke_dp2 --device cpu

myenv/bin/torchrun --standalone --nproc-per-node=1 scripts/train_salad.py \
  --cfg_version smoke_bf16 --device cuda
```

两个入口均支持 `--cfg_version` 和显式的 `--config`，二者择一。
MoE 的 `--num_total_iters N` 是 `--stop-after N` 的兼容别名：在第 N 个成功更新后暂停，
保留完整的 LR/结构调度。恢复时若省略 `--output`，会使用 checkpoint 所在的运行目录，
并从 `run_metadata.json` 找回语料 manifest。

## W&B 日志

所有 MoE 配置继承 `configs/ns97m.yaml` 中的设置，vanilla 和 smoke 同样启用：

```yaml
is_wandb: true
wandb_project: SALAAD-MoE
wandb_entity: hao-ma-eth-z-rich
```

原生 DP 训练器只由 rank 0 创建 run。记录完整的合并配置、训练 loss/NLL、学习率、
梯度范数、raw/reconstructed 验证指标、router 统计和 SALAAD 分解诊断，横轴为
`optimizer_step`。逐 expert 的原始路由计数保存在本地 `metrics.jsonl` 中。
W&B 调用前后保存并恢复训练 RNG，开启日志不会改变训练随机序列。

默认在线记录，复用现有 W&B 登录或 `WANDB_API_KEY`；密钥不写入配置。
需要离线调试时可在命令前加 `WANDB_MODE=offline`，或在配置中写
`wandb_mode: offline`。基础配置仍默认在线，当前 CLI 默认的 `smoke_dclm_bf16`
则显式选择离线记录。

`wandb_run.json` 保存 run ID 和最后记录步数。在原输出目录从最新 checkpoint 恢复时
复用同一 run；从较早 checkpoint 回退或改用新输出目录时创建新 run，避免步数冲突。
初始化、日志写入错误会传播到全部 DP ranks。

## DCLM 与 tokenizer

固定 dataset revision `a3b142c183aebe5af344955ae20836eb34dcf69b`，Pythia tokenizer
revision `bb1e3e710cdf6b524461d543cfb5ba773f0a81b6`。只下载 tokenizer/config，
不下载 Pythia 模型权重。`shards.json` 记录明确选定的 shard 及 SHA256；不自动下载
整个 DCLM。`dclm_shard_names.json` 应为预先选定的仓库相对文件名 JSON 数组。

```bash
myenv/bin/python scripts/fetch_moe_sources.py \
  --config configs/ns97m.yaml --output /scratch/moe_sources \
  --shard-list /scratch/dclm_shard_names.json

myenv/bin/python scripts/prepare_moe_data.py \
  --config configs/ns97m.yaml \
  --shard-manifest /scratch/moe_sources/shards.json \
  --tokenizer-json /scratch/moe_sources/tokenizer/tokenizer.json \
  --output /scratch/moe_tokens
```

已有本地 JSONL、JSONL.GZ、JSONL.ZST/JSONL.ZSTD 或 Parquet 可以用 `--input shard1 shard2 ...` 替代
`--shard-manifest`。UTF-8 原文 SHA256 决定去重与划分：余数 0–99 validation，
100–199 test，200–9999 train。SQLite 提供磁盘上的全局去重。划分不依赖文件名
或随机划分进程。98/1/1 是本方案的选择，不声称来自 MirrorMoE 的未披露设置。

每篇追加 EOD=0，不加 BOS。2049-token 样本产生 2048 个预测，stride=2048，
跨文档 attention/position 不 reset，EOD 计入 loss。验证/测试最多保存各 8192 个
完整序列；至少需要每个 split 有一个完整序列，否则不发布 manifest。训练入口
还会检查 held-out 规模是否达到配置要求，不足时应补充输入 shards。

准备较小子集时可加 `--max-train-tokens N`，其中 N 是预测 token 数且须为
`seq_length` 的整数倍。训练文件最多写入 N+1 个 token，额外的一个用于标签位移；
达到训练上限后仍会继续填充验证/测试集合，三者均满后停止处理。
这是上限，若输入不足，须根据生成的 manifest 检查实际 token 数。
当前本地真实文本 smoke 和云端下载位置见 [DCLM 数据记录](moe_dclm_data.md)。

原生 reader 使用 epoch 内的仿射全排列，保存全局 sequence cursor，无需把全部
样本索引存入内存。Megatron 使用其 GPTDataset 索引/打包，dataset seed 固定为
`corpus_order_seed`；两个后端的训练样本顺序不保证相同。配对比较须使用同一后端、
同一语料 manifest、相同初始化 seed、相同 batch 配置。

## 默认原生 DP：97M / 480M

先在同样的语料和硬件上分别短跑 vanilla 与 SALAAD。以下命令停在第 200 步，
保持配置中完整的 2100 步 LR/结构调度，覆盖训练前初始化和后续多次结构更新：

```bash
myenv/bin/torchrun --standalone --nproc-per-node=2 scripts/train_moe.py \
  --config configs/ns97m_vanilla.yaml --device cuda \
  --data-manifest /scratch/moe_tokens/manifest.json \
  --output /scratch/moe_ns97m_vanilla --stop-after 200

myenv/bin/torchrun --standalone --nproc-per-node=2 scripts/train_moe.py \
  --config configs/ns97m.yaml --device cuda \
  --data-manifest /scratch/moe_tokens/manifest.json \
  --output /scratch/moe_ns97m --stop-after 200
```

检查短跑结果后，使用对应配置和 checkpoint 恢复到完整预算；SALAAD 组例如：

```bash
myenv/bin/torchrun --standalone --nproc-per-node=2 scripts/train_moe.py \
  --config configs/ns97m.yaml --device cuda \
  --data-manifest /scratch/moe_tokens/manifest.json \
  --output /scratch/moe_ns97m \
  --resume /scratch/moe_ns97m/checkpoints/step_00000200
```

480M 改成 `ns480m.yaml` 和 `--nproc-per-node=4`，并补充匹配的 vanilla 继承配置。
原生顺序 expert 实现是当前默认训练路径，尚无 97M/480M 吞吐量测量。
LR sweep 修改配置中的 LR，并为每次
trial 新建输出；选定后从同一 seed 重新完整训练。通过继承配置切换 vanilla 或
independent-LS，其余 batch、数据、LR、tokens 保持一致。

默认从第 0 步启用 SALAAD。若另行选择复用 vanilla 前缀的实验，需要在目标继承配置中
显式设定 `salaad.state_initialization_step: 100`。此时先用 vanilla 配置运行
`--stop-after 100`，随后用 SALAAD 配置和新输出目录，传入
`--branch-from /scratch/vanilla/checkpoints/step_00000100`。这会继承完整模型、Adam、
数据游标和 RNG，并从当前 masters 新建辅助状态。入口要求分支点恰好是目标配置的
初始化步，且 model/data/training/parallel/seed 完全相同；不能把更换 LR 或 batch
误当作同一前缀比较。

## 可选 Megatron 适配入口

本节保留早期参考环境的接入方法。仅在明确选择此后端时执行；默认的原生 DP
vanilla/SALAAD 实验不依赖本节的环境安装、模型转换或验证。此入口同样只启用 DP。

采用 [FLAME-MoE 固定快照](https://github.com/cmu-flame/FLAME-MoE/tree/e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78)
中的 Megatron 子模块 `cbaf684c5d03997e0fdd5347c5e2d371c381a3d8`。
入口检查 HEAD 和 tracked 文件是否干净，并检查 CUDA/TE/Apex 等基础依赖。
参考该快照的安装脚本配置独立环境（Python 3.10、Torch 2.6.0+cu124、TE 1.11
对应子模块和 Apex），不要覆盖现有 vision 环境。环境预检不等于 ABI/实际 kernel
通过验证，必须随后运行短跑。

```bash
git clone https://github.com/cmu-flame/FLAME-MoE.git /scratch/FLAME-MoE
git -C /scratch/FLAME-MoE checkout e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78
git -C /scratch/FLAME-MoE submodule update --init --recursive Megatron-LM apex TransformerEngine
```

完成该环境安装后，让 pinned Megatron 位于 PYTHONPATH，重新准备一个新目录，
加 `--write-megatron-indexed`，会同时生成 split 的 `.bin/.idx` 和统一 token manifest。
三个 split 分别物化，不能再对 train 做另一次百分比划分。indexed 文件也记录 SHA256。

```bash
export PYTHONPATH=/scratch/FLAME-MoE/Megatron-LM
python scripts/prepare_moe_data.py \
  --config configs/ns97m.yaml \
  --shard-manifest /scratch/moe_sources/shards.json \
  --tokenizer-json /scratch/moe_sources/tokenizer/tokenizer.json \
  --write-megatron-indexed --output /scratch/moe_indexed

python scripts/train_moe_megatron.py \
  --config configs/ns97m.yaml \
  --megatron-path /scratch/FLAME-MoE/Megatron-LM --check-environment

torchrun --standalone --nproc-per-node=2 scripts/train_moe_megatron.py \
  --config configs/ns97m.yaml \
  --megatron-path /scratch/FLAME-MoE/Megatron-LM \
  --data-directory /scratch/moe_indexed \
  --tokenizer-directory /scratch/moe_sources/tokenizer \
  --output /scratch/moe_megatron_ns97m --stop-after 200
```

本后端的完整运行去掉 `--stop-after`。恢复时增加 `--resume /scratch/moe_megatron_ns97m`。
它也支持 `--branch-from`，参数是 vanilla save root。
`--dry-run` 可打印全部原生 flags，无需安装 Megatron。主入口选择 TE attention +
SequentialMLP；GroupedMLP 的矩阵映射有单独测试，TEGroupedMLP、EP/TP sharding
和 distributed optimizer 会被明确拒绝。

hook 保留原生 optimizer 的 global clipping/step，绝不再次向最终 scalar loss
加入 Megatron 已自动附加的 balance/z-loss。SALAAD owner 更新在 optimizer 成功后
执行。原生 training、master weights、Adam、scheduler、RNG、consumed samples 保存
在 Megatron checkpoint；`salaad/iter_XXXXXXX` 同步保存完整 auxiliary shards 和
供统一 eval/export 使用的 native weight conversion。只有 complete marker 发布后
才可恢复。转换副本不是原生 Trainer 的恢复 checkpoint。

周期验证使用独立模型副本和固定 monitor token 集，输出 raw/reconstructed NLL。
末尾 validation/test 使用各自完整配置规模。保留最近 2 个以及 monitor raw-NLL
最优的最多 3 个 checkpoint；这里实现的是 raw-NLL 排序，没有假装实现多目标
Pareto 选择。保存期间会额外写一份转换后的 dense 权重，应预留相应磁盘空间。
当前转换已测试 full-MHA QKV 重排、fused SwiGLU 拆分和 TE norm 参数映射；尚需
在实际 Megatron GPU scorer 上比较 logits/NLL，不能把 CPU 协议测试作为该验证。

## 导出与质量检查

```bash
myenv/bin/python scripts/export_moe.py \
  --checkpoint /scratch/moe_ns97m/checkpoints/step_00002100 \
  --output /scratch/moe_ns97m_export.pt --device cuda

myenv/bin/python scripts/evaluate_moe.py \
  --model /scratch/moe_ns97m_export.pt --mode exported \
  --data-manifest /scratch/moe_tokens/manifest.json \
  --split test --device cuda --output /scratch/moe_export_test.json

myenv/bin/python scripts/evaluate_moe_tasks.py \
  --model /scratch/moe_ns97m_export.pt --mode exported --device cuda \
  --tokenizer-directory /scratch/moe_sources/tokenizer \
  --output /scratch/moe_export_tasks.json
```

将 model 改为 checkpoint 目录，mode 改为 `raw`/`reconstructed` 即可评估另外两种
权重。以上使用完整 97M 原生运行的 checkpoint；短跑检查可替换成已存在的第 200 步目录。
导出遵循“训练得到什么就导出什么”：shared 和 L 按 checkpoint 原值、原精度直接保存，
S 的全部非零值无损转换为 CSR，column/crow 索引为 int32，未分解权重也保留原值和精度。
没有 rank/density 上限、数值秩截断、top-k 筛选、额外 SVD 或 BF16 转换。
训练中的 rank/density 控制器目标仍由 `salaad.controller` 管理。

`exported` 路径从文件读取 shared/L/S，按训练中相同的顺序计算 `shared + L + S`，
原生模型与 `reconstructed` 权重逐位一致。新文件格式为 `salaad_moe.export.v2`。
JSON 报告使用原模型实际 dtype 计算体积，分别列出 tensor bytes 和真实文件 bytes；
导出文件不保证比 dense 模型更小。评估使用物化后的 dense 权重。

六个任务固定为 `arc_easy, openbookqa, hellaswag, piqa, boolq, social_iqa`，
zero-shot，seed 42；报告 acc、可用时的 acc_norm，以及六任务不加权 mean acc。
使用独立 likelihood adapter，不将模型冒充为 stock Llama/Mixtral。
`--limit` 仅用于调试，会写入结果。正式质量门槛仍是待验证目标：raw 相对 vanilla
≤0.02 nats/token，exported 相对 raw ≤0.03，总计 ≤0.05。

## 测试

```bash
myenv/bin/python -m unittest discover -s tests/moe -v
myenv/bin/torchrun --standalone --nproc-per-node=2 tests/moe/distributed_worker.py
```

测试涵盖数学更新、selected/full softmax 区分、未访问 expert 梯度、causal attention、
label shift、rank 统计、controller freeze、owner 全局失败传播、空 owner rank、
完整恢复、实际 CSR 读取、BF16 masters、重计算及 Megatron master-gradient 顺序。
还覆盖 `--cfg_version` 根目录配置、W&B rank 0 记录、日志开关下训练状态完全一致、
恢复 run ID 和日志错误传播。
