# DCLM 下载与真实文本 smoke

日期：2026-09-16。数据固定为 `mlfoundations/dclm-baseline-1.0`，revision
`a3b142c183aebe5af344955ae20836eb34dcf69b`；Pythia tokenizer 固定为
`bb1e3e710cdf6b524461d543cfb5ba773f0a81b6`。只获取 tokenizer 配置与词表文件。

## 云端语料

主机：`hma2@login.cluster.is.localnet`。

```text
/lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916/
  download_plan.json
  sources/
    tokenizer/
    shards/
    download_progress.jsonl
    shards.json
```

下载前检查共享文件系统约有 19 TiB 可用。选择
`global-shard_01_of_10/local-shard_0_of_10/` 按路径排序的前 48 个完整切片，
共 10,502,832,978 字节（10.50 GB）。下载器逐个核对远端仓库记录的文件大小和
SHA-256，全部通过后才发布 `sources/shards.json`。
本次 48 个切片已全部下载并通过校验，最终 manifest 已发布；tokenizer 已同步并核对哈希。

这些文件是原始压缩 JSONL。97M 配置的目标为 **4,404,019,200 个训练预测 tokens**；
是否足够、最终使用哪些文档，都需要经过实际分词确认。当前尚未生成完整的云端
4.4B-token 训练文件，也未启动正式模型训练。CPU 预处理作业等待用户提供 HTCondor bid。

云端仓库仍在 `feature/salaad-vision`，本地为 `feature/moe-salaad`；本次只传输数据与
分词器缓存，没有同步代码或切换云端分支。后续云端预处理还需先安排匹配的 MoE 代码版本。
当前状态与校验摘要保存在 [验证记录](moe_dclm_data_validation.json)。

## 本地小切片

下载前本地约有 54 GiB 可用。下载首个完整原始切片，共 141,306,362 字节。
`data/moe_corpora/dclm_20260916/sources_local/shards.json` 记录其 HF 缓存路径与校验和；
tokenizer 文件保存在同级 `tokenizer/`。

准备命令如下。输出目录必须尚不存在，重新准备时换一个目录。

```bash
myenv/bin/python scripts/prepare_moe_data.py \
  --config configs/smoke_dclm_bf16.yaml \
  --shard-manifest data/moe_corpora/dclm_20260916/sources_local/shards.json \
  --tokenizer-json data/moe_corpora/dclm_20260916/sources_local/tokenizer/tokenizer.json \
  --output data/moe_corpora/dclm_20260916/smoke_tokens \
  --max-train-tokens 1048576
```

`--max-train-tokens` 限定训练预测 token 数，训练文件额外保留一个 token 生成最终标签。
达到训练上限后仍继续填充验证与测试集合；数据不足时 manifest 会显示实际数量。
读取 `.zst`/`.zstd` 采用流式解压，无需将整份文本展开到磁盘。

| 划分 | 完整序列数 | 文件中的 token 数 | 文件大小（字节） |
|---|---:|---:|---:|
| train | 32768 | 1048577 | 4194308 |
| validation | 16 | 513 | 2052 |
| test | 16 | 513 | 2052 |

每条序列长 32，训练预测 token 数为 1,048,576。词表使用真实 Pythia token IDs，
包含 added tokens 后为 50,277，模型词表 padding 到 50,304。此配置明确关闭随机语料生成。

## 已完成的 smoke test

2026-09-17 起，CLI 的 `parse_args()` 默认选择 `smoke_dclm_bf16` 和上述本地
`smoke_tokens/manifest.json`，直接运行 `myenv/bin/python scripts/train_salad.py` 即可。
显式指定其他配置或恢复 checkpoint 时保留其原有数据选择，`--data-manifest` 可覆盖默认路径。

所有训练仍从同一个 `main()` 进入：

```bash
myenv/bin/torchrun --standalone --nproc-per-node=1 scripts/train_salad.py \
  --cfg_version smoke_dclm_bf16 --device cuda \
  --data-manifest data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json \
  --output data/salaad_moe/smoke_dclm_bf16/20260916_real_text
```

RTX 3050 Ti Laptop 上完成 8 个 BF16 optimizer steps、4 次 SALAAD 结构 sweep，
训练实际消费 1024 个预测 tokens。raw 验证 NLL 为 10.8092079163，重构验证 NLL 为
10.8092093468；第 8 步完整 checkpoint 已生成。真实 W&B SDK 的离线 run 为
`570b3895`，项目 `SALAAD-MoE`，已记录到第 8 步。

`tests/moe/test_workflow.py` 的 15 项检查通过，包括压缩文件的多帧读取，以及训练 token
达到上限后仍填充 held-out 数据的行为。本次 smoke 验证数据读取和训练流程，不用于判断
模型预训练质量。
