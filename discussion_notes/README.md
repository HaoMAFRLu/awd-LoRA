# 讨论记录索引

更新时间：2026-10-13（Asia/Shanghai）。

先读 [MoE-SALAAD 最新讨论与实验记录](MOE_SALAAD_DISCUSSION_LOG.md#current-state)。
本次归档实验回顾、自由残差抵消共享约束的问题、三组 L2 penalty 的完成结果，
以及用户纠正 P 更新意图后的三个闭式重跑任务。旧实验和决定按日期保留。
**仅在用户要求时更新讨论，不随每次代码修改追加。**

## 当前方案：dense X_e + L2，闭式 P 后接软 Sinkhorn

不分解为 L+S。初始化 X 为专家均值、所有 P_e=I、X_e=W_e-X、U=0；
每 10 步按 W→P→X→X_e→U 更新，残差用 rho/(rho+λ) 的 L2 近端收缩。
rho=1e-5，P 使用闭式最小二乘、正截断和软 Sinkhorn；
clip_min=1e-5、最多 10000 轮、容差 1e-5、允许提前停止，不硬化。

用户澄清本意是闭式 P 后，已按“其余参数不变”重新提交 λ=1e-3、1e-4、1e-5。
代码提交 `363d5ef` 已同步；三组仍为 2100 步、seed 42、weight decay=0、每组 4 H100、bid 100。

| λ | 配置 | Condor job | W&B | 最近核验的启动状态 |
|---|---|---|---|---|
| 1e-3 | [配置](../configs/ns97m_sinkhorn_closed_form_l2_lambda1e-3.yaml) | **17696741.0** | [310b9896](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/310b9896) | Running / step 10 |
| 1e-4 | [配置](../configs/ns97m_sinkhorn_closed_form_l2_lambda1e-4.yaml) | **17696742.0** | [c6289d83](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/c6289d83) | Running / step 11 |
| 1e-5 | [配置](../configs/ns97m_sinkhorn_closed_form_l2_lambda1e-5.yaml) | **17696743.0** | [a22c9035](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/a22c9035) | Running / step 2 |

状态来自 **2026-10-13 09:11:57（Asia/Shanghai）的启动核验**，不是最终训练结果，
本次文档更新未重新查询在线状态。实际配置均与提交计划一致。
见 [验证摘要](../data/moe_analysis/closed_form_l2_lambda_sweep_20261013/validation.json)、
[提交记录](../data/moe_analysis/closed_form_l2_lambda_sweep_20261013/submissions.json) 和
[启动快照](../data/moe_analysis/closed_form_l2_lambda_sweep_20261013/startup_verification.json)。

## 已完成的 gradient + L2 对照及 λ=0

三组 gradient 任务 17695197.0 / 17695198.0 / 17695199.0 均完成 2100 步，
对应 λ=1e-3 / 1e-4 / 1e-5；末 100 步 NLL 分别为 **4.32713 / 4.30466 / 4.20795**。
penalty 压低了 X_e，但相对重构误差分别约 **209.3% / 125.7% / 22.8%**。
残差变小没有自动转化为更好的训练 NLL，X 或 XP 变大也不代表有效共享。

旧 λ=0 对照为 **17684559.0，W&B [0221dbae](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/0221dbae)**，
末 100 步 NLL 为 **4.17559**。旧对照使用闭式 P、clip_min=1e-8，
与上述梯度三组不是严格的 λ-only 消融。没有另行提交新的 λ=0 对照。

- [主记录：公式、X 与 XP 分列范数表、结果解释](MOE_SALAAD_DISCUSSION_LOG.md#current-state)
- [范数对比 CSV](../data/moe_analysis/gradient_l2_norms_20261013/comparison_with_unregularized_x_xp.csv)
  / [完整统计和配置差异](../data/moe_analysis/gradient_l2_norms_20261013/comparison_with_unregularized.json)
  / [曲线](../data/moe_analysis/gradient_l2_norms_20261013/penalty_norm_curves.png)
- [10 月 12 日实验清单](../data/moe_analysis/experiment_inventory_20261012/experiments.csv)
  / [完整配置和结果快照](../data/moe_analysis/experiment_inventory_20261012/experiment_summary.json)：
  当时 18 个正式 run，不包括后来的六个 L2 任务；排除 L+S 后的九项结果见主记录。

## 已完成的 shared-expert baseline：待统一 evaluation

每层 **1 个 shared + 63 个 routed experts**，宽度均为 176；
**1+7 为主要 baseline，1+8 为增加专家计算量的补充对照**。
两组 shared 模型参数量相同，为 97,192,192；与原 64-routed 模型仅相差 2048 个 router 参数。
1+7 与原 top-8 的专家 MLP 理论计算量一致，1+8 为其 1.125 倍；这不是整模型运行时间比例。

| 配置 | Job | W&B |
|---|---|---|
| [1+7](../configs/ns97m_shared_1plus7.yaml) | **17690460.0** | [807d6bb1](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/807d6bb1) |
| [1+8](../configs/ns97m_shared_1plus8.yaml) | **17690461.0** | [80378abb](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/80378abb) |

**2026-10-10 08:20–08:23（Asia/Shanghai）** 核验的 W&B 状态均为 `finished`，
两组都已完成 2100 步、4,404,019,200 个预测 token。本次归档采用这份已核验数据，未重新查询在线状态。

| 训练指标 | 1+7 | 1+8 |
|---|---:|---:|
| 最后一步 NLL | **4.16865** | 4.17549 |
| 末 100 步平均 NLL | **4.18209** | 4.18875 |
| 末 500 步平均 NLL | **4.20522** | 4.21257 |
| W&B 记录运行时间 | 4.82 小时 | 4.99 小时 |

末 100 步逐步对齐后，100 / 100 步都是 1+7 更低，平均差 **0.00666233**。
这次单 seed 实验中，1+8 增加激活专家数未带来训练 NLL 收益；保持 1+7 为主 baseline。
两组路由负载分配较均衡，记录指标无 NaN/Inf。以上 NLL 不含路由辅助损失，
也不是验证集结果；两个 run 尚无验证集/下游评测，不能据此确定泛化或 consensus 的收益。

两组保持相同初始权重和数据顺序；2100 步、seed 42、LR 3e-4、**weight decay=0**，
每组 4 张 H100、bid 100，关闭 SALAAD/consensus/Sinkhorn。
shared 分支系数为 1，routed 权重和为 1；路由和辅助损失仅统计 routed experts。
软件提交 **2433bdf** 已同步本地/origin/集群；155 项测试及双进程 BF16 训练、逐位恢复通过。

- [shared baseline 训练结果与结论](MOE_SALAAD_DISCUSSION_LOG.md#history-through-20261012)
- [完整统计](../data/moe_analysis/shared_wandb_20261010/comparison.json) /
  [收敛曲线](../data/moe_analysis/shared_wandb_20261010/training_comparison.png) /
  [逐步对齐比较](../data/moe_analysis/shared_wandb_20261010/comparison_by_step.csv)
- [10 月 9 日设计、实现与启动快照](MOE_SALAAD_DISCUSSION_LOG.md#history-through-20261009)
- [1+7 原提交记录](../data/moe_analysis/shared_expert_baselines_20261009/ns97m_shared_1plus7/submission.json) /
  [1+8 原提交记录](../data/moe_analysis/shared_expert_baselines_20261009/ns97m_shared_1plus8/submission.json)：保留当时的 Running 快照。

## 较早完成的 consensus：cosine-Hungarian 初始化，随后固定 150 轮软 Sinkhorn

训练 **17687476.0** 已完成 2100 步，使用
[ns97m_sinkhorn_cosine_init](../configs/ns97m_sinkhorn_cosine_init.yaml)，
[W&B 81e36dbd](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/81e36dbd)。
P_e 在训练前按原始均值模板的通道 cosine 相似度做一次匈牙利初始化；
随后每 10 步更新一次，每次固定 150 轮 Sinkhorn，容差 1e-5，不在末尾硬化。

分析 **17690447.0** 已完成：最终 512 个 P_e 匈牙利投影后 **0 个单位矩阵、512 个非单位矩阵**，
平均 98.7/176 个通道偏离原位置。软 P_e Frobenius 均值 1.686；
相对相同种子和规则的独立初始排列，30.46% 的通道对应发生变化。
这些结构结果不等于任务性能提升或知识集中在 X，新的硬化模型尚未进行 evaluation。

- [最终 P_e 统计](../data/moe_analysis/cosine_init_final_pe_20261009/17690447_0/pe_row_maxima.json)
- [与初始化的比较](../data/moe_analysis/cosine_init_final_pe_20261009/17690447_0/initial_comparison.json)
- [10 月 5 日讨论、10 月 6 日归档快照](MOE_SALAAD_DISCUSSION_LOG.md#history-through-20261006)
- [早期英文 ADMM PDF](moe_salaad_admm_20261005/moe_salaad_admm.pdf) /
  [问答 PDF](moe_salaad_qa_20261004/moe_salaad_qa.pdf)：保留各自历史版本，不用旧初始化描述覆盖最新配置。

集群经 `local_scripts/mpi_vpn_socks.sh` 的 SOCKS 11080 连接，保留 ETHZ/Cisco。
10 月 9 日 shared 双任务的代码同步和提交已成功，后文的 SSH 阻塞与“尚未提交”均为历史状态。

## 2026-09-23 归档的决定与训练（历史梯度版本）

当前采用 `E_i ≈ X P_i + L_i + S_i`：E 是完整专家权重，X 是 shared，
L/S 各专家独立；P 放宽为双随机矩阵，通过确定性的 log-Sinkhorn 和梯度更新。
这里的新 shared X 对应旧讨论的 H。

**训练开始前初始化分解和 P，不保留 vanilla 前缀。**
从第 10 步起，每 10 个 optimizer steps 同步更新 P、shared、L、S、U，
每轮 P 内部做 **K_P=128** 步 SGD。总计 2100 步；原学习率的 21 步 warmup 保留。
旧版 Sinkhorn 文档中的“前 100 步 vanilla”已被最新决定替代。

原 8 步作业 **17594019.0** 已正常完成 2100 步。
按用户“其余参数不变，只将 8 步增加到 128 步”的要求，已从头提交 **17604698.0**。
该作业于 **9 月 23 日 14:22:14 Europe/Zurich** 在第 **690** 步结构更新时报错退出，
最后完成 689 步；原因是一次 Sinkhorn 在最多 100 轮归一化内未满足 1e-5 行列和容差。
**训练尚未恢复；集群保留第 600 步完整 checkpoint。**

- [128 步 W&B：8e289238](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/8e289238) /
  [8 步 W&B：861f0ac3](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/861f0ac3)
- [128 步提交与配置核对](../data/moe_analysis/sinkhorn_k128_launch_20260923/README.md) /
  [停止原因与 checkpoint 数值检查](../data/moe_analysis/sinkhorn_failure_20260923/README.md)
- [正式配置](../configs/ns97m_sinkhorn.yaml) / [Condor 文件](../sub/moe_ns97m_sinkhorn.sub)
- [通俗算法 PDF](moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.pdf) /
  [LaTeX](moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.tex) /
  [文档说明与编译方法](moe_sinkhorn_alignment_20260922/README.md)
- [实现说明](../docs/moe_sinkhorn_alignment.md) / [软件验证](../docs/moe_sinkhorn_validation.json)

128 步运行使用提交 `4a244be`，提交时本地、origin 和集群已同步，分支 `feature/moe-salaad`。
与原 8 步运行相比，唯一训练参数差异为 `sinkhorn.inner_steps: 8 → 128`。
资源仍为 **4 H100 / DP4 / bid 100**。模型、数据、seed、AdamW/LR、batch、
L/S 控制器、2100 步预算、P 步长及移动惩罚均不变。
rho=1e-5，alpha/beta 控制器增益 0.02/0.002，结构间隔 10 步。
正式数据仍使用 `data/moe_corpora/dclm_20260916/tokens_ns97m/manifest.json`。

任务前向始终使用完整 E；soft P 不具有硬置换的 FFN 函数不变性。
软 P 对应的 shared 更新需要解最小二乘方程，不能继续直接求平均。
P 的 SGD 步长 1.0 作用于 F/rho，与模型 AdamW 学习率 3e-4 分开。
本轮研究重点是通道对应与共享的作用；不以范数变大证明学到了有用知识。

## 2026-09-23 分析结果（历史梯度版本）

- **看实际 XP 的方向、幅度和对应。** 固定同一个最终 X 比较初始 P 与最终 P：
  97.87% 的通道夹角不超过 1°，三个投影及联合向量的最近邻对应均未改变；
  少数通道幅度下降约 20%–24%。P 确实更新，但尚无明确对齐或任务收益。
- **8 步偏保守的证据来自冻结子问题。** 8 步只完成 256 步已观察下降量的 4.79%–12.93%，
  128 步完成 60.97%–81.60%；256 步也不是已知最优解，不能据此保证 NLL 改善。
- **128 步的整轮开销估计为约 3.5%–4.0%。** 这是旧状态函数实测后的外推，
  不是新运行的完整耗时；以后若归一化更慢或上限提高，需要重新估计。
- **128 和 100 属于两层循环。** 128 是更新 A 的 SGD 次数；100 是每次从 A 生成 P
  所允许的行/列归一化轮数。每轮结构更新、每层共调用 Sinkhorn 129 次。
- **尚未验证修复。** 第 600 步的层 2、6 冻结检查分别最多用 85、90 轮，原上限内通过；
  上限 1000 的备用检查没有执行。没有重现第 690 步失败状态，也没有修改参数或恢复训练。

详细证据与图表：

- [8 步训练结果](../data/moe_analysis/sinkhorn_inspection_20260922/README.md)
- [实际 XP 通道比较](../data/moe_analysis/xp_mapping_comparison_20260923/README.md) /
  [端点分解分析](../data/moe_analysis/sinkhorn_dynamics_20260923/README.md)
- [内部 SGD 步数诊断](../data/moe_analysis/sinkhorn_inner_steps_20260923/README.md)
- [计算开销测量与外推](../data/moe_analysis/sinkhorn_compute_overhead_20260923/README.md)

## 9 月归档的已完成分析与已提交对照

首次动态硬置换 `17590593.0`（[89b39a7d](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/89b39a7d)）
已正常完成 2100 步。末 100 步训练 NLL 为 **4.243286**，
原未对齐 consensus 为 **4.234395**；当前动态硬置换没有改善这项任务指标。
每 100 步重匹配处出现分解误差尖峰，最终 checkpoint 正好在第 2100 步匹配之后。

两次最终 checkpoint 对比：24 个 H 范数全部变大，1536 个 S 中 1520 个变大，
1536 个 L 中 1449 个变小。保留逐专家数值，不对专家取均值；
这些变化不等于各部分功能贡献的变化。

- [完整训练结果与图表](../data/moe_analysis/aligned_20260922/README.md)
- [H/S/L 的全部 24 组范数表](../data/moe_analysis/aligned_20260922/norm_comparison.md)
- [逐专家范数 CSV](../data/moe_analysis/aligned_20260922/norm_expert_comparison.csv)

三组硬置换时机对照已在 9 月 22 日提交，仍使用 H+L+S：

| P 的时机 | Job | W&B |
|---|---|---|
| 第 0 步生成，之后固定 | `17593583.0` | [d8c91e59](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/d8c91e59) |
| vanilla 100 步后生成，之后固定 | `17593584.0` | [85cbab9c](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/85cbab9c) |
| 第 100 步生成，第 300、500、…步重新匹配 | `17593586.0` | [7c4edf5d](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/7c4edf5d) |

三组各自提交后曾核验为 Running，最终结果尚未归档；
见 [提交记录](../data/moe_analysis/alignment_schedule_20260922/submissions.json)。
取消 vanilla 前缀只适用于最新 Sinkhorn 实验；上述既有作业保留原配置。
曾讨论 dense H_i 的 `X_i=X_hat+H_i` 方案，但没有采用或提交。

## 保留的工作约定与历史资料

统一使用独立 MoE 训练入口 `scripts/train_salad.py:main()`，代码注释为英文。
正式训练不做在线 validation；W&B 保留逐专家指标，不增加专家均值或额外指标。
每 100 步保存、仅保留最新 checkpoint；导出保留训练得到的完整分解，不追加截断或剪枝。

- [9 月 22 日提交与启动讨论快照](MOE_SALAAD_DISCUSSION_LOG.md#history-through-20260922)
- [截至 9 月 21 日提交的讨论快照](MOE_SALAAD_DISCUSSION_LOG.md#history-through-20260921)
- [9 月 17 日及以前历史讨论](MOE_SALAAD_DISCUSSION_LOG.md#history-through-20260917)
- [旧硬置换完整训练算法 PDF](moe_channel_alignment_20260921/moe_channel_alignment.pdf)
- [原 consensus 参数与存储分析](../data/moe_analysis/final_parameters_20260919/README.md)
- [原 consensus 全部 projection 范数](../data/moe_analysis/final_norms_20260919/README.md)
- [独立 MoE 代码指南](../docs/moe_training_walkthrough.md)
- [基础公式与 PDF](moe_salaad_formulation_20260908/README.md)
- [早期训练方案与 PDF](moe_salaad_training_plan_20260908/README.md)

按 2026-10-13 用户“保存git状态并推送”的要求，本索引和 MoE 主讨论记录纳入 Git，
随本次提交推送到 origin。其余讨论附件、分析数据和训练产物继续由 gitignore 忽略，
相关本地链接需配合原工作区使用；现有 PDF 未改写。历史快照的旧状态不覆盖最新决定。

## 既有 SALAAD-Vision 记录

- [SALAAD-Vision 项目交接](PROJECT_HANDOFF_SALAAD_VISION.md)
- [SALAAD-Vision 实验日志](SALAAD_VISION_EXPERIMENT_LOG.md)

Vision 记录中的实验结论属于对应视觉实验，不能作为 MoE-SALAAD 的训练结果。
