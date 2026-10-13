# MoE-SALAAD 讨论记录

更新时间：2026-10-13（Asia/Shanghai）。
工作目录：`/home/hao/Desktop/MPI/awd-LoRA`。  
当前分支：`feature/moe-salaad`，跟踪 `origin/feature/moe-salaad`。

先读下方 [当前状态与后续讨论](#current-state)。本次按用户“更新下我们的讨论”归档
10 月 12–13 日的实验回顾、自由残差问题、L2 penalty 结果，以及 P 更新方式的纠正。
**当前采用 dense 残差，不分解为 L+S；P 使用闭式最小二乘、正截断和软 Sinkhorn。**
初始化保持 X 为专家均值、全部 P_e=I、X_e=W_e-X、乘子为零。
三个 gradient + L2 实验均已完成；用户澄清本意是闭式更新后，三个相同 λ 的新实验已重新提交。
新实验最近一次核验为 **2026-10-13 09:11:57（Asia/Shanghai）的启动快照**，不是最终结果；
本次记录复用已有核验数据，未重新查询在线状态。

后文保留 [10 月 12 日 shared baseline 结果归档](#history-through-20261012)、
[10 月 9 日 P_e 分析、baseline 设计与启动快照](#history-through-20261009)、
[10 月 5 日讨论、10 月 6 日归档的快照](#history-through-20261006)、
[10 月 5 日较早的 L/S 方案](#history-20261005-early)、
[9 月 23 日梯度版本](#history-through-20260923)、[9 月 22 日提交与启动快照](#history-through-20260922)、
[截至 9 月 21 日的讨论](#history-through-20260921) 和
[9 月 17 日及以前的历史快照](#history-through-20260917)。
历史段落的“当前”均指当时；旧的 cosine 初始化、L/S、梯度 P、网络阻塞或“尚未提交”不覆盖本节决定。

<a id="current-state"></a>

## 2026-10-13：dense 残差的 L2 约束、结果分析与闭式 P 重跑

### 1. 实验回顾与比较口径

10 月 12 日的实验清单覆盖当时仍保留的 **18 个正式 W&B 训练 run：17 个完成、1 个在第 689 步后失败**，
另列 2 个短任务。这个计数不包含随后提交的 3 个 gradient + L2 和 3 个闭式 P + L2 任务。
全部配置、最终指标和失败状态见 [完整清单 JSON](../data/moe_analysis/experiment_inventory_20261012/experiment_summary.json)、
[实验表 CSV](../data/moe_analysis/experiment_inventory_20261012/experiments.csv) 及
[逐 run 配置快照](../data/moe_analysis/experiment_inventory_20261012/configs/)。
已删除且无法恢复结果的 run 不计入；纯分析、evaluation 和软件 smoke test 不算正式训练。

用户随后要求先排除 L+S；该快照中对应的 9 个完成实验如下，均训练 2100 步。
NLL 是完整 W_e 前向的训练语言模型损失，不包含路由辅助损失；末 100 步为 2001–2100 步的算术平均。

| 实验 | 主要配置差异 | Condor job | W&B | 最后一步 NLL | 末 100 步 NLL |
|---|---|---|---|---:|---:|
| Vanilla，WD=0 | 不启用 SALAAD；64 routed / top-8 | 17573606.0 | [1e417403](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/1e417403) | 4.17018 | 4.18284 |
| Vanilla，WD=0.1 | 仅将 vanilla 的 weight decay 改为 0.1 | 17676080.0 | [5a6216c8](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/5a6216c8) | 4.15487 | 4.16911 |
| Dense 残差 + 软 Sinkhorn，WD=0 | P=I 初始化；最小二乘、正截断、最多 150 轮 Sinkhorn；P→X | 17675385.0 | [cb0e00d6](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/cb0e00d6) | 4.16074 | 4.17454 |
| Dense 残差 + 软 Sinkhorn，WD=0.1 | 相同 dense 残差与软 P 算法；weight decay=0.1 | 17675621.0 | [bd49dedf](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/bd49dedf) | 4.15906 | 4.17264 |
| Dense 残差 + 硬 Hungarian | P=I 初始化；硬 Hungarian；结构顺序 X→P→残差→乘子 | 17682842.0 | [40b62f8a](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/40b62f8a) | 4.15842 | 4.17226 |
| Dense 残差 + Sinkhorn 上限 10000 | P=I 初始化；Sinkhorn 最多 10000 轮，达标提前停止 | 17684559.0 | [0221dbae](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/0221dbae) | 4.16191 | 4.17559 |
| Dense 残差 + cosine 初始化 | cosine-Hungarian 初始化 P；每次固定 150 轮软 Sinkhorn，不提前停止 | 17687476.0 | [81e36dbd](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/81e36dbd) | 4.17078 | 4.18368 |
| Shared expert 1+7 | 1 shared + 63 routed；每 token 激活 1+7；关闭 SALAAD | 17690460.0 | [807d6bb1](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/807d6bb1) | 4.16865 | 4.18209 |
| Shared expert 1+8 | 1 shared + 63 routed；每 token 激活 1+8；关闭 SALAAD | 17690461.0 | [80378abb](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/80378abb) | 4.17549 | 4.18875 |

显式 shared-expert MLP 与权重层面的 consensus X 是不同机制；后者仍由完整 W_e 执行 routed 前向。
与 Bingcong 的论文结果作比较时，需要先对齐模型、数据、训练预算和评估口径；
本地 training NLL 或其指数不能直接与论文的 held-out/test PPL 相减，现有数字不足以确定严格复现差距。

**Dense + 硬 Hungarian** 使用单位矩阵初始化、X→P→X_e→U 的辅助更新顺序，
联合 gate/up/down 的通道匹配代价，用 Hungarian 求一个离散置换，不是将 P 固定为 I。
最终 step 2100 保存的 **512/512 个 P_e 都是单位矩阵**，共执行 210 次结构更新。
由于没有逐次保存完整 P，这只能说明最终结果，不能据此断言整个训练中 P 从未改变。
见 [最终排列核验](../data/moe_analysis/hungarian_20261007/permutation_snapshot_audit.json)。

### 2. 优先解决的问题：自由 X_e 抵消了共享约束

沿用 W_e（完整专家权重）、X（consensus）、X_e（dense 残差）、P_e（通道映射）和
U_e=Y_hat_e/rho。文档方向下，gate/up 的共享映射为 XP_e，down 为 P_e^T X；
以下统记为 M_e(X,P_e)，具体存储转置不影响 Frobenius 范数。

当 λ=0 且 U 初始为零时，自由残差更新为 X_e=W_e-M_e，随后 U=0。
供下一段 W 更新使用的 anchor Q_e=M_e+X_e-U_e 因此等于最近一次结构更新时的 W_e。
两次结构更新之间仍有 rho(W_current-W_last_structure) 的近端作用，
但精确计算下，这个作用不随 X/P 的具体分解改变。P 或 X 的变化可以被 X_e 完全吸收。
因此，**重构误差接近零本身不能证明学到了有效共享**。
此前冻结 W、改变 X/P 的检查也得到几乎不变的 anchor，
见 [自由残差抵消检查](../data/moe_analysis/hungarian_20261007/free_residual_anchor_check.json)。

这被确定为先处理的结构问题。其他问题仍需区分：软 P 可能混合通道而不是接近置换，
X、XP 或 X_e 的范数不能单独代表功能贡献，完整 W 的训练 NLL 也不能替代重构模型或验证集评估。
先加入 X_e 约束，再检查可行性、任务指标和 P 的行为，不同时改动多个训练参数。

### 3. L2 penalty 的位置、初始化与当前更新公式

残差目标增加

$$
\frac{\lambda}{2}\sum_{\ell,q,e}\|X_{\ell,q,e}\|_F^2,
$$

其中 q 遍历 gate/up/down；这是所有层、投影、专家及矩阵元素的平方和，**不是均值**。
保持训练前初始化 X 为同层同投影的专家均值、全部 P_e=I、X_e=W_e-X、U_e=0，
不固定参考专家；初始化时不提前收缩 X_e，也不先将 P 软化。

从第 10 个 optimizer step 起每 10 步更新一次结构，顺序为
**W → P → X → X_e → U**。设本轮 W 已更新，其他旧变量固定：

$$
R_e=W_e^{new}-X_e^{old}+U_e^{old}.
$$

当前 P 步联合 gate/up/down 做闭式最小二乘，候选正截断后经软 Sinkhorn；
X 步再用新的 P 和同一个 R 解共享最小二乘。随后在原残差更新位置施加 L2 近端：

$$
X_e^{new}=\frac{\rho}{\rho+\lambda}
\left[W_e^{new}-M_e(X^{new},P_e^{new})+U_e^{old}\right],
$$

$$
U_e^{new}=U_e^{old}+W_e^{new}-M_e(X^{new},P_e^{new})-X_e^{new},
\qquad
Q_e^{new}=M_e(X^{new},P_e^{new})+X_e^{new}-U_e^{new}.
$$

这里 penalty 作用于辅助变量 X_e，不是直接对模型 W 加同名 weight decay。
本轮 rho=1e-5，λ=1e-3、1e-4、1e-5 的单次候选收缩系数分别为 **1/101、1/11、1/2**。
这些系数作用于含 U 的残差候选，不能直接解释为最终范数相对 λ=0 的比例。
结构更新后应满足 rho U_e≈λ X_e；这是实现检查条件，不等于原始重构约束已经收敛。
λ 的强度取决于 λ/rho、求和口径和更新频率，没有脱离这些条件的通用最优值。

### 4. 已完成的 gradient + L2 三组结果

最初按“gradient 后接软 Sinkhorn”的描述实现并运行，代码提交为 `2a2169f`。
三组训练均完成 2100 步；10 月 13 日查到训练和对应 norm 分析作业的 ExitCode 均为 0。
这批 P 候选采用一次梯度步，η=1，然后正截断和软 Sinkhorn；它们保留为已完成的对照实验。

| λ | 训练 job | W&B | 最后一步 NLL | 末 100 步 NLL | norm 分析 job |
|---|---|---|---:|---:|---|
| 1e-03 | 17695197.0 | [50913562](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/50913562) | 4.31339 | 4.32713 | 17696689.0 |
| 1e-04 | 17695198.0 | [5fecffff](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/5fecffff) | 4.29120 | 4.30466 | 17696690.0 |
| 1e-05 | 17695199.0 | [38256758](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/38256758) | 4.19431 | 4.20795 | 17696691.0 |

最终 checkpoint 的范数汇总如下。X 在每层每投影只统计一次（8×3=24 个矩阵）；
XP、X_e、W 均堆叠全部专家（8×3×64=1536 个矩阵）。
各列先将矩阵范数平方求和再开方，不是逐矩阵范数的算术平均。
表中 XP 也包含 down 的 P^T X。若把 X 对 64 个专家重复展开，其范数是表中 X 的 8 倍；
因此不能把 X 与 XP 两列直接当成相同统计口径的“共享比例”。
令 r=W-M(X,P)-X_e，最后一项范数比为 ||r||_F/||W||_F。

| 实验 | ‖X‖_F | ‖XP‖_F | ‖X_e‖_F | ‖W‖_F | ‖r‖_F / ‖W‖_F | 末 100 步 NLL |
|---|---:|---:|---:|---:|---:|---:|
| 旧 λ=0，闭式 P | 64.9464 | 49.8087 | 160.4027 | 165.4817 | 2.101e-08 | 4.17559 |
| 梯度 P，λ=1e-05 | 74.4875 | 55.7561 | 78.3250 | 101.5692 | 0.2277 | 4.20795 |
| 梯度 P，λ=1e-04 | 135.6113 | 105.0273 | 22.6212 | 66.1503 | 1.2571 | 4.30466 |
| 梯度 P，λ=1e-03 | 185.5614 | 152.2385 | 3.5847 | 65.5151 | 2.0929 | 4.32713 |

**penalty 确实压低了 X_e，但当前梯度版本没有因此得到更好的训练结果。**
同批三组除 λ 外设置一致；λ 从 1e-5 增至 1e-3 时，X_e 范数减少约 95.4%。
λ=1e-5 相比旧 λ=0，X_e 减少约 51.2%，同时 W 也减少约 38.6%，
末 100 步 NLL 增加 0.03236；不能把残差变小全部解释为有用信息进入了 X。

按 λ=1e-5、1e-4、1e-3 排列，最终相对重构误差约为 **22.8%、125.7%、209.3%**，
强 λ 下 X 和 XP 反而明显变大。
这些误差在最终结构更新之后测量，后期曲线亦维持较高水平，
不能仅归因于 W 刚更新、辅助变量尚未刷新的瞬时偏差。
日志中未缩放乘子 Y_hat=rho U 的范数与 λ||X_e|| 的相对偏差小于 3e-7，支持 penalty 更新按公式生效；
可行性和训练指标仍需要另外判断。
三组 P 的平均 Frobenius 范数约 1.72–1.73，远低于 176 维置换矩阵的 sqrt(176)≈13.27，
说明软矩阵的质量分散，不能将其直接解释为清晰的硬通道对应。

旧 λ=0 对照为 **job 17684559.0，W&B [0221dbae](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/0221dbae)**。
它使用闭式 P，clip_min=1e-8；上述三组用梯度 P，clip_min=1e-5。
旧配置省略 early_stopping 时默认也是 true。因此它与新三组**不是仅改变 λ 的严格消融**，
不能把全部 NLL 差距归因于 penalty。旧配置指纹已用实际保存的训练配置核对；
W&B 中整数/浮点 JSON 表示差异造成的 inventory 指纹不同不是训练配置改变的证据。

可复查资料：

- [三组结果与 norm 统计](../data/moe_analysis/gradient_l2_norms_20261013/comparison.json)
  / [与旧 λ=0 的配置差异和结果比较](../data/moe_analysis/gradient_l2_norms_20261013/comparison_with_unregularized.json)
- [X 与 XP 分列 CSV](../data/moe_analysis/gradient_l2_norms_20261013/comparison_with_unregularized_x_xp.csv)
  / [范数与重构误差曲线 PNG](../data/moe_analysis/gradient_l2_norms_20261013/penalty_norm_curves.png)
  / [PDF](../data/moe_analysis/gradient_l2_norms_20261013/penalty_norm_curves.pdf)
- [原 gradient 三任务提交记录](../data/moe_analysis/gradient_l2_lambda_sweep_20261012/submissions.json)

### 5. Gradient 与闭式 P 的区别，以及用户纠正后的决定

对联合最小二乘子问题 (1/2)||AP-B||_F²，闭式候选为 A†B；
梯度版本仅做 P_old-η A^T(AP_old-B)，η=1。两者之后都正截断并做软 Sinkhorn。
梯度步针对除去公共 rho 的目标，实际步长没有额外乘上 1e-5。
一次梯度步通常不能等同于解完最小二乘；闭式仅保证投影前的无约束子问题，
截断和 Sinkhorn 之后仍不能保证整个结构目标单调下降。

固定 A/B 且不做投影的二次问题，其常用收敛步长条件为 0<η<2/||A||_2²。
先前从 λ=1e-5 最终 X 得到的逐层曲率下界约为 2.94–4.21，
对应允许步长上限的上界约为 0.48–0.68，提示 η=1 值得检查。
这只是冻结子问题的风险信号，不能单独证明整个交替算法发散，
也不能据此认定梯度步长是上述训练退化的唯一原因。

用户随后明确：“我的本意是使用闭式更新。修改一下，然后重新跑这三个实验，其余参数全都保持不变。”
**当前决定是恢复闭式 P 候选，再正截断和软 Sinkhorn，重跑相同三个正 λ。**
已有求解器包含这条路径，本次改动配置和提交入口即可；保留原 gradient 配置与结果。
只替换 P 更新规则，去掉不再适用的 P learning_rate 字段，模型 AdamW 学习率保持不变。
本次未另行提交 gradient λ=0 对照或新的 evaluation。

### 6. 已提交的闭式 P + L2 三组实验

代码提交 **`363d5ef`** 已通过 git 同步到 origin 和集群的 feature/moe-salaad。
新增 [共同配置](../configs/ns97m_sinkhorn_closed_form_l2.yaml)、三个 λ 配置和对应 Condor 文件，
并更新 [算法实现说明](../docs/moe_sinkhorn_alignment.md)。
新实验名和输出目录与 gradient 版本分开，避免覆盖已完成结果。

| λ | 配置 | Condor job | W&B | 最近核验的启动状态 |
|---|---|---|---|---|
| 1e-3 | [配置](../configs/ns97m_sinkhorn_closed_form_l2_lambda1e-3.yaml) | **17696741.0** | [310b9896](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/310b9896) | Running / step 10 |
| 1e-4 | [配置](../configs/ns97m_sinkhorn_closed_form_l2_lambda1e-4.yaml) | **17696742.0** | [c6289d83](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/c6289d83) | Running / step 11 |
| 1e-5 | [配置](../configs/ns97m_sinkhorn_closed_form_l2_lambda1e-5.yaml) | **17696743.0** | [a22c9035](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/a22c9035) | Running / step 2 |

提交时间为 **2026-10-13 09:07:20–09:07:22（Asia/Shanghai）**；
表中状态核验时间为 **同日 09:11:57**。三组实际保存的完整配置均与计划一致，
P 规则均为 closed_form_clip_sinkhorn，λ 均正确，已检查日志未见错误行。
前两组已执行 step 10 首次结构更新；λ=1e-5 在该快照中尚未达到首次更新。
这里只归档启动证据，**没有闭式重跑的最终 NLL 或 norm 结果**。

与对应 gradient 实验逐项核对，保留：

- 原 97,194,240 参数模型：8 层、hidden 256、16 heads、64 routed/top-8、expert FFN 176，
  同一 DCLM token 数据、顺序和 seed 42；seq_len=2048。
- AdamW peak LR=3e-4、min LR=3e-5、21 步 warmup、cosine schedule、weight decay=0；
  load-balancing 系数 0.1、router z-loss 系数 0.001。
- global batch=1024、DP4、每卡 micro batch=16、梯度累积 16；
  2100 optimizer steps，共 4,404,019,200 个预测 token。
- rho=1e-5、每 10 步结构更新、均值/I/差值/零乘子初始化；
  Sinkhorn temperature=1、clip_min=1e-5、max_iterations=10000、
  marginal_tolerance=1e-5、early_stopping=true，末尾不做 Hungarian 硬化。
- 每任务 4 H100、16 CPU、64000 MB、bid 100，保留相同机器筛选条件和其他训练设置。

提交前已完成完整配置差异检查、三份 CLI dry-run、11 项既有 Sinkhorn 测试，
并针对 λ/rho=100、10、1 做三组各 8 步的 CPU 联合 smoke。
初始化均核验通过，最大 KKT 绝对误差约 1.86e-9，最大 Sinkhorn 行列和误差约 9.30e-6。
测试支持实现与配置符合本次决定，不提前代表正式训练收敛。
见 [验证摘要](../data/moe_analysis/closed_form_l2_lambda_sweep_20261013/validation.json)、
[计划及完整配置指纹](../data/moe_analysis/closed_form_l2_lambda_sweep_20261013/planned_submissions.json)、
[提交记录](../data/moe_analysis/closed_form_l2_lambda_sweep_20261013/submissions.json) 和
[启动核验](../data/moe_analysis/closed_form_l2_lambda_sweep_20261013/startup_verification.json)。

### 7. 后续分析与工作约定

下一步等闭式三组完成，沿用同一口径比较 X、XP、X_e、W、重构误差、乘子、P 的集中程度和训练 NLL。
优先比较相同 λ 的 gradient/闭式两组；与历史 λ=0 比较时仍需注明 clip_min 不同。
如果以后需要严格的 λ-only 消融，应另建同配置的 λ=0；这是待讨论的实验，不是已提交任务。
统一 validation/下游评估及 shared 功能贡献分析也仍是后续工作。

MPI 连接沿用 local_scripts/mpi_vpn_socks.sh 的本地 SOCKS 11080 路径，可保留 ETHZ/Cisco VPN；
重跑提交前已核验集群连接。此次更新文档没有重新连接 VPN 或重提任务。
随后用户要求“保存git状态并推送”：本主记录和讨论索引纳入 Git，随本次提交推送到 origin。
其余讨论附件、分析数据和训练产物继续由 gitignore 忽略，保留在原位置。
**仅在用户要求时更新讨论**。旧 PDF 保留各自历史版本，不用旧描述覆盖本节最新决定。

<a id="history-through-20261012"></a>

## 2026-10-12 归档：shared 1+7 / 1+8 的 W&B 训练结果

### 1. 完成状态与比较口径

用户在 10 月 10 日要求查看这两项 W&B 结果，本节记录该次查询和分析。
两组均已完成 **2100 optimizer steps、4,404,019,200 个预测 token**，W&B 状态均为 `finished`。
数据核验于 **2026-10-10 08:20–08:23（Asia/Shanghai）** 完成；
这里记录 W&B 的完成状态，没有用该状态代替新的 Condor history / ExitCode 核验。

两组 W&B 配置仅实验名称和 routed top-k 不同，学习率逐步一致；
沿用相同初始权重、数据顺序、seed 42 和 weight decay=0。
每层都是 1 shared + 63 routed，分别激活 1+7、1+8 个 expert。
比较指标为 **`train/lm_nll`，不包含 load-balancing loss 或 router z-loss**。
末 100 步指第 2001–2100 步，末 500 步指第 1601–2100 步，均为这些训练 batch 的算术平均。

### 2. 结果与观察

| 指标 | Shared 1+7 | Shared 1+8 |
|---|---:|---:|
| Condor job 标识 | 17690460.0 | 17690461.0 |
| W&B | [807d6bb1](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/807d6bb1) | [80378abb](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/80378abb) |
| W&B 状态 / 最终步数 | finished / 2100 | finished / 2100 |
| 最后一步训练 NLL | **4.16865** | 4.17549 |
| 末 100 步平均训练 NLL | **4.18209** | 4.18875 |
| 末 500 步平均训练 NLL | **4.20522** | 4.21257 |
| W&B 记录运行时间 | 4.82 小时（17352.61 秒） | 4.99 小时（17951.14 秒） |
| 末 100 步平均 step_seconds | 7.165 秒 | 7.341 秒 |

末 100 步逐步对齐后，**100 / 100 步均为 1+7 的训练 NLL 更低**，
平均差 `NLL(1+8) - NLL(1+7)` 为 **0.00666233**，逐步差值范围约 0.00522–0.00798。
这次训练中，1+8 多激活一个 expert 没有带来训练 NLL 收益；保持 **1+7 为主 baseline**、
1+8 为额外专家计算预算的补充对照。

两组训练曲线整体下降，记录的所有数值指标均有限，无 NaN/Inf。
末 100 步各层 routed assignment CV 的均值约为 **1.1%–2.0%**，路由负载分配较均衡；
这项计数统计不能单独证明不同 expert 学到了不同功能。
W&B 总运行时间中 1+8 比 1+7 高约 **3.45%**；两组使用不同 worker，
该实测值不能直接作为纯架构开销结论，也不能与专家 MLP 理论计算量增加 12.5% 等同。

本次只分析了 **单个 seed 的训练指标**。两个 run 中没有验证集或下游评测指标；
尚不能据此判断泛化优劣、consensus 相对 shared baseline 的收益，或 X 承担了多少知识。
下一步仍是将 vanilla non-shared、最新 consensus、shared 1+7 和 shared 1+8
放到相同验证集与下游任务中 evaluation。此项尚未因本次记录而提交。
shared/routed 分支输出和梯度范数的独立诊断仍是后续建议，未新增相关日志或缩放消融。

### 3. 可复查的数据与图表

分析保存于 `data/moe_analysis/shared_wandb_20261010/`。
核验时去除了 W&B 分页边界的相同行，并单独补读分页结果缺失的第 1668 步；
最终每组均具有 **1–2100 的完整、唯一记录**，最终 NLL 与 run summary 一致。

- [完整统计 JSON](../data/moe_analysis/shared_wandb_20261010/comparison.json)
- [训练曲线 PNG](../data/moe_analysis/shared_wandb_20261010/training_comparison.png) /
  [PDF](../data/moe_analysis/shared_wandb_20261010/training_comparison.pdf)：100 步移动平均、后期放大和两组差值。
- [逐步对齐比较 CSV](../data/moe_analysis/shared_wandb_20261010/comparison_by_step.csv)
- [1+7 完整指标 CSV](../data/moe_analysis/shared_wandb_20261010/807d6bb1_history_complete.csv) /
  [1+8 完整指标 CSV](../data/moe_analysis/shared_wandb_20261010/80378abb_history_complete.csv)
- [1+7 W&B 配置与 summary](../data/moe_analysis/shared_wandb_20261010/807d6bb1_metadata.json) /
  [1+8 W&B 配置与 summary](../data/moe_analysis/shared_wandb_20261010/80378abb_metadata.json)

下方 10 月 9 日的 Running / step 1、3 和原提交记录保留为启动时快照，
不覆盖本节完成状态。本次只更新本地讨论记录及索引，继续沿用“仅在用户要求时记录”的约定。

<a id="history-through-20261009"></a>

## 2026-10-09：最终 P_e 检查与 shared-expert baseline

本节保留当时的分析、设计及启动快照；两个 shared run 的完成结果见上方最新归档。

### 1. 记录范围与符号约定

只在用户明确要求记录时更新讨论或问答文档；本次授权不改成自动记录后续问题。
沿用用户文档符号：W_e 为完整专家权重，X 为 consensus，X_e 为 expert-specific 残差，
P_e 为通道映射，Y_hat_e 为乘子。不为这些量另换名称。

本轮研究从“检查 P_e 是否仍趋向单位排列”推进到“增加显式 shared expert 作为对照”。
权重层面的 consensus X 与前向中的 always-on shared MLP 是不同机制。
当前 consensus 模型仍使用完整 W_e 执行 8 个 routed experts 的前向计算，
不额外执行一个以 X 为权重的 shared MLP，不能把它计为第 9 个激活 expert。

### 2. 最新 consensus 训练与最终 P_e

最新完成的训练为 **17687476.0**，最终 checkpoint 为 **step 2100**，正常退出。
[训练配置](../configs/ns97m_sinkhorn_cosine_init.yaml) 使用 seed 42、weight decay=0，
[W&B run 81e36dbd](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/81e36dbd)。
训练前，以原始专家均值为模板，按通道 cosine 相似度对每个专家做一次匈牙利匹配得到初始 P_e；
再根据对齐关系初始化 X、X_e 和零乘子。此过程不修改原始 W_e。
之后每 10 个 optimizer steps 更新一次辅助状态，共 210 次；
P_e 按闭式最小二乘、正截断、**固定 150 轮 Sinkhorn** 更新，容差 1e-5，
`early_stopping=false`，训练中不在 Sinkhorn 之后追加匈牙利硬化。
旧文档里的 P_e=I 初始化和“最多 150 轮、可提前停止”属于先前版本。

2026-10-09 的分析作业 **17690447.0** 已完成：对最终全部 8 层 × 64 专家的
176×176 软 P_e，以保存元素为分数直接做最大权重匈牙利匹配；不取对数、不设置单位排列优先回退。
投影结果单独保存，原训练 checkpoint 未被修改。

| 检查项 | 最终结果 |
|---|---:|
| 匈牙利投影后为单位矩阵 | **0 / 512** |
| 匈牙利投影后非单位矩阵 | **512 / 512** |
| 相对单位排列发生换位的通道 | **50,520 / 90,112（56.06%）** |
| 每个投影矩阵换位通道数 | 平均 98.671875；最少 52、最多 146 |
| 匈牙利目标值相对单位排列的增益 | 每个矩阵均为正；最小 1.875154 |
| 软 P_e Frobenius 范数 | 平均 1.685970；范围 1.425752–2.041974 |
| 软 P_e 每行最大元素的均值 | 0.062395 |
| 软 P_e 对角元素的均值 | 0.035265 |
| 对角元素为该行严格最大值的比例 | 33.5771% |
| 软 P_e 最大行和/列和误差 | 分别约 3.50e-7 / 3.67e-7 |

与相同 seed、相同模型及初始化规则的独立第 0 步初始化分析相比，
512 个最终硬排列全部发生变化，共 **27,447 / 90,112（30.46%）** 个通道对应关系改变。
这个比例比较“最终硬排列与初始排列”；上表 56.06% 比较“最终硬排列与单位排列”，不要混用。
初始排列来自 `initial_alignment_20261008/17687237_0` 的 `unaligned_mean_cosine`，
并非从本次训练另存的第 0 步 checkpoint 读取。

结论限于：最终硬投影没有全部回到单位矩阵，也没有逐矩阵原样保留初始排列。
软 P_e 仍然分散，Frobenius 范数远低于纯置换矩阵的 sqrt(176)=13.266499。
这些结构统计不能单独证明知识集中在 X，或证明任务性能优于 baseline；
本轮没有对这个新 checkpoint 执行硬化后的语言模型 evaluation。

证据：[软矩阵统计](../data/moe_analysis/cosine_init_final_pe_20261009/17690447_0/pe_row_maxima.json)、
[匈牙利投影统计](../data/moe_analysis/cosine_init_final_pe_20261009/17690447_0/hungarian_projection.json)、
[初始/最终排列比较](../data/moe_analysis/cosine_init_final_pe_20261009/17690447_0/initial_comparison.json)、
[本地独立核验](../data/moe_analysis/cosine_init_final_pe_20261009/17690447_0/local_validation.json)。

### 3. 为什么同时跑 1+7 和 1+8

用户提出：需要显式 shared expert baseline，才能衡量自己的 consensus 方法相对它的收益。
讨论决定：**1+7 为主对照，1+8 为额外计算预算下的补充对照；两组同时提交。**
每层仍保留 64 份同宽 expert MLP 权重：把其中一份用于 shared，剩余 63 份参与路由。
shared 与 routed 的 FFN 中间宽度都为 176。

| 方案 | 专家池 | 每个 token 执行 | 专家 MLP 理论计算量 |
|---|---|---|---:|
| vanilla non-shared | 64 routed | top-8 routed | 1.00× |
| 当前 consensus | 64 routed | top-8 routed | 1.00× |
| shared 主 baseline | 1 shared + 63 routed | 1 + top-7 routed | 1.00× |
| shared 补充 baseline | 1 shared + 63 routed | 1 + top-8 routed | 1.125× |

上述计算比例仅指同宽 expert MLP 的理论前向工作量，不能直接当作整模型 FLOPs 或训练时长比例。
consensus 的额外结构更新开销应另外统计。
两组 shared 模型总参数均为 **97,192,192**，原 non-shared 为 **97,194,240**；
专家权重总量完全一致，差额 2048 来自每层 router 少一个输出、共 8 层 × hidden 256。
若保留 64 个 routed 再另加一个 shared，会变成 65 份 expert 权重；本次没有采用这一设置。

这一预算对齐方式也与 [DeepSeekMoE 原论文 §3.2](https://arxiv.org/html/2401.06066v1#S3.SS2)
从专家池划出 shared experts、相应减少 routed 激活数量的原则一致。
最终应以相同训练 token 数和相同评测下的 NLL/PPL、下游表现来判断收益；
一组 seed 的结果不能单独说明统计稳定性，更不能直接证明 X 承担了大部分知识。

### 4. Shared 与 non-shared 的实现区别

新增 always-on SwiGLU 分支，shared 始终处理所有 token。令 h 为送入 MoE 的 token 表示，
f_e 为第 e 个 routed expert 的 MLP，f_shared 为 shared expert 的 MLP，
I_K(h) 为 router 在 63 个 routed experts 中选择的 top-K 集合，g_e 为选中后的 softmax 权重。
当前 shared 版本输出为：

```text
y = f_shared(h) + sum_{e in I_K(h)} g_e * f_e(h)
sum_{e in I_K(h)} g_e = 1,    K = 7 或 8
```

shared 输出系数固定为 1；没有单独的 shared gate，也不把 shared 与 routed 输出一起再做归一化。
non-shared 版本只包含 routed 加权和。加入 shared 后输出尺度可能变化，后续应观察两条分支的贡献。

必须随结构调整的内容均已实现：

- router 输出维度从 64 改为 63，选择数量分别为 7、8；`model.num_experts` 只表示 routed 池。
- load-balancing loss 只统计 63 个 routed experts，路由分配比例分母为 token 数 × 对应 top-k；
  shared 不进入负载均衡。Router z-loss 也只由 routed logits 计算。
- shared 使用相同的 SwiGLU 结构和初始化规则；gate/up 标准差 0.02，down 采用同样的
  `std_div_sqrt_2depth` 缩放。shared 参数纳入 AdamW、全局梯度裁剪、DP 同步和完整 checkpoint。
- 两组 shared baseline 均关闭 SALAAD，因此没有 X、X_e、P_e 或乘子的结构更新；
  当前未实现 shared expert 与 SALAAD 同时启用，配置校验会拒绝这种组合。
- 原 non-shared / consensus 路径继续可用。新增 shared baseline 走当前原生 PyTorch 训练器，
  Megatron 适配器尚未支持这个新结构，会明确拒绝而不是忽略 shared 配置。

shared 每个 token 都有梯度，routed 只由选中它的 token 产生任务梯度。
本次没有因此给 shared 设置不同学习率，也没有按路由频率额外缩放其梯度。
保持相同训练配方先建立基线；shared 分支缩放、单独学习率等若以后调整，应作为独立消融。

### 5. 配置、训练参数及提交证据

| 设置 | 1+7 | 1+8 |
|---|---|---|
| 配置 | [ns97m_shared_1plus7.yaml](../configs/ns97m_shared_1plus7.yaml) | [ns97m_shared_1plus8.yaml](../configs/ns97m_shared_1plus8.yaml) |
| family | `llama_style_shared_expert` | 相同 |
| num_experts / num_shared_experts | 63 / 1 | 63 / 1 |
| router_topk | 7 | 8 |
| Condor job | **17690460.0** | **17690461.0** |
| W&B | [807d6bb1](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/807d6bb1) | [80378abb](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/80378abb) |
| 提交后启动核验 | step 1，NLL 10.886858 | step 3，NLL 10.769493 |

任务状态表是 **2026-10-09 09:39（Asia/Shanghai）** 的最后核验快照：两组均为 Running，
已输出有限训练指标且 W&B `mode=online`。这里的早期 NLL 位于不同步数，不能用于比较最终效果。
本次归档未重新查询实时进度，也没有新增最终评估结果。

两组除实验名称与 routed top-k 外，展开配置一致；从同一 seed 生成完全相同的初始模型权重，
使用相同语料与数据顺序。与原 non-shared 比较时，保留相同 seed 和初始化分布，
但架构及参数形状有变化，不声称跨架构每一份初始权重逐元素相同。

共用训练设置：8 层、hidden 256、FFN 176；seed 42；2100 optimizer steps；
global batch 1024、每卡 micro batch 16、梯度累积 16、序列长 2048，共 **4,404,019,200** 个预测 token。
AdamW：peak LR=3e-4、min LR=3e-5、21 步 warmup 后 cosine，betas=(0.9, 0.95)、epsilon=1e-8、
**weight decay=0**，全局梯度裁剪 1.0；负载均衡系数 0.1，router z-loss 系数 0.001。
BF16 任务计算、FP32 参数和梯度；每 100 步保存，仅保留最新完整 checkpoint。
每组 **DP4 / 4 张 H100 80GB / 16 CPU / 64,000 MB / bid 100**；独立任务、无前后依赖。
`is_wandb=true`，提交环境明确设置 `WANDB_MODE=online`，分别保存独立 run。

代码提交 **2433bdf60ec33039156013ec277349ae69ced2e8** 已在本地、origin、集群同一分支同步。
提交前 **155 项 MoE 单元测试通过**；另通过两种 top-k 的双进程 BF16 训练、
模型/Adam/读取位置逐位恢复和 DP 副本一致性检查。
数值检查覆盖 shared+routed 输出及梯度、参数量、raw checkpoint evaluation、两个 top-k 初值一致。
这些是软件检查，不能作为最终模型质量比较结果。

训练数据继续使用集群项目下的 `data/moe_corpora/dclm_20260916/tokens_ns97m/manifest.json`。
集群输出分别为：

```text
/lustre/home/hma2/projects/awd-LoRA/data/moe/ns97m_shared_1plus7/17690460_0
/lustre/home/hma2/projects/awd-LoRA/data/moe/ns97m_shared_1plus8/17690461_0
```

本地提交记录、完整配置、配置哈希及 W&B 核验：
[1+7 记录](../data/moe_analysis/shared_expert_baselines_20261009/ns97m_shared_1plus7/submission.json)、
[1+8 记录](../data/moe_analysis/shared_expert_baselines_20261009/ns97m_shared_1plus8/submission.json)。
集群连接继续使用 `bash local_scripts/mpi_vpn_socks.sh ssh`，经本机 SOCKS 11080 访问 MPI，保留 ETHZ/Cisco。
双 VPN 曾阻塞 SSH 的旧记录属于历史；本轮已通过该方式同步和提交成功。

### 6. 后续分析及记录约定

训练结束后，将 vanilla non-shared、最新 consensus、shared 1+7、shared 1+8 放在同一评测设置下比较。
主要报告不含路由辅助项的验证 NLL/PPL，以及相同下游任务、训练 token 数和实际耗时。
1+7 用于接近相同前向预算的比较；1+8 用于观察额外 expert 计算预算下的对照表现。

讨论建议另外观察 shared 输出和 routed 合并输出的范数比例、分支梯度大小及路由使用情况。
目前只保留既有总梯度和 routed 路由日志，**尚未新增分支级输出/梯度范数日志，也未开展缩放消融**。
这些是后续建议，不表示已提交额外分析作业。

本目录和 `data/moe_analysis` 为本地忽略目录。本次讨论记录未 commit/push，也未同步到集群；
上述代码同步与任务提交是已经完成的实验操作，不能与笔记保存状态混淆。

<a id="history-through-20261006"></a>

## 2026-10-05 讨论、10 月 6 日归档：完整 ADMM、代码复查与待提交任务（历史）

### 1. 文档与交互约定

用户要求公式问答写入 LaTeX 并生成 PDF，只保留文字和公式，使用页码和公式编号。
先说明完整流程，每个变量第一次出现时解释含义，层索引不能遗漏。
用户明确要求“在这里回答”的问题直接在聊天中答；没有新问题时不自行添加内容。

随后用户清空新 ADMM 文档并开始用英文撰写，要求沿用其顺序和符号，
审核以数学正确性为主，不纠缠语法表达，不展开 AdamW 内部细节。
当前主文档为 [英文 ADMM LaTeX](moe_salaad_admm_20261005/moe_salaad_admm.tex) /
[PDF](moe_salaad_admm_20261005/moe_salaad_admm.pdf)，本次归档时共 4 页、31 个编号公式。

| 当前 PDF 的位置 | 内容 |
|---|---|
| 第 1 页，式 (5) | 带层索引的严格置换约束 |
| 第 2 页，式 (11)–(17) | 双随机松弛、增广拉格朗日、W 更新和 P 的拟合目标 |
| 第 3 页，式 (18)–(26) | P、共享 X、专家残差 X_e、乘子更新 |
| 第 4 页，式 (27)–(31) | 标准 MoE 权重初始化、P=I、共享均值、残差和零乘子 |

[较早问答 PDF](moe_salaad_qa_20261004/moe_salaad_qa.pdf) 保留 Sinkhorn 原理、
收敛条件以及不同 P 更新方法的讨论。其中参考专家、L/S 和梯度版本的描述属于历史方案，
不能用它覆盖当前英文 ADMM 文档和下面的决定。

### 2. 最终优化问题与五步顺序

文档先写严格置换约束，再把二元约束放松为非负约束，行和与列和仍等于 1。
第 i 层第 e 个专家使用 `P_e^i`；同一专家的 up、gate、down 共用同一份 P，
不同层各自拥有共享矩阵、映射和残差。省略层索引时，
`P_hat_e = diag(P_e, P_e, P_e)`，拼接后的分解为 `W_e = X P_hat_e + X_e`。
文档的未缩放乘子记为 `Y_hat_e`，代码保存 `U_e = Y_hat_e / rho`，`rho=1e-5`。

当前采用 **W → P → 共享 X → 专家残差 X_e → 乘子 Y_hat** 五步：

1. **W**：固定辅助状态，优化网络任务损失与增广项。任务损失可能耦合专家，
   因此网络参数 theta 联合更新，再取其中的 W_e；实际使用优化器近似求解。
2. **P**：固定旧共享矩阵、旧残差、旧乘子，以
   `R_e = W_e_new - X_e_old + Y_hat_e_old / rho` 为拟合目标。
   将三个投影纵向拼接，共同求无约束最小二乘的最小范数解，
   然后逐元素截断到 `1e-8`，最后执行 Sinkhorn。
3. **共享 X**：固定新的 P，解跨专家求和的最小二乘问题。文档式 (23) 为
   `X_new = [sum_e R_e P_hat_e_new^T] [sum_e P_hat_e_new P_hat_e_new^T]^dagger`。
   求和已在公式内，结果后面不再额外求和或除以 E；硬置换的特殊情形才退化为普通平均。
4. **X_e**：当前残差不受约束、没有额外正则，直接取闭式解
   `X_e_new = W_e_new - X_new P_hat_e_new + Y_hat_e_old / rho`。
5. **Y_hat**：用更新后的原变量计算
   `Y_hat_e_new = Y_hat_e_old + rho (W_e_new - X_new P_hat_e_new - X_e_new)`。
   代入第 4 步可知，精确算术下新乘子为零；实现保留 FP32 舍入结果。

P 的规则是用户明确固定的“闭式更新、截断、Sinkhorn”，不要求该组合给出约束子问题最优解。
不恢复旧 P 移动惩罚、不做 A/P 的内部梯度迭代、不加误差下降接收筛选。
**所有专家的 P 都参与更新，`fix_reference=false`，没有固定参考专家。**
共享问题允许奇异，使用伪逆取最小范数解，不能沿用旧方案中“参考专家保证可逆”的说法。

当前 `residual_mode=dense`，L/S 分解关闭，`controller=null`，不运行低秩/稀疏控制器。
文档已明确：不限制 X_e 时，残差可以吸收全部 W，分解等式本身不限制网络权重；
这不保证共享部分有非平凡内容、分解唯一、残差很小或模型获得压缩。
本轮没有据此额外添加正则或改回 L/S。

文档中的 k 按用户要求称为 epoch。代码仍按 optimizer step 调度：
训练前初始化，从第 10 步起，每 10 步更新一次 P/X/X_e/U；没有新增按完整语料遍历计数的 epoch 调度。
W 的任务前向始终使用完整专家权重。代码矩阵布局与文档互为转置：
gate/up 的共享部分为 `P.T @ shared`，down 为 `shared @ P`，核对公式时需保持这一约定。

### 3. 初始化与 Sinkhorn 上限

用户最终指定的初始化在第一个优化器步之前逐层执行：

- W 使用对应 vanilla MoE 原有的随机初始化，不用分解变量反过来初始化 W。
- 每个专家直接设 `P_e=I`；不做硬匹配、初始软化或 Sinkhorn。
- 每个投影的共享 X 为同层专家 W 的均值。
- `X_e = W_e - X`，`Y_hat_e=0`，代码中的 U 也为零。

此时分解等式成立，增广项的权重梯度为零，允许浮点舍入误差。
辅助初始化不改变 W，也不额外消耗模型随机数。
初始单位矩阵直接保存，`alignment_logits=None`；首次 P 更新后才保存有限的
`A = temperature * log(Q)`，其中 Q 是截断后的正矩阵、temperature=1。
A 只是 log 域归一化的表示，不是独立优化变量。

用户随后要求归一化上限改为 **150**。正式及 smoke 配置均已修改，
每次迭代包含一轮行归一化和一轮列归一化；最大行/列和误差容限保持 `1e-5`。
达到上限仍未达标时，报错并保留旧辅助状态及 anchor，不静默放宽容差或增加上限。
结构更新发生在 W 的优化器步之后，出错需要从完整 checkpoint 恢复，不能直接重试同一内存训练步。

已讨论的理论边界：严格正输入的 Sinkhorn 行列缩放有收敛理论，但不保证预设 150 次足够；
这也不等于整个非凸、多块、近似 ADMM 有全局收敛保证。

### 4. 代码实现、复查和已知限制

本地已更新 `config.py`、`solver.py`、`sinkhorn.py`、`export.py`、`tracking.py`、
训练器注释、正式/smoke 配置及测试，说明见 [当前实现文档](../docs/moe_sinkhorn_alignment.md)。
新 checkpoint 保存完整残差，初始无 A 的单位矩阵状态可保存、恢复和导出；
新导出格式为 `salaad_moe.export.v5`。旧 L/S checkpoint 不能直接用当前 dense 配置续训。

用户要求再次仔细审查后，复现并修复了三类问题：

1. 推理/导出读取 checkpoint 时，缺少 `training.pt` 与 `complete.json` 的步数、配置和配置哈希一致性检查。
2. 导出共享矩阵或最终重构权重中的 NaN/Inf 未被完整拒绝。
3. 各分量有限但相加后 anchor 溢出的情况未被拒绝；现在更新和恢复都在提交状态前检查。

最后一轮 **101 项单元测试通过**；DP2 八步训练及从第 2、3、7 步恢复的逐位一致性通过，
正式配置 dry-run 通过。实际非收敛矩阵还验证了严格 150 次上限和失败后辅助状态不变。
当天早先的 90/96/98 项结果属于不同中间版本，不能替代最终 101 项结果。

另一次保持正式矩阵尺寸的 CPU/BF16 检查使用 8 层、64 专家、176 中间通道、256 hidden size，
但使用合成数据、batch=1、序列长 8。它在第 10 个优化器步、第 3 层（索引 2）
因 150 次未达 `1e-5` 而停止，最后成功完成 9 步。
最小二乘和归一化都用 FP64 独立复核时，第 150 次误差为 `2.9978719639944806e-5`，
第 175 次检查才以 `9.79983304583154e-6` 达标。
这是一个已知的 150 次不足的输入，不能仅归因于 FP32 精度，也不能推断正式 batch 必然失败。
**上限依照用户决定仍为 150；没有改成 175 或更大。**
本机 CUDA 驱动不可用，本轮没有重新验证 GPU 路径，不能声称“已确保零 bug”。

为便于以后复查，已把小型验证摘要从临时目录归档：
[验证总览](../data/moe_analysis/dense_admm_20261005/validation_summary.json)、
[DP2](../data/moe_analysis/dense_admm_20261005/dp2_validation.json)、
[CPU 短序列失败检查](../data/moe_analysis/dense_admm_20261005/formal_cpu_short_probe.json)、
[FP64 复核](../data/moe_analysis/dense_admm_20261005/sinkhorn_fp64_probe.json)。
本次写讨论记录时没有重跑测试或训练。

### 5. Baseline 核对与已授权但未完成的提交

用户明确要求：“其余参数跟 baseline 保持一致，然后提交任务。”
已核对本地 `ns97m_vanilla` 展开配置，以及 9 月 23 日归档的实际运行配置：
除实验名与 SALAAD 算法部分外，model、data、training、parallel、evaluation、seed、W&B 设置一致。
提交文件的 launcher、数据路径、CPU/GPU/内存、环境变量和硬件排除条件也与 vanilla 一致。
本次无法访问集群，因此没有重新读取远端 baseline 的实际配置或队列。

| 项目 | 保持的设置 |
|---|---|
| 模型 | 8 层、hidden 256、FFN 176、64 experts / top-8；97,194,240 参数 |
| 数据 | 同一 DCLM manifest；序列长 2048，seed 42 |
| Batch | global 1024；每卡 micro batch 16，累积 16 次，普通 DP4 |
| 预算 | 2100 个 optimizer steps |
| 优化器与学习率 | AdamW；3e-4 → 3e-5，21 步 warmup 后 cosine；原 betas、epsilon、weight decay 保持 |
| 计算与损失 | BF16 任务计算、FP32 参数；裁剪 1.0；balance/z-loss 系数 0.1/0.001 |
| 保存 | 每 100 步保存，仅保留最新 checkpoint |
| 资源与出价 | 4 张 H100 80GB、16 CPU、64,000 MB，**bid 100 已获用户授权** |
| ADMM | rho=1e-5，训练前初始化，每 10 个优化器步更新辅助变量，Sinkhorn 上限 150 |

入口：[正式配置](../configs/ns97m_sinkhorn.yaml) /
[Condor 文件](../sub/moe_ns97m_sinkhorn.sub)。
拟在集群目录 `/lustre/home/hma2/projects/awd-LoRA` 执行
`condor_submit_bid 100 sub/moe_ns97m_sinkhorn.sub`。
数据 manifest 为
`/lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916/tokens_ns97m/manifest.json`。

**实际状态：没有执行提交命令，没有新 job ID，没有新训练结果。**
本地 HEAD 仍为旧提交 `4a244be5aed18bbda9bb4404b1fd25c9c72c36d7`；
今天的 ADMM 改动仍在工作区，未 commit/push，未同步到集群。
原先的多个分析/可视化脚本属于其他未跟踪文件，后续提交代码不能把它们一并暂存。
核对和阻塞证据保存在 [preflight.json](../data/moe_analysis/dense_admm_20261005/preflight.json)，
待提交配置保存在 [config.prepared.json](../data/moe_analysis/dense_admm_20261005/config.prepared.json)。

### 6. VPN 与 SSH 排查：已证实的事实

用户要求重试集群连接后，多次在沙箱外尝试，
`login.cluster.is.localnet` 的四个地址 `10.15.1.6`、`.7`、`.41`、`.42`
均在 TCP 22 端口超时，未进入 SSH 认证阶段。

用户随后提供 OpenConnect 日志，显示 `CSTP connected`、分配 `10.1.5.250`。
进一步实测确认：MPI 的 `tun0` 已建立，集群路由走 `tun0`，
经该隧道访问内网 DNS `10.3.7.82:53` 成功，并有实际收发流量。
因此“VPN 没有连接”不是准确结论；前面的 404 已由后续成功连接处理，
`--no-dtls` 表示使用 SSL 隧道，也不能据此判断连接失败。

同时运行 ETHZ Cisco Secure Client，连接 `sslvpn.ethz.ch/staff-net`，
接口 `cscotun0`、地址 `10.6.192.124`。MPI 的公网网关 `192.124.25.4`
当时通过 `cscotun0`，即 MPI VPN 的外层连接经过 ETHZ VPN。
用户明确要求 **让 MPI 不经过 Cisco，同时保持 Cisco 连接**。

本地 Wi-Fi 为 `wlp0s20f3`、地址 `192.168.71.22`、网关 `192.168.71.1`。
指定 Wi-Fi 直连 MPI HTTPS 的预检超时，未证实本地网络到 MPI 网关能够直接连通。
为只改变 MPI 网关而保留 Cisco，准备了
[临时单地址路由脚本](../local_scripts/mpi_vpn_direct_route.sh)。
自动执行因 `sudo: a password is required` 未修改路由，随后用户在自己的终端用 sudo 执行。

用户执行后，路由仍走 Cisco，清理报 `No such process`。
进一步读取本机 Cisco 日志，已找到明确证据：
**2026-10-05 23:42:31 Asia/Shanghai，`csc_vpnagent[1131]` 自动删除了
`192.124.25.4/32 via 192.168.71.1 dev wlp0s20f3`。**
日志关键词是 `Routing table - fixed - deleted route`，随后报告自动修复路由成功。
所以失败原因不是用户没有 sudo 权限，而是 Cisco 在成功添加之后移除了这条路由。
清理时路由已经不存在，脚本已修正对此情况的误导提示，语法检查通过。
最后核验：没有遗留该临时路由，Cisco 状态仍为 Connected，未完成 MPI 直连分流。

Cisco 当前 IPv4 模式为 `Split Exclude`，统计中的 Non-Secured Routes 只有
`192.168.71.0/24` 和 `172.18.0.0/16`；MPI 网关不在其中。IPv6 为 `Tunnel All Traffic`。
普通本地静态路由方案在当前策略下不成立。
受支持的后续方案是由 ETHZ VPN 管理员允许对 MPI 网关的 split-exclude，
或配置对应域名 `vpngw.tuebingen.mpg.de` 的动态排除；参考
[Cisco 分流配置说明](https://www.cisco.com/c/en/us/td/docs/security/vpn_client/anyconnect/Cisco-Secure-Client-5/admin/guide/cisco-secure-client-admin-guide-new/configure-vpn-access-intro/c_select_and_exclude_vpn_traffic.html)。
本次没有联系管理员、修改 Cisco 配置或断开 Cisco。

**证据边界：已证明 Cisco 删除了直连路由，解释的是本次分流失败；
尚未证明它也是最初集群 SSH 超时的根因。**
相关结构化记录也保存在上述 preflight.json。

### 7. 后续继续工作的起点

1. 先恢复可用的集群 SSH 访问，保留用户关于 Cisco 的约束；不重复尝试已被自动删除的普通路由方案。
2. 用户提交授权、4 H100/DP4/bid 100 和 baseline 对齐要求继续有效，不应再次询问已有出价。
3. 连通后核对本地、origin 与集群分支及快进条件，只提交本轮训练所需改动，
   经 git push/fetch/ff-only 同步；当前不能假定集群已拥有新算法。
4. 确认配置仍为自由残差、全部 P=I 初始化、上限 150，再提交一次任务并核验 receipt、job ID 与状态。
   150 次不足的已知输入已经告知用户，不能为通过测试或提交而擅自提高上限。
5. 本次归档仅更新本地讨论、索引及小型验证摘要；没有因为写记录而执行新的网络修改或任务提交。

<a id="history-20261005-early"></a>

## 2026-10-05 较早快照：固定 P 三步更新，仍保留 L/S（已被上文替代）

以下是当天较早的原记录；其中固定参考专家、L/S 和 90 项测试均属于当时版本。
它保留 P 更新规则的讨论过程，不代表当前完整 ADMM 配置。

用户决定：“我不需要最优解，就按照闭式更新、截断、Sinkhorn 来更新 P，固定下来。”

每轮结构更新固定旧 shared X 与 `R = E-L_old-S_old+U_old`，执行：

1. 三投影联合求无约束最小二乘闭式解 P_LS；秩不足时取最小范数解。
2. `Q_ij = max(1e-8, (P_LS)_ij)`，使用正下限以便在 log 域归一化。
3. `P_new = Sinkhorn(Q)`，随后依次更新 shared、L、S、U。

参考专家 0 的 P 固定为 I。A 仅保存 `temperature * log(Q)`，参考专家 A=0；
不再做 A/P 梯度更新、旧 P 移动惩罚或重构误差下降接收判断。
代码使用拼接矩阵的一次伪逆计算各专家闭式解；一层每轮调用一次 Sinkhorn。
正式配置仍在训练前初始化，每 10 个 optimizer steps 做一次结构更新。

本地实现、配置和问答 PDF 已更新；90 项单元测试、正式配置 dry-run、DP2 训练及逐位恢复通过。
本次没有提交或恢复正式训练，以下历史实验不代表新规则的训练结果。

公式见 [当前 PDF，问题 7](moe_salaad_qa_20261004/moe_salaad_qa.pdf) /
[LaTeX](moe_salaad_qa_20261004/moe_salaad_qa.tex)；
实现见 [说明](../docs/moe_sinkhorn_alignment.md) /
[配置](../configs/ns97m_sinkhorn.yaml)。

<a id="history-through-20260923"></a>

## 2026-09-23 状态与后续讨论（历史梯度版本）

### 1. 最新决定与训练状态

用户要求：**其余参数不变，将每轮 P 的内部 SGD 从 8 步增加到 128 步，然后从头训练。**
已执行，运行时完整配置和提交参数均逐项核对；唯一训练参数差异为
`salaad.channel_alignment.sinkhorn.inner_steps: 8 → 128`。

| 实验 | Job / W&B | 已核实状态 |
|---|---|---|
| 原 Sinkhorn，K_P=8 | `17594019.0` / [861f0ac3](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/861f0ac3) | 9 月 22 日正常完成 2100 步，退出码 0 |
| Sinkhorn，K_P=128 | `17604698.0` / [8e289238](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/8e289238) | 9 月 23 日第 690 步结构更新失败；最后完成 689 步，退出码 1；尚未恢复 |

128 步版本提交为 `4a244be5aed18bbda9bb4404b1fd25c9c72c36d7`，
分支 `feature/moe-salaad`；本地、origin 和集群已在提交实验时同步。
修改包括正式配置的一个值、提交文件对应注释、现有配置测试的预期值。
相关 9 项测试通过；第 10 步首次 128 次更新也通过启动核验。这些检查未覆盖后来第 690 步的失败状态。

两组都在训练前初始化 P/X/L/S/U，每 10 个 optimizer steps 更新一次结构；
seed 42、完整预算 2100 步、DP4 / 4 H100 / 16 CPU / 64,000 MB / bid 100 不变。
P 学习率 1、温度 1、move_penalty_over_rho=1、初始化软化 0.1、
归一化上限 100、容差 1e-5 均未改动。任务 AdamW/LR、21 步 warmup、batch、
数据、rho=1e-5、L/S 控制器以及每 100 步保存且仅留最新 checkpoint 的策略均保持原样。

当前训练已停止。集群仅保留最新完整恢复点：
`data/moe/ns97m_sinkhorn/17604698_0/checkpoints/step_00000600`。
提高归一化上限仍是待验证、待决定的额外变更，本次没有修改这一参数或重启训练。

证据：[128 步提交与配置核对](../data/moe_analysis/sinkhorn_k128_launch_20260923/README.md)、
[停止原因和后续数值检查](../data/moe_analysis/sinkhorn_failure_20260923/README.md)。

### 2. K_P=8 的完整结果：P 有更新，但尚无明确任务收益

下表为完整 2100 步运行最后 100 个训练步的 NLL 均值，均值只沿时间计算。

| 方法 | 末 100 步训练 NLL |
|---|---:|
| Sinkhorn，K_P=8 | 4.241300 |
| 动态硬置换 | 4.243286 |
| 未对齐 consensus | 4.234395 |
| Independent L+S | 4.245390 |
| Vanilla | 4.182841 |

Sinkhorn 的最终逐专家投影重构误差范数范围为 0.116244–0.248045，
动态硬匹配为 1.100745–2.709680，未对齐 consensus 为 0.115726–0.240302。
抽取的 96 条专家投影曲线没有出现硬匹配实验那样的周期性大尖峰；这是抽样观察。
当前结果支持软更新更平稳，但没有显示优于未对齐 consensus 的训练 NLL。
这些不是验证集结果，也没有隔离“继续学习 P”相对“固定初始 P”的独立任务收益。

较早因 DNS 问题无法读 checkpoint 的限制已经解除。相关 JSON 中的旧访问失败说明
属于当时状态，后续矩阵分析以 9 月 23 日实际 checkpoint 检查为准。
见 [训练结果](../data/moe_analysis/sinkhorn_inspection_20260922/README.md)。

### 3. 用户明确纠正：要看实际 XP_init 和 XP_final，不能只看 P 的 Frobenius 变化

评估 P 是否发挥排列作用，重点应放在 **固定同一份 X 后实际映射出的通道方向、幅度和对应关系**。
按当前 P 的方向，gate/up 比较 `P_init.T @ X_final` 与 `P_final.T @ X_final`，
down 比较 `X_final @ P_init` 与 `X_final @ P_final`。
两边均使用最终 X，不把 X 自身的训练变化混进 P 的影响。

初始 P 来自同 seed、同模型初始化和同硬匹配代码的固定 P 对照 `17593583.0`，
再使用正式 Sinkhorn 的 `0.9 P_hard + 0.1/176` 和归一化规则生成初始软 P。
排除固定 P=I 的参考专家后，共 504 个专家、1512 个专家投影、266112 个投影通道向量；
保留逐专家和逐通道结果，不对专家求平均。

| 投影 | 通道夹角范围 | 通道向量范数变化范围 | cosine 最近邻对应改变 |
|---|---:|---:|---:|
| gate | 0.033°–6.155° | -19.85%–+4.20% | 0 / 88704 |
| up | 0.033°–1.063° | -23.61%–+4.77% | 0 / 88704 |
| down | 0.035°–1.600° | -23.65%–+4.78% | 0 / 88704 |

97.87% 的投影通道向量夹角不超过 1°；幅度变化超过 10% 的向量只有 419 个，占 0.157%。
将 gate/up/down 拼成联合通道向量后，88704 个通道的最近邻对应也全部保持原位。
最近邻是在实际映射向量之间按 cosine 搜索得到，不是 P 元素的 argmax。
这是起点/终点比较，不能排除训练中间曾短暂改变对应关系。

最大变化出现在 layer 7 / expert 36 / channel 37：gate/up/down 的幅度分别下降约
19.85% / 23.61% / 23.65%，方向变化分别为 6.155° / 1.063° / 1.600°。
所以不能把所有通道都概括为几乎不变；总体上对应与方向稳定，变化主要在幅度，少数通道变化明显。

补充范数结果：P 相对变化 1.0705%–5.2044%，固定最终 X 后映射矩阵相对变化
1.0018%–5.2483%；若把 X 自身训练变化也算入，总映射变化为 48.7245%–199.5344%。
后者用重建的初始化 X，CPU/CUDA 求解存在舍入差异；这些范数不是可相加的功能贡献百分比。
固定最终 X 后，对共同适配的 `B=E-L-S`，所有 1512 个非参考专家投影拟合误差都降低，
降幅 0.0134%–5.7709%；这不能单独证明任务收益。

结论限定为：**P 确实生效并改变了 XP，但本轮继续优化 P 的对齐与任务收益尚不明显。**
不能仅凭小 Frobenius 变化说 P 没有更新，也不能凭重构改善说它提高了语言建模质量。

证据：[直接 XP 通道比较与图表](../data/moe_analysis/xp_mapping_comparison_20260923/README.md)、
[端点分解及逐专家结果](../data/moe_analysis/sinkhorn_dynamics_20260923/README.md)。

### 4. 每次只走 8 步是否不足：冻结最终状态的离线检查

固定 K_P=8 运行的最终 X、`R=E-L-S+U`，从最终保存的 A/P 出发，
单次求解期间始终固定 P_old，比较更多 SGD 步、lr=8、去掉移动惩罚。
这是新构造的冻结子问题，不是历史某次更新的回放，也不是训练初始化的 P_init 对比。

在 lr=1、移动惩罚比为 1 时，各专家结果为：

| SGD 步数 | 相对 256 步已观察到下降量的完成比例 | 剩余梯度范数 / 起点 |
|---:|---:|---:|
| 8 | 4.79%–12.93% | 90.79%–98.36% |
| 32 | 18.22%–39.99% | 69.43%–93.71% |
| 64 | 34.20%–60.76% | 51.77%–88.07% |
| 128 | 60.97%–81.60% | 35.47%–78.33% |
| 256 | 100%（参照） | 23.36%–63.84% |

所有 504 个非参考专家在 8 步以后继续降低目标；256 步不是已知最优解，也未充分收敛。
8 步的目标相对下降只有 0.0000338%–0.0001527%，256 步为 0.0006010%–0.0014325%。
联合 XP 通道的最大夹角从 8 步的 0.1058° 增至 128 步的 0.6921°、256 步的 0.7470°。
联合通道指标不能当作 gate/up/down 分别计算的极值。

固定 8 步把 lr 从 1 增至 8，重构平方误差下降量为原先的 6.08–7.59 倍；
固定 128 步去掉移动惩罚，下降量为原先的 1.11–1.98 倍。
这些都是离线备选诊断，**正式新训练没有改变学习率或移动惩罚**。

据此建议先用 K_P=128、lr=1 做对照，以检验更新偏保守是否限制对齐。
这不是确认 128 为最优值，也不是确认增加迭代就一定改善 XP 或 NLL。
正式 8 步函数与诊断实现的 A/P 逐元素一致；分析作业 `17604262.0` 正常完成。
见 [内部优化诊断](../data/moe_analysis/sinkhorn_inner_steps_20260923/README.md)。

### 5. 128 步的 compute overhead：函数实测与整轮外推

在一张 H100 上，对未修改的正式函数逐层计时，预热后每种 K 重复 3 次。
按 DP4 中各卡负责两层、每 10 步更新一次结构来外推：

| K_P | 单层 P 更新耗时 | 每卡两层合计 |
|---:|---:|---:|
| 8 | 0.109–0.126 秒 | 0.220–0.251 秒 |
| 64 | 0.838–0.955 秒 | 1.703–1.899 秒 |
| 128 | 1.717–1.950 秒 | 3.460–3.881 秒 |

128 相对 8 每次结构更新多 3.240–3.630 秒，摊到训练步多 0.324–0.363 秒。
相对旧运行末 100 步平均 7.708 秒，预计增幅 4.20%–4.71%。
旧运行整轮约 5 小时 20 分钟，按 210 次结构更新估计多 11.34–12.71 分钟，
整轮增幅约 3.54%–3.97%，额外 0.76–0.85 H100 GPU·小时。

这是固定旧运行最终状态的函数基准，加上既有训练日志的外推，**不是完整 128 步训练实测**。
各卡范围不是统计置信区间。新训练状态可能需要更多归一化轮数；后来确实触及了 100 轮上限。
因此不能把该估计视为新运行或提高归一化上限后的保证耗时。
基准作业 `17604298.0` 正常结束；见 [开销测量](../data/moe_analysis/sinkhorn_compute_overhead_20260923/README.md)。

### 6. 为什么停止：SGD 次数与归一化次数是两层循环

作业 `17604698.0` 于 **2026-09-23 14:22:14 Europe/Zurich** 报错退出，
最后记录的 step=689、NLL=4.76658249；在尝试 step=690 的结构更新时出现：

```text
SALAAD sweep failed (no state committed): rank 2:
Sinkhorn did not reach the configured marginal tolerance
```

这是训练进程的数值收敛检查失败，Condor 日志没有显示抢占或 OOM。
rank 2 负责 layer 2、6；现有错误没有记录具体层、专家、内部 SGD 序号或最终边际误差。
不能仅凭该报错判断 loss 发散，或者确定需要增加到多少归一化轮。

P 的可行条件为非负、行和及列和均为 1；每个专家 P 是 176×176。
固定当前 A，从 `Z=A/temperature` 开始，每一轮执行：

```text
Z ← Z - logsumexp(Z, 按行)
Z ← Z - logsumexp(Z, 按列)
P = exp(Z)
error = max(所有专家的最大行和误差、最大列和误差)
```

行归一化后再做列归一化，一般会再次改变行和，因此需要交替迭代。
代码每 5 轮或达到上限时检查一次；error≤1e-5 提前返回，100 轮耗尽仍不满足则报错。
严格正矩阵的理论收敛不等于保证 100 轮内达到给定精度。
归一化只处理行列约束，不保证接近硬排列或改善 XP 的对齐。

```text
每 10 个 optimizer steps，固定本轮 X、R 和 P_old：
    重复 128 次 SGD：
        P = Sinkhorn(A)             # 每次最多 100 轮行/列归一化
        g_A = 对联合重构 + 移动惩罚反传，穿过 Sinkhorn
        A ← A - 1.0 * g_A
    P_final = Sinkhorn(A)           # 最后再计算一次
    更新 X → L → S → U → 阈值与 Q
```

因此每层每次结构更新有 129 次 Sinkhorn 调用；100 不是 128 步 SGD 的总次数限制。
增加归一化上限会给难收敛的调用更多计算预算，已达标的调用仍提前停止；
数学目标可保持不变，但数值近似、梯度及耗时仍可能改变。

### 7. 第 600 步 checkpoint 的后续只读数值检查

诊断作业 `17605508.0` 已正常完成，计算约 8.3 秒。
第 600 步的完整标记、training.pt 和四个 rank 文件均存在；模型与 checkpoint 留在集群。
读取 rank 2 的层 2、6，保存的 logits 均有限，用原设置重建 P 与保存值逐元素一致。

在固定第 600 步 X 和最终保存 L/S/U 构造的 R 上，各做一次 128 步 P 更新：

| 层 | Sinkhorn 调用数 | 单次调用最多归一化轮数 | 各调用返回时的最大误差 |
|---:|---:|---:|---:|
| 2 | 129 | 85 | 9.95398e-6 |
| 6 | 129 | 90 | 8.04663e-6 |

两层在原 100 轮上限内都通过，因此脚本预留的上限 1000 分支**没有执行**。
这提示当时上限余量不大，但不是第 690 步失败状态的回放，不能说已经验证 1000 可以修复训练。
完整日志见 [数值检查](../data/moe_analysis/sinkhorn_failure_20260923/README.md)。

### 8. 当前算法各步骤更新的变量

令 M_i 为映射后的 shared：gate/up 为 P_i^T X，down 为 XP_i。

| 时机 / 顺序 | 更新变量 | 操作 |
|---|---|---|
| 训练前一次 | P、X、L/S/U | 硬匹配后软化 P，最小二乘拟合 X；L=U=0，S=E-M；参考专家 P=I |
| 每个训练步 | 完整 E 及其他模型参数 | 完整权重前向；任务梯度做 DP 平均后，加 rho(E-Q)，裁剪，再 AdamW |
| 每 10 步，第 1 项 | A、P | 固定 X 与 R=E-L_old-S_old+U_old，128 步 SGD 优化 F/rho；P_old 整轮固定 |
| 第 2 项 | 共享 X | 固定新 P，对 R 做最小二乘求解；各层、各投影独立 |
| 第 3 项 | 专家 L | 对 E-M-S_old+U_old 做 streaming SVT，使用旧低秩阈值 |
| 第 4 项 | 专家 S | 对 E-M-L_new+U_old 做逐元素软阈值，使用旧稀疏阈值 |
| 第 5 项 | 专家 U | U_new=U_old+E-M-L_new-S_new |
| 第 6 项 | 阈值、Q | 按 rank/density 偏差调整阈值，Q=M+L+S-U，广播给后续模型训练 |

只有 P 子问题重复 128 次，其余结构变量每轮各更新一次。
同层 gate/up/down 共用一份 P，各自有共享 X；L/S/U 各专家独立，始终保留原通道顺序。
P 子问题没有额外乘上小 rho 缩小 SGD 梯度，模型结构梯度的 rho 则仍为 1e-5。

通俗推导、两通道归一化示例、嵌套循环和图表已同步到
[最新算法 PDF](moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.pdf) /
[LaTeX 源文件](moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.tex)。

### 9. 未完成事项与后续约束

- 停止后的训练尚未恢复，128 步是否改善实际 XP 对应或 NLL 尚无完整结果。
- 修复方向是保留 K_P=128、学习率及 1e-5 容差，提高数值归一化预算；
  具体上限尚未验证充分，也尚未获得改变“其余参数不变”约束的后续指令。
- 恢复点为 step 600。更改配置会触发当前严格的 checkpoint 配置哈希校验，
  需要明确处理兼容的数值求解设置变更，保留模型、Adam、辅助状态、数据游标和 RNG；不能静默绕过校验。
- 当前只保留最新 checkpoint，无法重建此前完整逐步 P/XP 轨迹。今后若增加监测，
  应保留逐专家的实际 XP 方向、幅度、对应和子问题下降数据；当前没有增加 W&B 指标。
- 仍遵守：不对专家求平均、不在训练中加 validation、不额外剪枝/截 rank，
  不因更新讨论而提交、取消、修改或重启训练。

本次只更新本地讨论与算法 PDF，保留旧快照；没有把讨论笔记 commit/push 或同步到集群。

---

<a id="history-through-20260922"></a>

## 2026-09-22 历史快照：提交与启动核验

> 以下保留当时的决定和启动核验记录。K_P=8 已完成，K_P=128 的后续决定和失败状态见上方最新章节。


### 1. 最新决定：从训练开始使用软 P，保留 L/S

用户当前选择的是 `E_i ≈ X P_i + L_i + S_i`：E 是完整专家权重，X 是 shared，
L/S 仍是各专家自己的低秩/稀疏部分。P 放宽为双随机矩阵，用确定性的 log-Sinkhorn
从分数表 A 生成，再用梯度逐渐调整 A。这里的新 shared X 对应旧文档的 H，
新 E 对应旧文档的完整权重 X；后面的旧结果继续保留 H/L/S 记号。

**最终时机已明确修改为：训练前初始化，不保留前 100 步 vanilla。**
在 optimizer step **10、20、…、2100**，P 与 shared/L/S/U 同步更新，
每轮 P 做 **K_P=8 个内部 SGD 步**。计数单位始终是 optimizer step，不是 epoch。
总预算仍为 2100 步，学习率仍保留原来的 **21 步 warmup**。
此前 soft 方案中的“vanilla 100 步、第 110 步开始更新”已被此次决定替代；
三组已经提交的硬置换实验保留各自原定时机。

讨论中还提出过 `X_i = X_hat + H_i`、H_i 为 dense 的简化方案。
这不是当前采用的实验：用户随后明确三组时机对照仍用原来的 H+L+S，
之后提出的 Sinkhorn 方案也继续保留 expert-specific L/S。没有提交 dense H_i 方案。

研究重点继续是 **FFN 通道的对应关系和共享对训练是否有用**，不是压缩。
“专家通道未对齐可能导致直接平均抵消”仍是假设；已有实验没有证明它是 shared
效果不佳的唯一原因，也没有证明 shared 范数越大越有用。

### 2. 已完成的硬置换实验：任务表现与分解行为

首次动态硬置换作业 `17590593.0` 已于 9 月 22 日 **06:12:03 Europe/Zurich**
正常结束，退出码 0，完整 2100 步；W&B 为
[89b39a7d](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/89b39a7d)。
该实验从训练前初始化，每 10 步更新结构、每 100 步重新匹配 P。

| 方法 | 最后一步训练 NLL | 最后 100 步 NLL 均值 |
|---|---:|---:|
| 动态硬置换 consensus | 4.229962 | 4.243286 |
| 原未对齐 consensus，rho=1e-5 | 4.220859 | 4.234395 |
| Independent L+S | 4.232547 | 4.245390 |
| Vanilla | 4.170180 | 4.182841 |

这些是同预算的完整 dense 专家前向训练 NLL，均值只沿训练步计算。
动态硬置换比原未对齐版的末 100 步 NLL 高 0.008891；当前结果没有显示任务收益。
四组日志均已核验覆盖 1–2100 步，原先 independent/vanilla “最终结果待核实”的
状态已由此次分析补充。训练 NLL 不能代替验证集表现，也不是重构权重的任务结果。

硬置换重匹配处出现周期性重构误差尖峰：第 2090 步的逐专家 projection
绝对误差区间为 0.160075–0.439237，第 2100 步匹配后变为 1.100745–2.709680。
末次 checkpoint 正好处在这个尖峰；对齐版的相对重构误差区间是
57.36%–163.19%，原未对齐版是 3.42%–19.03%。
这支持进一步检查“改变对应关系是否扰动已形成的分解”，不能仅凭日志把原因
全部归给 P，或把末次更低的秩/密度解释为稳定收益。

最终 step-2100 checkpoint 的 H/S/L 范数配对结果如下。
H 每层每投影一份，共 24 份；L/S 各有 1536 个 expert projection，
没有对专家取均值或中位数。

| 分量 | 对齐后变大 | 对齐后变小 | 新/旧 Frobenius 范数比范围 |
|---|---:|---:|---:|
| H | 24 | 0 | 2.951–7.593 |
| S | 1520 | 16 | 0.827–1.982 |
| L | 87 | 1449 | 0.561–1.115 |

H 全部变大，S 大多变大，L 大多变小，但范数不能相加，
也不能据此断言知识从 L/S 转移到了 H。以上仍是匹配后的终点快照。
完整证据见 [训练结果分析](../data/moe_analysis/aligned_20260922/README.md)、
[24 组范数表与解释](../data/moe_analysis/aligned_20260922/norm_comparison.md)、
[逐专家范数 CSV](../data/moe_analysis/aligned_20260922/norm_expert_comparison.csv)。

### 3. 已提交的三组硬置换时机对照

用户确认只调整 P 的初始化/重新匹配时机，仍使用 H+L+S；
前 100 步 vanilla 包含在 2100 步预算内，不额外延长训练。

| 实验 / 配置 | P 的生成与更新时间 | Condor job | W&B |
|---|---|---|---|
| `ns97m_aligned_fixed` | 第 0 步生成，之后固定 | `17593583.0` | [d8c91e59](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/d8c91e59) |
| `ns97m_aligned_warm100_fixed` | vanilla 100 步后生成，之后固定 | `17593584.0` | [85cbab9c](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/85cbab9c) |
| `ns97m_aligned_warm100_every200` | 第 100 步生成，第 300、500、…、2100 步重新匹配 | `17593586.0` | [7c4edf5d](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/7c4edf5d) |

前两组固定的是 P，shared/L/S/U 仍继续更新。
第一组在第 10、20、…步更新结构；后两组在第 100 步初始化分解，
第 110、120、…步更新结构。用户所说“建立稳定关系”是这项对照的动机，
并不代表已经确认训练 100 步就形成稳定通道。

三组使用代码 `2053fae`，于 9 月 22 日约 10:22 Europe/Zurich 提交，
均为 4 H100 / DP4 / bid 100。各自提交后的核验记录为 Running；
此处没有补查或归档三组的最终结果。
见 [提交记录](../data/moe_analysis/alignment_schedule_20260922/submissions.json)。

### 4. Sinkhorn 的实际算法，用通俗话说

P 不再只允许“交换两个通道”，也允许“一个通道借用几个 shared 通道的加权组合”。
P 的元素非负，每行、每列的和都是 1。一个 SwiGLU 通道同时包含 gate 的一行、
up 的一行和 down 的一列，所以同层三个投影必须共用同一个 P。

按 `P[shared_channel, native_channel]` 的方向：

```text
shared_gate_i = P_i^T X_gate
shared_up_i   = P_i^T X_up
shared_down_i = X_down P_i
E_i ≈ shared_i + L_i + S_i
Q_i = shared_i + L_i + S_i - U_i
```

硬置换可以整体重编号并保持原 FFN 函数；软 P 混合通道，一般没有这种函数不变性。
本实现的前向始终使用完整 E，P 只参与构造分解参考和结构梯度。
更新 P 不搬动 E、L/S/U、Adam 或 streaming 基的通道位置。

完整执行顺序为：

1. **训练前初始化。** 用现有三投影联合硬匹配产生起点，对非参考专家取
   `P=0.9 P_hard + 0.1/m`，`A=temperature*log(P)`；m 是通道数。
   专家 0 固定 P=I、A=0。固定 P 求初始 shared，设 L=0、S=E-shared_i、U=0，
   沿用原 streaming 基初始化。数学上 Q=E，初始结构梯度为零，模型权重不变。
2. **每个 optimizer step。** 先累积任务梯度并做 DP 平均，对全部专家投影
   加一次 `rho*(E-Q)`，再执行原有梯度裁剪和 AdamW。
   分解已在训练前建立；首步梯度为零来自 Q=E，不是保留 vanilla 阶段。
3. **每 10 步更新 P。** 固定本轮旧 shared 和 `R=E-L_old-S_old+U_old`，
   从已有 A 继续做 8 步 SGD。优化三投影的真实联合重构平方误差，
   再加 `||P-P_old||²` 的变化惩罚。8 步内的 P_old 始终是本轮开始前的同一份 P。
   Sinkhorn 负责生成满足行列约束的 P；梯度由实际重构目标和变化惩罚决定。
4. **固定新 P，更新 shared。** 软 P 一般不满足 `P P^T=I`，不能继续简单求平均。
   计算 `M=sum_i(P_i P_i^T)`，解下面三个线性方程；实现使用 Cholesky，
   不显式求逆，不添加 ridge。
5. **更新专家自己的部分。** 用新 shared 顺序更新 L、S、U、控制器和 Q，
   保留原 SALAAD++ streaming SVD 和 L/S 阈值规则。所有层候选状态经 DP
   协商成功后才提交。末步需要补结构更新时，P 也同步更新。

```text
M X_gate_new = sum_i(P_i R_gate_i)
M X_up_new   = sum_i(P_i R_up_i)
X_down_new M = sum_i(R_down_i P_i^T)
```

专家 0 的 P=I 使 M 至少含一份 I。对软 P 而言，这同时限制参考专家的 shared
不能混合通道，是实际建模约束，不只是改编号。

P 子问题整体除以 rho 后优化 `F/rho`，保留重构与变化惩罚的相对权重，
避免小 rho 直接压小 P 的梯度；模型梯度中的 rho 不变。
首版配置如下，新增值尚未经过正式训练质量调优：

| 设置 | 值 |
|---|---:|
| P 更新间隔 | 10 optimizer steps，与 L/S 一致 |
| K_P，内部 SGD 步数 | 8 |
| Sinkhorn temperature | 1.0，固定 |
| P 的 SGD learning rate | 1.0，作用于 F/rho |
| lambda_move / rho | 1.0 |
| 初始化软化量 | 0.1 |
| 每次 Sinkhorn 行列归一化上限 | 100 轮 |
| 最大行列和误差容限 | 1e-5 |

K_P=8 指优化 A 的梯度步数；100 指每次生成 P 时的归一化上限，两者不同。
P 的 SGD 学习率也与模型 AdamW 的 3e-4 分开。
这里没有加入 Gumbel 噪声、温度退火或硬化导出。
仅对距离矩阵做一次 Sinkhorn、再用旧的 shared 平均公式，不等于上述算法。

### 5. 最新正式训练：配置、版本和提交证据

代码提交 `47e4e09d852b5896e942259ac08340a1a8e2e68e` 已 commit/push，
集群 `feature/moe-salaad` 已 fast-forward 到同一版本。
正式入口是 [ns97m_sinkhorn.yaml](../configs/ns97m_sinkhorn.yaml)，
提交文件是 [moe_ns97m_sinkhorn.sub](../sub/moe_ns97m_sinkhorn.sub)。

提交前与上一组 `17593586.0` 的实际 `config.resolved.json` 对比：
仅实验名、分解初始化方式/时机和 P 的方法/设置改变。
模型、数据、seed、任务优化器、L/S 设置、训练预算和资源保持一致。

| 项目 | 当前设置 |
|---|---|
| 模型 | 8 层，hidden 256，expert FFN 176，64 experts / top-8，97,194,240 参数 |
| 数据 | 同一 DCLM manifest 和数据指纹；序列长 2048 |
| Batch / DP | global batch 1024，每卡 micro batch 16，累积 16 次；DP4 |
| 预算 / seed | 2100 optimizer steps，4,404,019,200 预测 tokens；seed 42 |
| AdamW / LR | 3e-4 → 3e-5，21 步 warmup 后 cosine；betas 0.9/0.95，epsilon 1e-8，weight decay 0 |
| 任务计算 | BF16；FP32 参数和辅助状态；裁剪阈值 1.0；balance/z-loss 权重 0.1/0.001 |
| L/S | rho=1e-5，gain_alpha/gain_beta=0.02/0.002，target rank/density=0.15/0.10，gamma=0.999 |
| 结构 | 训练前初始化，每 10 步一次 sweep；svd_chunk_size=8 |
| 保存与资源 | 每 100 步保存，只保留最新 checkpoint；4 H100 80GB、16 CPU、64,000 MB、bid 100 |

正式 manifest：
`/lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916/tokens_ns97m/manifest.json`。

**Sinkhorn job `17594019.0`** 于 **2026-09-22 11:26:22 Europe/Zurich**
提交（09:26:22 UTC），11:26:24 开始执行。
11:26:31 的队列核验为 Running，节点 `g206`；随后已核验 W&B 初始化、
DP4 runtime metadata 和保存的配置：初始化步 0、结构/P 间隔 10、K_P=8。
这些是提交后的启动记录，不代表训练已经完成或已有质量收益。

- W&B：[861f0ac3](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/861f0ac3)。
- 输出：`data/moe/ns97m_sinkhorn/17594019_0/`（相对集群仓库）。
- 日志：`jobs/moe_ns97m_sinkhorn_17594019_0.{log,out,err}`。
- [本地提交记录](../data/moe_analysis/sinkhorn_20260922/submission.json)。

### 6. 实现验证、保留约定与下一步

Sinkhorn 实现后全套 MoE 测试 **88 项通过**，另完成实际 DP2/Gloo 训练与
模型/Adam/辅助状态/数据游标/RNG 的逐位恢复、CUDA BF16 CLI 暂停恢复，
以及 64 专家、176 通道、256 hidden 的单层 CUDA 数值检查。
取消 vanilla 前缀后，相关 **9 项测试再次通过**；20 步小模型检查确认：
训练前已有分解，P/L/S 只在第 10、20 步同步更新，结构约束在第 10 步之前已参与训练。
这里的 smoke 验证不等于正式 97M 任务效果验证。

Checkpoint 保存 A 和实际 P，并检查二者一致、行列和、参考专家和三投影共享关系。
软 P 导出为 v4，保存完整 FP32 P，重构 `shared_i+L_i+S_i`，不含 U；
继续支持旧 v2/v3，不把软 P 强制变成硬排列。
详见 [实现说明](../docs/moe_sinkhorn_alignment.md)、
[验证记录](../docs/moe_sinkhorn_validation.json)、
[通俗算法 PDF](moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.pdf)、
[LaTeX](moe_sinkhorn_alignment_20260922/moe_sinkhorn_alignment.tex)。
验证 JSON 的 `formal_sinkhorn_cluster_job_submitted_at_validation=false`
记录验证时尚未提交，实际提交状态以上一节的 receipt 为准。

继续遵守原有约定：统一 MoE 训练入口与英文代码注释，普通 DP4，
不在训练中加 validation，不擅自加 W&B 指标或对专家求平均，
不额外剪枝/截 rank/改精度，checkpoint 只保留最新一份。
仅在用户要求时更新讨论；不因更新文档而提交、取消或修改训练。

后续先读取这四个已提交实验的结果，比较相同训练预算下的 NLL、结构更新处的波动
和保存状态中的分解行为。P 的行列误差、变化量及混合程度可离线检查。
与第三组硬置换相比，Sinkhorn 同时改变了 P 表示、初始化时机和更新频率，
不能把后续全部差异只归因于 soft P。若要分离影响，需要另做时机相同的硬 P 对照；
这项对照及固定 L/S 的 H/零 H/随机 H 功能消融尚未提交。

本目录与 `data/moe_analysis/` 由 gitignore 忽略；本次更新是本地讨论资料，
不代表已同步到集群。下面旧快照中的“对齐尚未运行”“未提交固定 P 对照”
和“本地没有可用 CUDA”等状态，已由本节的新记录替代。

<a id="history-through-20260921"></a>

## 截至 2026-09-21 训练提交的历史快照

以下保留上一版原文；其中“当前”和“后续待核实”指当时状态。

### A. 用户确认的工作约定

1. 仅维护当前独立 MoE + SALAAD 训练框架，不再要求兼容旧 dense LLM/Vision。
   原生训练都进入 `scripts/train_salad.py:main()`；保留 `__main__` 中逐项参数赋值和
   显式调用。代码 comments/docstrings 使用英文。
2. 本地 IDE 直接 F5、无参数启动真实 DCLM smoke；默认配置 `smoke_dclm_bf16`，
   manifest 为 `data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json`。
   单进程也创建真实 dist 进程组，CUDA 用 NCCL、CPU 用 Gloo。
3. 正式 97M 实验使用 **4 张 H100、普通 DP4、bid 100**。用户已经提供 bid，
   旧记录中的“等待 bid”不再成立。每张卡保留完整模型，不使用 expert parallel。
4. 同训练器、同数据、同预算对比 vanilla、consensus SALAAD 和 independent L+S。
   参考论文不构成必须复现其框架或绝对分数的要求。
5. **只有用户要求更新讨论时才修改本记录**，不随每次代码修改自动追加。
6. 不在训练期间执行 raw/reconstructed validation；删除相关日志和
   `reconstruction_nll_gap` 等附加指标。配置中的离线 evaluation 任务清单不意味着
   训练循环会执行评估，也不表示已经完成这些任务。
7. SALAAD 在 W&B 上总共只有两个区：结构、超参数；不同层/projection/expert
   写入指标名称，不按层创建更多区。保留逐专家数据，不擅自取专家平均或增加指标。
8. 导出遵循“训练得到什么就导出什么”：不额外截 rank、筛 S 的 top-k 或强制降低精度。
   现有 exporter 保留完整 dense L、shared 和所有 S 非零值（CSR）。
9. 已移除 `target_ramp_end_step`、`freeze_coefficients_step`；使用固定控制器目标，
   不增加目标 ramp 或系数冻结。控制器阈值允许负向超调，阈值算子内部截零仍保留。
10. 结构更新采用 SALAAD++ streaming SVD，保留单一更新算法；初始化仅做一次普通
    SVD 准备迭代基，不添加多 driver/fallback。`svd_chunk_size` 只控制 expert 批大小。
11. 每次保存成功后删除上一份完整 checkpoint，只保留最新一份；不继续累积多份。
    用户此前也已要求清理旧一轮训练输出及相应 W&B 内容，旧 run 不应混入下面的新实验。
12. 本轮研究重点是**通道置换、FFN 的表达以及共享是否有效**，不是重新讨论压缩。
    L 和 S 始终是各专家独立的一份；“通道”指 FFN 隐藏神经元，不是 attention head。
13. 用户已认可通俗文档中的完整训练算法，并要求据此修改代码、提交训练。
    新实验保持此前实际训练的所有原有超参数不变；新增通道匹配设置单独记录。
14. 说明文档使用通俗语言，给出完整执行顺序和伪代码；LaTeX/PDF 以内容为主，不做花哨排版。

### B. 当前配置、数据与代码版本

本次归档核对的 HEAD 为 `3867b42d62ff3d6e4f90fca65da8df9135615de1`；
本地跟踪 `origin/feature/moe-salaad`。提交对齐训练前已完成 commit/push，集群在
同一分支 fast-forward 拉取，并核验两端都是该提交；本次归档未重新远程查询。

| 用途 | 配置 | HTCondor 文件 |
|---|---|---|
| Consensus：`X = H + L + S` | [ns97m.yaml](../configs/ns97m.yaml) | [moe_ns97m_salaad.sub](../sub/moe_ns97m_salaad.sub) |
| Vanilla：关闭 SALAAD | [ns97m_vanilla.yaml](../configs/ns97m_vanilla.yaml) | [moe_ns97m_vanilla.sub](../sub/moe_ns97m_vanilla.sub) |
| Independent：`X = L + S` | [ns97m_independent_ls.yaml](../configs/ns97m_independent_ls.yaml) | [moe_ns97m_independent_ls.sub](../sub/moe_ns97m_independent_ls.sub) |
| Aligned consensus：`X_i ≈ Hbar_i + L_i + S_i` | [ns97m_aligned.yaml](../configs/ns97m_aligned.yaml) | [moe_ns97m_aligned.sub](../sub/moe_ns97m_aligned.sub) |

当前正式训练设置（smoke 有自己的覆盖值）：

| 项目 | 设置 |
|---|---|
| 模型 | 8 层，每层 64 experts，每 token 选 8 个；没有 always-on shared expert |
| Hidden / expert FFN | 256 / 176，SwiGLU；每个 projection 有 45,056 个参数 |
| 每个 expert | gate/up/down 共 `3 × 45,056 = 135,168` 个参数 |
| 全模型 | 97,194,240 个参数；expert 部分 69,206,016，其余 27,988,224 |
| DP / 资源 | 4 H100 80GB，16 CPU，64,000 MB host memory；bid 100 |
| 序列 / 全局 batch | 2048 / 1024；每卡 micro batch 16，累积 16 次 |
| 预算 | 2100 optimizer steps，4,404,019,200 个预测 tokens |
| AdamW | LR `3e-4`，min LR `3e-5`；beta1 `0.9`，beta2 `0.95`，weight decay `0` |
| 随机种子 / Adam epsilon | `42` / `1e-8` |
| LR 调度 | 前 21 步线性 warmup，剩余完整预算使用 cosine decay |
| 路由辅助 loss 权重 | balance `0.1`，z-loss `0.001` |
| 精度 / 裁剪 | BF16 autocast，FP32 参数；全局梯度范数上限 1.0 |
| 辅助状态 / 结构频率 | 第 0 步训练前初始化；每 10 个 optimizer steps 更新一次，每次 1 sweep |
| 当前 rho | `1e-5`；之前另完成 `1e-6` 的对照 |
| 控制器增益 | `gain_alpha=0.02`，`gain_beta=0.002`；初始 alpha/beta 为 0 |
| 固定目标 | effective rank ratio `0.15`、density `0.1`；线性奇异值质量覆盖率 `gamma=0.999` |
| Streaming SVD | 每次一次 QR 正交迭代，`svd_chunk_size=8`，迭代基随 checkpoint 保存 |
| Checkpoint | 每 100 步及训练结束保存，`keep_latest_checkpoints=1` |

这里用户所说“beta 改为 0.002”已落实为控制器 `gain_beta`，不是把每个 expert 的
动态 beta 固定成 0.002。实现保存 `tau_l=alpha/rho`、`tau_s=beta/rho`，每次结构更新：

```text
tau_l += 0.02  * (effective_rank_ratio - 0.15)
tau_s += 0.002 * (density - 0.1)
logged alpha = rho * tau_l
logged beta  = rho * tau_s
```

云端 DCLM 分词已经完成，正式任务共享以下 manifest：

```text
/lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916/tokens_ns97m/manifest.json
```

训练 token 文件为 uint32，共 4,404,019,201 个 token、17,616,076,804 字节。
多出的一个 token 用于构造右移标签，预测 token 预算仍为 4,404,019,200。
原始 48 个压缩切片共 10.50 GB，不应与分词后 17.62 GB 文件混为一谈。
本地 smoke 继续使用自己的小切片 manifest。

关键提交：`15c47b1` 删除训练中验证、保留逐专家指标并修改 beta gain；
`a0b451a` 改为 cosine LR、W&B 分类和仅保留最新 checkpoint；
`0c4f7d2` 记录梯度 cosine 并设 rho=`1e-6`；`6552b49` 将 rho 改为 `1e-5`；
`6e29a66` 增加匹配的 independent L+S 配置和任务；
`3867b42` 实现联合通道匹配、原顺序下的共享重构、恢复/导出、验证及匹配原训练设置的提交文件。

### C. 新一轮实验身份与已核验状态

以下表格记录各自最近已核验的状态；不是本次归档时对所有任务的实时查询。

| 实验 | Condor job | W&B run | 已核验情况 |
|---|---|---|---|
| Vanilla | `17573606.0` | 本次归档未核对 ID | 已提交新一轮任务；最终步数未在本次记录中核实 |
| Consensus，rho=`1e-6` | `17573648.0` | [49a58b3a](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/49a58b3a) | 已完成 2100 步；9 月 19 日读取最终 checkpoint 分析 |
| Consensus，rho=`1e-5` | `17574041.0` | [a811c403](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/a811c403) | 已完成 2100 步；9 月 19 日读取最终 checkpoint 分析 |
| Independent L+S，rho=`1e-5` | `17579035.0` | [3bb230b9](https://wandb.ai/hao-ma-eth-z-rich/SALAAD-MoE/runs/3bb230b9) | 9 月 19 日已启动；此前最近核验为运行中、checkpoint 到 1200 步；未归档最终结果 |
| Aligned consensus，rho=`1e-5` | `17590593.0` | 提交后尚未取得 ID | 9 月 21 日 23:49:25 提交；随后核验为 Idle（排队），尚无正式训练结果 |

云端仓库根目录为 `/lustre/home/hma2/projects/awd-LoRA`，各输出相对路径：

```text
data/moe/ns97m_vanilla/17573606_0/
data/moe/ns97m_salaad/17573648_0/checkpoints/step_00002100/
data/moe/ns97m_salaad/17574041_0/checkpoints/step_00002100/
data/moe/ns97m_independent_ls/17579035_0/
data/moe/ns97m_aligned/17590593_0/
```

用户要求“同样参数，但不加 consensus”。已提交的 independent L+S 使用
`shared_mode: none`，H 初始化为零并始终保持为零；L/S/U、rho、控制器、LR、数据、
seed、训练预算及硬件设置沿用 rho=`1e-5` 的 consensus 实验。
提交前已与参考 run 的实际 runtime config 比较，除实验名称及 shared_mode 外相同。
这项实验仍启用 SALAAD 的 L+S 约束，与 vanilla 的完全关闭 SALAAD 有区别。

新 aligned consensus 以已完成的 `17574041.0` 为基线。提交前比较的是该 run 的
`config.resolved.json` 和 `run_metadata.json`，不仅是当前 YAML；正式数据 manifest
由 CLI 传入，因此从 metadata 核对实际路径。模型、数据、seed、2100 步预算、AdamW、
LR、rho、结构间隔、控制器、精度、checkpoint、DP4 和 W&B 原有设置均保持一致。

配置差异仅为实验名称、对齐初始化、辅助 owner 改为按层分配、结构顺序增加排列更新，
以及新增的 `channel_alignment`：每 100 步匹配，初始化最多 10 轮，改进容差 `1e-6`，
参考专家 0。原结构更新仍每 10 步一次。新 `.sub` 保留原资源、环境、节点/GPU 排除规则
和原有启动参数，仅更换配置、输出和日志路径。

提交命令为 `condor_submit_bid 100 sub/moe_ns97m_aligned.sub`，只提交了一次。
时间为 2026-09-21 23:49:25 Europe/Zurich（21:49:25 UTC）；作业请求 4 H100 80GB、
16 CPU、64,000 MB host memory。集群已核验相同 DCLM manifest 存在，SciPy 指派求解可用。
最后一次检查为 `JobStatus=1`（Idle），`.log` 已记录提交事件，`.out/.err` 尚未生成；
这只代表提交后的核验时点，不能据此宣称已经开始或完成训练。

云端提交记录及日志相对路径：

```text
data/moe/ns97m_aligned/17590593_0/submission.json
jobs/moe_ns97m_aligned_17590593_0.log
jobs/moe_ns97m_aligned_17590593_0.out
jobs/moe_ns97m_aligned_17590593_0.err
```

`submission.json` 已记录 job、bid、代码提交、基线、数据路径、原超参数一致性和新增对齐设置。

### D. W&B 分类、指标定义与梯度讨论

W&B 项目为 `hao-ma-eth-z-rich/SALAAD-MoE`，横轴为 `optimizer_step`。
当前分类由 [tracking.py](../salaad_moe/tracking.py) 实现：

| 区域 | 内容 |
|---|---|
| `train` | `lm_nll`、`load_balancing_loss`、`router_z_loss`、`learning_rate` |
| `train_stats` | task/constraint 梯度范数及比值、裁剪前总梯度范数、梯度 cosine、step 时间、已消费序列与预测 token 数 |
| `router` | 每层 `assignment_cv`、`mean_token_entropy` |
| `salaad_structure` | 每层、每 projection、每 expert 的 `diff`、`density`、`effective_rank_ratio` |
| `salaad_hyperparameters` | 每 expert 的 `alpha`、`beta`；全局 `rho` 只记录一次 |

指标实例为 `salaad_structure/layer_0_gate_expert_0_diff` 和
`salaad_hyperparameters/layer_0_gate_expert_0_alpha`；层号等标识放在唯一的 `/` 后面，
保证 SALAAD 总共只有两个区。没有专家均值，也不记录此前额外加入的
`weight_squared`、`linear_mass_rank_ratio` 等项。

`train/lm_nll` 是预测下一个 token 的平均交叉熵，是当前训练日志中直接反映语言建模
表现的 loss，越低表示在这些训练 token 上预测越好；不能直接当作未见数据上的表现。
balance loss 与 z-loss 用于路由训练，不代表语言建模质量。
`assignment_cv` 描述专家接收 token 次数的相对不均衡程度；低值表示分配更均匀。
`mean_token_entropy` 是每个 token 在全部专家上的路由概率熵再平均；越高越分散、
越低越集中，不能简单判断越低越好。`effective_experts_from_entropy` 是此前熵指标的
指数尺度变换，按用户要求删除，避免重复展示。

结构约束通过梯度叠加实施，不另造一个需要 backward 的 loss：

```text
g_task = gradient of (LM loss + weighted balance loss + weighted z-loss)
Q = Hbar_i + L_i + S_i - U_i
g_structure = rho * (X - Q)
g_combined = g_task + g_structure
```

这里 Hbar_i 是按排列表对应回专家原顺序的 H；原不对齐 consensus 中 Hbar_i=H，
independent L+S 中 Hbar_i=0。任务梯度完成累积与 DP 平均后，只加一次结构梯度，
覆盖全部 expert 权重，包括当步未被路由使用的专家；不额外除以专家数或重复加到每个 microbatch。

在 DP 归约后、全局裁剪及 AdamW 之前，记录
`train_stats/constraint_task_gradient_cosine`：

```text
cosine = <g_task, g_structure> / (||g_task|| * ||g_structure||)
```

统计覆盖全模型，非 expert 参数的结构梯度为零；task 范数也包含这些参数。
cosine 正值表示两个原始梯度的方向有一致成分，负值表示有冲突，接近零表示近似正交；
任一范数为零时不报告 cosine。它衡量当步梯度几何关系，不保证经过 AdamW 预条件后
实际更新一定改善或损害任务表现。还需结合梯度范数比判断结构项的影响大小。
添加 penalty 也不能直接推出有限预算下的任务 loss 一定比 vanilla 差或好。

逐专家 diff 的定义是 **绝对 Frobenius 范数**：

```text
diff_i = ||X_i - (Hbar_i + L_i + S_i)||_F
```

没有除以 `||X_i||`，没有平方、rho 或 U，也没有在专家间取平均。
当前在结构更新后记录瞬时值。此前核对原 SALAAD 的 `||X-L-S||_F` 也是绝对量，
旧日志还存在时间窗口平均和不同采样时点，因此图上的值不能不加区分地直接比较。
当前单个 projection 是 `176 × 256`；此前讨论的 `2816 × 1024` 矩阵有其 64 倍元素，
相同逐元素误差下 Frobenius 范数约大 8 倍。结构压缩程度和更新时点也影响 diff；
较小 diff 本身不证明压缩效果或模型质量更好。

### E. 最终 checkpoint 的参数量与存储分析

9 月 19 日已在集群直接读取两个 consensus run 的 step-2100 checkpoint，
只回传小型 JSON/CSV 报告，未下载模型或 checkpoint。每个 run 覆盖 24 个组、
1536 个 expert projection；实际非零数及 effective rank 与最终 W&B 记录核对一致。

潜在双因子表示的参数计数为：

```text
P = 27,988,224 + 1,081,344 + 432 * sum(actual_rank_i) + sum(nnz(S_i))
```

这里 `432=176+256`；shared 每组只计一份。
`actual_rank` 是 checkpoint 中 streaming SVT 保留的全部正分量个数，
不是覆盖 99.9% 线性奇异值质量的 effective rank。未追加截断或剪枝。
该公式是潜在的双因子参数计数，没有据此生成新的因子导出文件，也不是对存储后的
FP32 dense L 逐位可逆重分解的保证。CSR 索引不算模型参数，但要计入实际字节数。

| 数量 | rho=`1e-6`，`17573648.0` | rho=`1e-5`，`17574041.0` |
|---|---:|---:|
| 原始模型参数 | 97,194,240 | 97,194,240 |
| shared 参数 | 1,081,344 | 1,081,344 |
| actual rank 总和 | 175,440 | 107,264 |
| L 的潜在双因子参数 | 75,790,080 | 46,338,048 |
| S 非零值数 | 30,833,275 | 18,771,791 |
| 潜在双因子全模型参数 | 135,692,923 | 94,179,407 |
| 相比原模型节省参数 | **-38,498,683（增加 39.61%）** | **3,014,833（节省 3.10%）** |
| 当前 dense-L exporter 的值元素数 | 129,108,859 | 117,047,375 |
| 原始 FP32 模型 tensor bytes | 388,776,960 | 388,776,960 |
| 潜在双因子 FP32 + CSR tensor bytes | 667,356,120 | 453,056,120 |
| 当前 dense-L exporter + CSR tensor bytes | 641,019,864 | 544,527,992 |

因此，rho=`1e-5` 的约 3.01M / 3.10% 是潜在双因子的**参数值计数节省**，
不是现有导出文件已经缩小。现有 exporter 仍保存 dense L，两组导出表示都比原始
模型大。即使使用该双因子表示，rho=`1e-5` 加上 FP32 值和 int32 CSR 索引后，
tensor 存储也约为 453.06 MB，高于原模型的 388.78 MB；这些估算不含文件容器开销。
训练 checkpoint 另外含 dense X、Adam 状态、辅助状态和迭代基，其磁盘大小不能
当作推理权重的压缩率。此前近 15 GB 的多个训练输出还受到保存多份 checkpoint 的影响。

完整证据：[参数报告](../data/moe_analysis/final_parameters_20260919/README.md)、
[汇总 JSON](../data/moe_analysis/final_parameters_20260919/summary.json)、
[逐 projection 计数](../data/moe_analysis/final_parameters_20260919/groups.csv)、
[逐 expert 计数](../data/moe_analysis/final_parameters_20260919/experts.csv)。

### F. shared、L、S 的范数与 shared 是否有用

随后已分析两个最终 checkpoint 的全部 projection：数值是
`sqrt(sum(A**2))` 的绝对 Frobenius 范数，由保存的 FP32 值以 FP64 累加计算。
每组一份 H、64 份 L 和 S；展示区间时使用专家最小值–最大值，**没有专家平均**。

以 rho=`1e-5` 的第 0 层为例（其余层和另一个 run 见完整表）：

| Projection | shared 范数 | L：64 专家最小–最大 | S：64 专家最小–最大 |
|---|---:|---:|---:|
| gate | 0.776 | 0.758–1.601 | 1.235–1.790 |
| up | 0.674 | 0.805–1.687 | 1.221–1.834 |
| down | 0.248 | 0.574–1.071 | 0.403–0.764 |

完整 [48 组范数表](../data/moe_analysis/final_norms_20260919/README.md)、
[逐专家 CSV（3072 行）](../data/moe_analysis/final_norms_20260919/experts.csv)、
[交互浏览页](../data/moe_analysis/final_norms_20260919/norms.html)
已保留，支持逐 run、层、projection 查看所有专家，不增加 W&B 指标。

用户观察：shared 明显小于 S，怀疑算法把 shared 当作噪声学习。
首先区分范数与元素范围：上述表格不是元素最大/最小值，也不能直接作为功能贡献大小。
原不对齐 consensus 的 H 子问题及更新为：

```text
min_H sum_i ||X_i - L_i - S_i + U_i - H||_F^2
H_new = mean_i(X_i - L_i - S_i + U_i)
```

平均包含同层、同 projection 的全部 64 个专家。top-8 是**每个 token**的选择，
整个 batch 可以使用多于 8 个专家。当前没有按 token 数或 router 权重对这个子问题加权；
采用这些权重会改变目标与更新规则，其收益尚未通过实验验证，不能写成“加权没有意义”。
原不对齐 SALAAD 按 `group_id % world_size` 分配辅助更新 owner；新对齐算法改为
`layer_index % world_size`，让一层的 gate/up/down 一起匹配和更新，随后广播 Q。
这是辅助状态/SVD 的分工，各 DP rank 的模型前向仍保留全部专家。

原不对齐版本初始化 H 为 mean(X)、L=0、S=X-H、U=0。在独立零均值初始化的近似下，
64 个专家平均后的 H 范数典型尺度约为单个 X 的 `1/sqrt(64)=1/8`；
这个解释只适用于相应假设，不能当作训练后所有组的固定比例。
训练中不同专家的方向相消，也可以使平均项较小。

当前前向使用 dense X，任务 loss 直接更新 X；H 通过分解与约束间接接收任务信息。
“H 没有直接任务梯度”不等于“H 完全没有学习”，但 H 的直接优化目标确实是矩阵
重构共识。H 不受 L/S 对应的低秩/稀疏惩罚，能承接共有的数值分量，
**算法并未因此保证 H 就是语义上的共有知识**。此前将其直接称为“共有知识”的说明过强。
H/L/S 的分解也不能仅靠名称解释成知识、技能和噪声；H 小同样不足以证明它是噪声。

目前提出、但尚未执行的功能检验：固定已训练 L/S，分别使用训练所得 H、零 H、
与 H 同范数的随机 H，比较实际重构模型的任务表现。
必须让前向真正使用相应的 `H+L+S`；只修改辅助状态 H 而仍用原始 X 前向不会完成该检验。
该实验检验 H 是否有用，不能单独证明它承载共有语义；它也不同于已提交的
“从训练开始就没有 H”的 independent L+S 对照。这项功能消融仍未执行；
随后已实现并提交的是下文的通道对齐训练。

### G. ResMoE、通道对齐及研究主张

[ResMoE: Space-efficient Compression of Mixture of Experts LLMs via Residual Restoration](https://arxiv.org/abs/2503.06881)
是 KDD 2025 的训练后 MoE 压缩方法；压缩过程不需要校准数据或额外训练。
它联合寻找专家隐藏通道的对应关系与 Wasserstein 共享中心，再用剪枝或截断 SVD
压缩各专家残差，保留原专家和路由。其形式为 `W_i_aligned ≈ H + compressed_delta_i`；
剪枝与 SVD 是不同方案，不是我们训练中的联合 L+S 分解。

SwiGLU 的通道匹配需同步置换 gate/up 的行与 down 的对应列，以保持原专家函数。
可将 `[W_gate, W_up, W_down.T]` 的每行作为一个通道描述，联合求匹配；固定匹配后，
共享中心仍是对齐后权重的平均。它处理的是“相同通道编号未必对应相同特征”的问题。
出处：[论文 §4 与附录 B.3](https://arxiv.org/html/2503.06881v1)，
[官方实现](https://github.com/iDEA-iSAIL-Lab-UIUC/ResMoE)。

| 比较 | ResMoE | 原不对齐 consensus SALAAD |
|---|---|---|
| 时机 | 训练后压缩 | 随任务训练更新分解 |
| 通道匹配 | 显式对齐 | 没有显式对齐 |
| 共享项 | 对齐后的中心 | corrected residual 的逐位置均值 |
| 专家残差 | 剪枝或低秩近似 | L 与 S 同时存在 |

这里的 shared 都是权重表示的一部分；不能直接把非线性 expert 的输出写成
`f_shared(x)+f_residual(x)`，也不能等同于额外增加一个 always-on shared expert 分支。
通道对齐是与当前疑问相关的可研究方向，但仍以权重距离为代理，不自动证明语义共有知识。
随后已将联合通道匹配接入本项目的训练中结构更新；这是下文记录的 aligned consensus，
不是直接复现 ResMoE 的训练后压缩流程。

此前关于“claim 压缩，还是 claim consensus 后效果更好”的讨论尚属研究定位：
候选主张是**训练中的 consensus 能改善任务表现与实际存储量之间的权衡**。
这不是已证实结论，也不把 `shared + residual` 本身当作新的结构发明。
应区分“相同训练配置下是否有帮助”和“相同实际存储预算下是否更有效”；
相同 target rank/density 不等于相同实际大小。
vanilla、independent L+S、consensus H+L+S 以及训练后共享压缩可作为不同参照，
但尚未完成这些参照的完整终点质量与字节数比较，不能据现有范数或训练曲线作强结论。
之后用户明确本轮优先讨论置换是否改变 FFN 表达、以及对齐后共享是否更有效；
上面的压缩定位保留为此前讨论，不作为本轮问题的替代。

### H. 置换的作用、研究价值与通道对应关系

用户最初提出将 `X = X_hat + L + S` 换成 `X = (L + S)P`，并明确 L/S 不共享。
后续讨论收敛到“先找到专家通道的对应关系，再计算 shared”，最终认可并实现的是
`X_i ≈ Hbar_i + L_i + S_i`：共享的是 H，L/S 仍由各专家独立维护。

对置换的判断需要区分三件事：

1. 单独改一个权重矩阵的行或列，通常会改变 FFN 的输出；不能任意挪动一个矩阵就称为对称变换。
2. SwiGLU 的隐藏通道可以整体换编号。使用同一个 P，同时把 gate/up 变成
   `P W_gate`、`P W_up`，把 down 变成 `W_down P^T`，原专家对任意输入的函数保持不变。
   直观地说，是把一个神经元的输入、门控和输出连接一起搬家。
3. 若 L/S 可自由训练，且约束是置换不变的秩、稀疏度等，单独加入 `X=(L+S)P`
   不会扩大可表示的矩阵集合；P 可以吸收到 L/S 中。它可能改变优化过程，但不能据此
   宣称 FFN 表达能力增强。对齐的实际意义来自后续共享约束依赖“哪些位置对应”。

之前“shared 不 work 是因为专家的 head 没对齐、直接平均没有意义”的说法需要收紧：
这里讨论的是 FFN 通道；相同编号不保证相同功能，因此未对齐可能让不同特征互相抵消。
但**这是待验证的原因，不是已证实结论**。直接平均本身是有效的最小二乘中心，
shared 范数小既不能证明平均完全无用，也不能证明 H 是噪声。

先对齐再求平均，是检验这一假设的有价值方向：让 shared 尽量汇总对应的通道，
而不只是相同编号的位置。它仍不保证任务表现变好，也不保证得到语义上的共有知识。
匹配减少的权重距离、训练 loss 和未见数据上的任务质量是不同的判断依据。

一个 SwiGLU 通道的作用同时取决于 gate 的一行、up 的一行和 down 的一列。
实现用三部分拼成通道描述，以联合平方距离建立代价表，再求一对一指派；
不能对 gate/up/down 各自独立匹配，也不能让多个原通道都选中同一目标位置。
正式模型每个专家有 176 个隐藏通道，因此每次指派的代价表是 `176 × 176`。

如果以后用数据检验“功能是否对应”，应让不同专家接收**同一批该层输入**，
比较通道响应及其输出贡献，而不是只比较各专家自己被路由到的不同 token。
单个通道 a 的贡献是 `down[:,a] * SiLU(gate[a,:] x) * (up[a,:] x)`；
只看激活相似度而忽略 down 也不完整。当前已实现的是权重距离匹配，未增加激活校准流程。

### I. 用户认可并已实现的完整训练算法

通俗版 [LaTeX 源文件](moe_channel_alignment_20260921/moe_channel_alignment.tex) 和
[10 页 PDF](moe_channel_alignment_20260921/moe_channel_alignment.pdf) 包含完整算法及伪代码。
它们保留认可算法时的方案版本，其中“尚未接入代码”是成文时状态；
当前实现与运行方式见 [代码说明](../docs/moe_channel_alignment.md)，以下记录实现后的约定。

**状态放在哪里。** X、L、S、U、Adam 状态和 streaming 迭代基 B 都保留专家原通道顺序，
只有 H 使用共同编号。每层每个专家一张排列表，gate/up/down 共用；不同层互不对齐。
H 是同层同 projection 的 64 个专家共享一份，L/S/U/B/阈值则每专家每 projection 各一份。
代码保存 int64 的 `permutation[i,a]=b`，表示原通道 a 对应 shared 位置 b，不保存稠密 P。

若写成矩阵，即 `P_i[b,a]=1`，则：

```text
Hbar_gate_i = P_i^T H_gate
Hbar_up_i   = P_i^T H_up
Hbar_down_i = H_down P_i
Q_i = Hbar_i + L_i + S_i - U_i
```

专家 0 固定为恒等排列，只用于固定共同编号，它在平均中的权重仍为 `1/64`。
更新 P 时不移动 X 或 Adam 状态，因此不会在那一刻改变原始模型的前向输出；
P/H 会改变重构权重和后续结构梯度的目标。

**初始化。** 所有 P 先为恒等排列，用专家 0 的三个投影作为 H 模板。
最多交替进行 10 轮联合匹配与对齐后平均；用完整匹配目标比较每轮前后，
改进不超过 `1e-6 * max(1, 旧目标)` 时停止。随后设置 `L=0`、`S=X-Hbar`、`U=0`、
两个阈值为 0，并用一次普通 SVD 准备 B。此时数学上 `Q=X`（FP32 计算可有舍入误差），
不物理重排模型。随机初始权重的匹配只是建立共同编号，不能解释为已经发现共有知识。

**每个 optimizer step。** 完成下面的顺序，期间前向一直使用 dense X：

```text
task loss = LM loss + 0.1 * balance loss + 0.001 * z-loss
累积 microbatch 任务梯度 → DP 平均
所有 expert projection 的梯度各加一次 rho * (X-Q)
全局梯度裁剪到 1.0 → AdamW
若到结构更新步，执行下面的一轮结构更新
```

**每 10 步的结构更新。** 先保留旧 L/S/U，计算 `R_i=X_i-L_i_old-S_i_old+U_i_old`。
如果同时是 100 的倍数，则先用旧 H 重新匹配各专家的 R。初始化匹配 X；
训练中匹配的是扣掉各专家 L/S 后、加上误差修正的 R，不重新拿完整 X 代替它。
原通道 a 对应共同位置 b 的代价为：

```text
C_i[a,b] = ||R_gate_i[a,:] - H_gate[b,:]||²
         + ||R_up_i[a,:]   - H_up[b,:]||²
         + ||R_down_i[:,a] - H_down[:,b]||²
```

用 SciPy `linear_sum_assignment` 求一对一最小总代价指派；距离以 FP32 计算，指派在 CPU 求解。
只有 `J_candidate < J_old - 1e-6 * max(1, J_old)` 才接受该专家的新排列；
相等或改进不足保留旧排列，参考专家 0 始终固定。匹配时原顺序的 L/S/U 不动。
随后同一层的三个 H 更新为：

```text
H_gate_new = mean_i(P_i R_gate_i)
H_up_new   = mean_i(P_i R_up_i)
H_down_new = mean_i(R_down_i P_i^T)
```

平均包含全部 64 个专家，不受某个 token 的 top-8 路由限制。将新 H 对应回各专家原顺序，
对每个 projection 依次执行：

```text
A_L = X - Hbar_new - S_old + U_old
(L_new, B_new, sigma_shrunk) = StreamingShrink(A_L, B_old, tau_l_old)
A_S = X - Hbar_new - L_new + U_old
S_new = sign(A_S) * max(abs(A_S) - tau_s_old, 0)
U_new = U_old + X - Hbar_new - L_new - S_new
tau_l_new = tau_l_old + 0.02  * (effective_rank_ratio - 0.15)
tau_s_new = tau_s_old + 0.002 * (density - 0.10)
Q_new = Hbar_new + L_new + S_new - U_new
```

新阈值下一轮生效，允许为负；阈值算子内部的 `max(...,0)` 保留。
density 是 S 的精确非零比例。effective rank ratio 用收缩后的近似谱按降序累加，
取覆盖 99.9% **线性和**的最少分量数，再除以 `min(rows,cols)`；不是奇异值平方和，
总和为零时秩比为零。

**StreamingShrink 的具体步骤。** 宽矩阵先转置成高矩阵 M，令 `k=min(rows,cols)`。
保留 `k × k` 迭代基 B；每轮只做一次 QR 迭代：

```text
B_new = qr(M^T (M B_old)).Q
Z = M B_new
sigma_j = ||Z[:,j]||_2
V[:,j] = Z[:,j] / max(sigma_j, 1e-12)
sigma_shrunk_j = max(sigma_j - tau_l_old, 0)
L_new = V diag(sigma_shrunk) B_new^T
```

如果输入转置过，最后将 L 转回。沿用 FP32 结构计算、谱重构关闭 TF32、expert chunk 8；
这不是每轮重新做精确 SVD。训练结束若还有未执行的结构更新，补一次；同一步不重复更新。
完整训练包含自适应有符号阈值和近似谱更新，不能由匹配子步骤的下降推出整体全局收敛。

**多卡、保存与导出。** 对齐版本由 `layer_index % world_size` 决定该层辅助 owner，
三投影一起处理；DP4 的 8 层对应每 rank 负责 2 层。候选状态先暂存，所有 rank
共同检查错误后才提交，再广播原顺序的 Q；其余 rank 仍保留完整模型执行普通 DP 前向。
checkpoint 保存 X/Adam/数据游标/RNG、H/L/S/U/阈值/B/P、通道轴及最近结构/匹配步数。
恢复时直接加载，不重新初始匹配或 SVD；检查排列一对一、三投影一致和参考专家恒等。

对齐导出为 `salaad_moe.export.v3`，记录 `native_to_shared` 方向；不对齐版本仍支持 v2。
实际重构是 `Hbar+L+S`，不含 U；保留完整 L、S 所有非零值及训练精度，不新增截断。
没有新增训练中 validation 或 W&B 指标。

主要实现：[alignment.py](../salaad_moe/alignment.py)、[solver.py](../salaad_moe/solver.py)、
[config.py](../salaad_moe/config.py)、[trainer.py](../salaad_moe/trainer.py)、
[export.py](../salaad_moe/export.py)、[训练入口](../scripts/train_salad.py)。
新增 SciPy 依赖和正式/单进程 smoke/DP2 smoke 配置，均包含在提交 `3867b42` 中。

### J. 已有检查与后续待核实事项

此前 cosine LR、日志分区和 checkpoint 保留改动后，57 项 MoE 测试通过。
independent L+S 提交前运行并通过 2 项相关检查：无共享/固定均值消融、
结构梯度包含未被路由使用的专家且使用求和约束；运行配置比较与 `.sub` 资源核对也通过。
对齐实现后，全套 MoE 单元测试 **75 项通过**；最终导出检查调整后，又通过 alignment
12 项和 workflow 16 项。验证包括已知排列恢复、穷举指派、显式 P 参考、SwiGLU
函数不变性、完整结构更新、FP32 结构梯度、失败时不提交状态及无效排列拒绝。

还完成了实际 CPU 小模型训练检查：

- 单进程连续训练 8 步，与在第 3 步/第 4 步保存后恢复的结果逐位相同，覆盖模型、
  Adam、辅助状态、P/B、数据游标和 RNG；任务使用 BF16，辅助状态保持 FP32、P 为 int64。
- 实际 DP2/Gloo 检查通过，包括无 owner 工作的 rank、匹配错误/非法排列的集体传播、
  同层联合 owner、Q 广播，以及 8 步训练与恢复的一致性。
- 真实 CLI smoke 在第 3 步暂停、恢复到第 8 步，成功保存最终 checkpoint。
- 导出重构与对应重构模型逐位一致，不包含 U；v3 拒绝损坏的排列，旧不对齐路径测试通过。

测试入口为 [test_alignment.py](../tests/moe/test_alignment.py) 和
[DP2 检查脚本](../tests/moe/alignment_distributed_worker.py)。本地临时结果曾保存于
`/tmp/moe_alignment_dp2_20260921/validation.json` 及
`/tmp/moe_aligned_cli_20260921/checkpoints/step_00000008`，临时目录不作为长期实验档案。
本地没有可用 CUDA；软件检查通过不能代替 H100 或正式任务质量结果。
这些都是实现、提交前已完成的检查，本次文档更新未重跑训练或测试。

后续继续实验时需先核实：

1. Aligned consensus `17590593.0` 是否实际启动、日志是否正常、W&B ID 及后续结果。
   任务已经提交，先查现有任务，不重复提交；不能把提交成功或 CPU 测试写成正式训练成功。
2. Independent L+S `17579035.0` 的最终状态、checkpoint 和结果；不要把此前的
   step-1200 运行中状态或“预计完成”记录成已完成结果。
3. 新一轮 vanilla 的最终结果；优先比较 aligned 与未对齐 consensus `17574041.0`，
   再结合 independent L+S/vanilla，以相同数据、预算、指标判断共享是否更有效。
4. “只初始化对齐，之后固定 P”已可通过 `match_every_optimizer_steps: 0` 配置，
   但尚未提交该对照；当前正式提交的是每 100 步动态匹配。固定 L/S 的 H/零 H/随机 H
   功能消融也仍未执行，不把这些候选实验写成已完成结果。
5. 若继续此前存储分析，核对 actual rank、S 非零数、shared/P 开销及 CSR 索引的实际字节；
   保持“训练得到什么就导出什么”。本轮表达与共享问题的判断仍需要任务表现证据。
6. 不因这些待核实事项恢复训练中 validation、增加 W&B 指标或擅自改变原训练超参数。

本目录和 `data/moe_analysis/` 当前均由 `.gitignore` 忽略，作为本地讨论及分析资料保存；
文档更新不表示这些结果已经 commit/push 到远端。

<a id="history-through-20260917"></a>

## 2026-09-17 及以前的历史快照

以下原记录按当时状态保留，含已被替代的当前值、待办和早期测试结果。
如与上方 2026-09-22 快照冲突，以上方用户决定和本次核对的代码为准。

本文记录研究设计、代码实现、入口与调试约定，以及真实 DCLM 数据准备进度。
后续继续本项目时，先按第 1 节的用户决定理解任务，并查看第 5.1、6.3 和 7 节的数据、
启动方法及待办；早期方案中的 Megatron 对齐要求已被替代。
本文中的已有测试、预算估算和后续实验分别标明。
2026-09-13 归档补齐配置目录、原有入口兼容和 W&B 的实现约定，当日仅核对代码与记录。
2026-09-16 完成独立 MoE 框架、英文注释、云端原始语料下载和本地真实文本 smoke；
随后按用户“更新我们的讨论”的要求归档。本次归档没有新增下载、提交作业或重跑训练。

### 2026-09-17 最新决定：导出完整保留训练结果

用户明确反对额外的 `actual_rank_cap` 和 `sparse_density_cap`，要求“训练得到什么就
导出什么”。这些导出截断是此前实现额外加入的，不应继续作为当前用户要求。
删除所有 MoE 配置中的 rank/density 上限、放宽预算和 CLI 覆盖参数；删除导出时的
数值秩阈值、小奇异值丢弃、S 的 top-k 筛选和强制 BF16 转换。

当前导出直接保存训练得到的 shared 和完整 L 原值；S 的全部非零值仅无损转换为 CSR，
索引为 int32，所有权重保留 checkpoint 精度，未分解权重也按原值保存。
不再执行导出 SVD 或额外分解。新格式为 `salaad_moe.export.v2`，评估时按原顺序
计算 `shared + L + S`。训练用的 streaming SVD、阈值控制器目标和训练入口不变。
此决定替代下方历史记录中关于导出 cap、BF16 因子和最终离线 SVD 的约定。

54 项测试通过，新增/调整检查覆盖超过旧 rank 上限、100% 的 S 密度、极小非零值
和原精度保留。没有因导出修改而重跑训练。
使用现有 `20260917_134610_714044/checkpoints/step_00000008` 实际导出至
`/tmp/moe_preserved_export_20260917.pt`：6 组、48 个 expert 矩阵、36,029 个 S 非零值
全部保留；shared/L/S/未分解权重逐项原值和 dtype 一致，重新加载的 exported 模型与
checkpoint 的 reconstructed 模型逐位一致。验证结果位于
`/tmp/moe_preserved_export_20260917.verification.json`。

### 2026-09-17 补充：改用 SALAAD++ streaming SVD

用户要求用 `salaadpp` 中的 streaming SVD 替换训练中的 SVD 分解，保留一种更新算法，
删除多个 driver 和 fallback。参考本地
`/home/hao/Desktop/MPI/salaadpp/salaadpp/train/salad/solver.py` 的 `_streaming_svd_step`，
将它扩展为分块的 batched expert 更新，当前 MoE 的 `streaming_svd_step()` 每次只执行一次
正交迭代，沿用上一轮基。长、宽、方矩阵的单次更新与参考函数逐项完全相同。

新增 `GroupState.svd_basis`，形状为 `[experts, k, k]`、FP32，`k=min(out,in)`。
初始化按 SALAAD++ 做一次普通 SVD，为专家残差准备基；保持原有 shared/L/S/U 初始值。
每次结构更新返回新基，随辅助状态保存到 checkpoint，恢复后继续相同的迭代。
训练循环只保留 streaming 路径，删除 `svd_backend/svd_driver/svd_fallback_driver` 配置，
保留控制 expert 批大小的 `svd_chunk_size`。离线导出仍用一次普通 SVD 来实施硬 rank cap；
普通 SVD 均直接调用 PyTorch 默认实现，无 driver 选择或重试。

streaming 列范数是近似奇异值；秩控制器先排序，再按现有的线性奇异值质量计算覆盖率，
不改为平方能量。阈值继续允许负向超调。每个 expert 增加 `k*k` 个 FP32 基元素，
旧版缺少基的 checkpoint 不能直接续训，未加入兼容或回退路径。

53 项测试通过，覆盖长宽矩阵、秩亏/零矩阵、分块处理、基不被原地修改，以及带 streaming
基的逐位一致 checkpoint 恢复；双进程 owner/非 owner 的保存恢复与错误传播检查通过。
从统一 `main()` 无参数运行本地真实 DCLM GPU BF16 smoke，完成 8 步、4 次结构更新，
step-8 checkpoint 中 6 个组均保存 `[8,24,24]` 的基。
输出：`data/salaad_moe/smoke_dclm_bf16/20260917_134610_714044/`，离线 run ID `c074a761`；
raw/reconstructed validation NLL 分别为 `10.809171199798584` / `10.809181213378906`。
这些验证不构成正式预训练质量或吞吐提升的结论。

### 2026-09-17 补充：控制器允许负向超调

用户确认仅移除 `tau_l/tau_s` 积分控制器更新后的 `.clamp_min(0)`，允许阈值超调到负数。
配置改为 `nonnegative_projection: false`，checkpoint 恢复接受有限的负阈值，继续拒绝
NaN/Inf。奇异值阈值 `(sigma - tau_l).clamp_min(0)` 和稀疏幅度阈值
`sign(x) * (abs(x) - tau_s).clamp_min(0)` 保持原样；负阈值可以产生扩张。
此决定替代 2026-09-08 推导中额外加入的控制器非负投影假设，与原 SALAAD 的有符号
积分更新一致。训练入口及初始化、结构更新调度保持现有约定。

51 项测试全部通过，覆盖控制器跨零、冻结后保持负值、负阈值算子行为，以及包含负阈值的
checkpoint 逐位一致恢复。无参数 GPU BF16 smoke 在 `cuda:0` 使用本地真实 DCLM
完成 8 步、4 次结构更新和完整 step-8 checkpoint。
输出：`data/salaad_moe/smoke_dclm_bf16/20260917_133310_110002/`，离线 run ID `1d66ad4c`；
raw/reconstructed validation NLL 分别为 `10.809196949005127` / `10.809221744537354`。

### 2026-09-17 补充：SALAAD 在训练开始前初始化

用户要求从一开始就初始化 SALAAD，采用已解释的初始状态。
`configs/ns97m.yaml` 和 `configs/smoke.yaml` 的 `state_initialization_step` 均改为 0，
正式 97M/480M 和本地 smoke 的继承配置随之生效。`Trainer.__init__()` 使用既有分支
调用 `manager.initialize(0)`：`shared=mean(X)`、`L=0`、`S=X-shared`、`U=0`，
准备 Q 后再执行首次 optimizer 更新。smoke 的结构更新步为 2、4、6、8。
可选 vanilla 前缀分支实验需在专用配置中显式请求延迟初始化。

51 项现有测试全部通过，覆盖训练前已初始化、结构调度、通信、日志、导出及逐位一致的
checkpoint 恢复。无参数 GPU BF16 smoke 完成 8 步、4 次结构更新和完整 checkpoint。
输出：`data/salaad_moe/smoke_dclm_bf16/20260917_124310_594686/`，离线 run ID `f072e084`；
raw/reconstructed validation NLL 分别为 `10.809196949005127` / `10.809221744537354`。

### 2026-09-17 补充：单进程也初始化 dist

用户要求即使 `world_size=1` 也模拟分布式执行。`initialize_device()` 现在为单进程
创建真实进程组，固定 `rank=0/world_size=1`；CUDA 使用 NCCL，CPU 使用 Gloo。
单进程使用 `HashStore`，无需填写 master 地址或端口；多进程仍使用 `torchrun` 环境。
训练中 `dist.is_initialized()` 为 `True`，参数广播、梯度归约和其他集体通信都会执行；
`main()` 的既有 `finally` 在退出时销毁进程组，IDE 仍直接 F5、无需参数。

7 项入口测试通过，其中暂停/恢复测试新增核对真实单进程进程组、广播/归约执行和退出清理。
无参数运行在 `cuda:0` 完成 8 步、4 次 SALAAD sweep、完整 step-8 checkpoint 和 W&B
离线记录。输出：`data/salaad_moe/smoke_dclm_bf16/20260917_123053_821598/`，
run ID `4060f0d1`；raw/reconstructed validation NLL 分别为
`10.809207916259766` / `10.80920934677124`。

### 2026-09-17 补充：IDE 无参数调试

用户确认直接从 IDE debug 启动，不输入命令行参数。已将 `.vscode/launch.json`
设为 `MoE: local DCLM smoke`：固定 `myenv/bin/python`、`scripts/train_salad.py`
和仓库工作目录，`args=[]`，单进程、W&B 离线；移除传入旧 vision 配置的调试项。
保留 `__main__` 中显式参数赋值和 `main(...)` 调用，`main()` 自身也默认使用真实文本 smoke。

默认配置 `smoke_dclm_bf16`、本地 DCLM token manifest、BF16、8 步、`rho=0.05`、
4 个 CPU 线程；自动选择 CUDA，否则 CPU。每次自动创建新的时间戳输出目录，
默认新建实验并实际训练。`rho/num_total_iters=None` 表示读取 YAML，
`output=None` 表示自动目录，恢复/分支路径为 `None` 表示不恢复，都不需要用户输入。
新增代码注释使用英文；未修改模型、优化器或 SALAAD 更新算法。

验证：7 项既有入口测试通过（含公共 `main()` 断点和暂停/恢复）；真正不带任何参数运行
`myenv/bin/python scripts/train_salad.py`，在 `cuda:0` 完成 8 步、1,024 个预测 tokens、
4 次 SALAAD sweep，保存完整 step-8 checkpoint，W&B 离线 run ID `63c6e01d`。
输出目录：`data/salaad_moe/smoke_dclm_bf16/20260917_122345_921292/`。
raw validation NLL `10.809207916259766`，reconstructed NLL `10.80920934677124`。

### 2026-09-17：本地启动默认使用真实 DCLM 切片

用户要求将 config 和 data-manifest 默认值放在 `parse_args()` 中。
当前 CLI 默认 `cfg_version=smoke_dclm_bf16`，数据默认指向仓库根目录下的
`data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json`，解析为绝对路径。
可直接运行 `myenv/bin/python scripts/train_salad.py`；`train_moe.py` 共用相同参数解析。
显式 `--data-manifest` 可以覆盖路径，选择其他配置或恢复/分支 checkpoint 时仍使用对应
配置或原实验的数据设置。保留了 `__main__` 显式参数赋值和 `main(...)` 调用结构。
已核对默认 manifest 确实读取真实文本，并通过 7 项既有入口检查；本次没有重跑 GPU 训练。

### 2026-09-16 最新决定：独立 MoE + SALAAD 框架

用户进一步要求：训练不再兼容原来的 dense LLM/Vision，以 SALAAD 为结构框架重写 MoE，
添加便于理解的注释。这一决定替代下方历史记录中的“统一入口仍兼容 vision/dense”要求。
用户明确要求代码 comments 和 docstrings 使用英文，后续新增注释也遵循这一约定。

当前 `scripts/train_salad.py` 是 MoE 专用入口，保留显式的 `__main__` 参数赋值和
`main(...)` 调用，已去除旧模型 JSON、旧训练器分流、HF login 和 vision 模型初始化。
`scripts/train_moe.py` 仅导入同一个 main 作为命令别名，不再独立组织后端或回调主脚本。
`salaad_moe/runtime.py` 准备设备/DP、输出目录和语料；`trainer.py` 组织任务训练；
`solver.py` 独立管理 SALAAD 共识分解。模型、数据、训练器、solver 和配置的关键步骤均补充
英文注释，阅读顺序见 [代码指南](../docs/moe_training_walkthrough.md)。

49 项测试通过，覆盖拒绝旧配置、禁止旧 LLM/Vision 模块导入、公共断点和暂停/恢复。
固定种子的 8 步 SALAAD/vanilla CPU 对比中，重写前后的模型、Adam、SALAAD 状态、数据游标
和 RNG 逐项完全一致。新入口的单 GPU BF16、双进程 CPU DP 各 8 步短跑通过，真实 W&B SDK
离线记录正常。详情见 [验证记录](../docs/moe_framework_validation.json)。这是小模型软件验证，
没有新增 97M/480M 正式预训练或多 GPU CUDA 性能结果。

随后按用户要求将代码、配置和测试中的中文注释及 docstrings 改为英文。对比修改前后的
27 个 Python 文件（忽略 docstrings 的 AST）和 7 个 YAML，训练逻辑与配置值均未改变；
语法检查通过，没有重跑训练测试。验证记录保留原测试版本的哈希及本次注释修改后的哈希。

### 2026-09-16：真实 DCLM 下载与本地 smoke

按用户要求先检查空间：云端共享文件系统约 19 TiB 可用，本地约 54 GiB 可用。
云端保存至 `/lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916/`，
下载固定 DCLM revision 的 48 个完整原始切片，共 10,502,832,978 字节，全部逐文件
核对官方 SHA-256。分词器哈希与此前固定版本一致。云端仍保留 `feature/salaad-vision`，
本地仍为 `feature/moe-salaad`，此次没有同步代码或切换分支。

本地下载首个 141 MB 切片，生成 1,048,576 个训练预测 tokens 的真实文本语料；
`configs/smoke_dclm_bf16.yaml` 使用 Pythia 真实词表且关闭随机数据生成。通过原来的
`scripts/train_salad.py` → `main()` 完成 8 步 GPU BF16 smoke，SALAAD 4 次 sweep、
验证、checkpoint、真实 W&B SDK 离线记录均正常；run ID 为 `570b3895`。
数据代码补充 `.zst/.zstd` 支持及 `--max-train-tokens` 上限，15 项 workflow 测试通过。

云端现在保存原始压缩语料，尚未分词或确认其精确 token 数，也未生成完整 4.4B-token
训练文件。CPU 预处理等待用户提供 HTCondor bid，并需要安排匹配的 MoE 代码版本。
详见 [数据记录](../docs/moe_dclm_data.md) 和 [验证结果](../docs/moe_dclm_data_validation.json)。

### 2026-09-16：所有训练先进入统一 main（历史入口调整）

以下是独立 MoE 重写前的入口修复过程。旧 vision/dense 分流、`train_moe.train()` 和
模型 JSON 参数均已被上方最新实现替代；保留下来的用户要求是所有训练进入同一个
`main()`，并保留 `__main__` 中显式的参数赋值与调用方式。

用户明确要求：

> 我要求所有训练都从主入口main进入

之前在 `scripts/train_salad.py` 的模块顶层识别 MoE，直接调用另一个训练入口后退出，
绕过了原来的 `main()`，因此其内部断点不会命中。该实现已移除。
现在脚本底部只解析参数并调用 `main()`；配置识别和 MoE/vision/dense 训练器选择均在
该函数内进行。`scripts/train_moe.py` 的直接启动命令也先进入 `train_salad.main()`，
MoE 后端由其调用 `train_moe.train()`，不再重复解析命令行。
需要共用断点时，设置在 `train_salad.main()` 的首条可执行语句。

用户随后明确反对改写原来的 `if __name__ == "__main__":` 代码结构。已恢复原有的
`args`、`cfg_version`、`folder`、`path_cfg`、`path_cfg_model`、`exclude_layers`
逐项赋值和显式 `main(...)` 调用，仅追加 MoE 所需参数。后续修复应保留这套写法，
不要再顺带简写成 `main(**vars(parse_args()))` 或重排既有入口结构。

本次 `myenv/bin/python -m unittest discover -s tests/moe -v` 共 47 项通过，新增验证覆盖：
两个命令实际命中统一 main 内的 Python 断点、函数参数不被进程 argv 覆盖、CPU 短跑
暂停/恢复与完整调度。另用隔离分布式初始化的方式确认 dense 与 Vision 配置进入旧训练分支。
日志测试使用 mock W&B；本次未重跑多进程/GPU/真实 SDK 测试，未新增正式训练结果。

## 1. 用户最新确认的默认要求

用户原话：

> 我如果没有特殊指明则接下来的讨论都是基于dp的,我不需要跟参考论文对齐 我只要跟vanilla训练对齐

具体执行口径：

1. 后续讨论默认普通数据并行 DP，`EP=TP=PP=CP=1`。
2. 当前仅考虑没有 always-on shared expert 的普通 MoE。
3. 使用 `scripts/train_salad.py --cfg_version <名称>` 读取 `configs/<名称>.yaml`，
   经显式 `main()` 直接创建 MoE Trainer；入口现只支持 MoE，配置位于 `configs/` 根目录。
4. 对照对象是同一训练入口、同一模型架构下的 vanilla MoE。
5. 参考论文只提供模型规模、架构和数据等背景；复现其训练框架、运行环境或绝对分数
   不构成当前实验的要求。
6. 现有 `scripts/train_moe_megatron.py` 保留为可选适配路径，不是默认入口、前置依赖
   或当前实验的完成条件。用户尚未要求删除此实现。
7. 初步验证暂不使用 2.5B。优先 97M，再扩展到 480M；614M 可作为后续扩展。
8. MoE 配置默认 `is_wandb: true`，项目名为 `SALAAD-MoE`；仅 DP rank 0 记录。
9. 代码 comments 和 docstrings 统一使用英文；讨论说明和阅读指南可以使用中文。
10. `__main__` 保留清晰的参数逐项赋值和显式 `main(...)` 调用；不要顺带改成
    `main(**vars(parse_args()))`。公共训练断点设在 `main()` 第一条可执行语句。
11. 已授权先检查空间、下载云端 DCLM 子集，并在本地下载小切片进行 smoke test。
    目前尚未收到 CPU 预处理作业的 HTCondor bid，也没有启动正式 97M/480M 训练。

第 3、8 条来自 2026-09-09 的用户纠正：

> 首先你的config保存的位置就不对,你读我的代码了吗,我读取的是configs/下的文件, 你多加一层moe我怎么读取. 其次enable wandb 并且项目名改为SALAAD-MoE

后续修改配置或接入训练功能时，应先检查现有读取逻辑，保持
`scripts/train_salad.py --cfg_version` 的用法，并确认配置字段实际被训练器使用。
2026-09-17 起 CLI 默认配置为 `configs/smoke_dclm_bf16.yaml`，默认读取已准备的真实
数据切片；随机 token 测试可显式选择 `configs/smoke_bf16.yaml`。启动和 W&B 设置见第 6 节。

DP 是并行策略，Megatron 是可采用 DP 的训练框架，两者不是互斥选择。两个现有入口
都只支持 DP；当前选择其中的原生 PyTorch 入口完成 vanilla/SALAAD 配对实验。

## 2. 研究设想与当前实现

在同一层、同一种投影内，对 K 个 expert 权重施加：

```text
X_i = X_hat + L_i + S_i,  i = 1, ..., K
```

- `X_hat`：同组全部 experts 的共识矩阵。
- `L_i`：每个 expert 独立的低秩残差。
- `S_i`：每个 expert 独立的逐元素稀疏残差。
- 每层的 gate、up、down 分别建立组；不跨层共享，不分解 router。
- 模型前向和任务训练使用完整 dense 权重 `X_i`；分解项及 dual 是辅助状态。
- `X_hat` 是 expert 内部的共享矩阵，不增加一个始终激活的 shared expert FFN。
- 当前所有组和 experts 共用一个固定正数 rho；每个 expert 有独立阈值。
  早期推导讨论过 expert-specific rho 的推广，当前代码没有采用该推广。

当前更新顺序：累积任务梯度 → DP 平均 → 一次加入 `rho * (X - Q)` → 全局梯度裁剪
→ AdamW → 到达结构周期时更新 shared/L/S/dual。其中 `Q = X_hat + L + S - U`，
`U` 为 scaled dual。约束梯度包含全部 experts，不按路由使用次数加权。

当前 97M/480M 配置在第 0 步、首次 optimizer 更新前初始化辅助状态，以后每 10 步
执行一次结构更新。有效秩统计采用奇异值之和的覆盖率，不是奇异值平方和。

## 3. DP 和 SALAAD 的分工

每个 DP rank 都保存完整模型，处理不同样本，并同步任务梯度。
SALAAD 按“层 × projection”的逻辑组分配辅助更新负责人：

```text
owner = group_id % DP_world_size
```

负责人保存该组的 shared/L/S/dual/阈值，执行 SVD、稀疏阈值和对偶更新；更新后广播 Q，
供各 rank 对自己的模型副本加入相同约束。其他 rank 不重复执行该组的分解。
这项分工分摊辅助状态和 SVD 开销，没有把模型前向计算按层切分到不同 GPU。

## 4. Vanilla 对齐原则

vanilla 和 SALAAD 使用同一原生 DP 训练器，并固定：

- 模型结构、初始权重和随机种子。
- 语料 manifest、tokenizer、packing、训练样本顺序及验证/测试集合。
- GPU 数和型号、DP 数、序列长度、每卡 micro batch、梯度累积和全局 batch。
- 计算精度、attention 重计算、AdamW、weight decay、学习率及完整调度、梯度裁剪。
- 训练成功步数和预测 token 预算、评估口径及 checkpoint/评估频率。

vanilla 保留相同的 router、load-balancing loss 和 z-loss。
SALAAD 组只增加分解约束、辅助更新以及必要的结构诊断。
主比较使用相同 token 预算，另外报告 SALAAD 的 SVD/通信耗时、总体训练耗时、峰值显存
和导出存储量；额外开销不能隐藏在质量比较之外。

现有 `ns97m_vanilla.yaml` 继承 `ns97m.yaml`，只改实验名并关闭 `salaad.enabled`。
默认使用相同初始化从头跑两组。若另行选择共享 vanilla 第 100 步 checkpoint 的实验，
需在目标配置中显式设定 `state_initialization_step: 100`，再由 SALAAD 通过
`--branch-from` 继承模型、Adam、数据游标和 RNG，建立辅助状态。
单独改变一组的 LR、batch、数据顺序或预算后，不能继续称为同条件配对对照。

`--stop-after 200` 表示在既定完整调度的第 200 步暂停，不把总训练步数改成 200。

## 5. 模型规模与预算判断

MirrorMoE 附录 B.6 / Table 7 的无 shared expert 模型如下。参数量是论文的近似命名：

| 总参数 | 每 token 激活参数 | MoE 层数 | 每层 experts | top-k | 当前范围 |
|---|---|---|---|---|---|
| 97M | 36M | 8 | 64 | 8 | 首轮验证，已有配置 |
| 480M | 115M | 12 | 64 | 8 | 规模扩展，已有配置 |
| 614M | 95M | 8 | 128 | 8 | 可后续扩展，尚无独立配置 |
| 2.5B | 270M | 16 | 128 | 8 | 当前初步验证不采用 |

来源：[MirrorMoE 本地 PDF](../paper/MirrorMoE_ICLR2027.pdf)。
正文的 100M/498M/625M/2.4B 含 shared experts，是另一组架构，不作为当前训练配置。

当前原生实现的精确参数计数：

| 配置 | 总参数 | 激活参数计数 | 建议硬件 / DP | 全局 batch / 长度 |
|---|---|---|---|---|
| NS97M | 97,194,240 | 36,638,976 | 2 × H100 80GB / DP2 | 1024 / 2048 |
| NS480M | 479,736,320 | 116,404,736 | 4 × H100 80GB / DP4 | 1024 / 2048 |

上述激活计数把 embedding 和 LM head 的全部参数计入；它与论文名称 A36M/A115M
并不完全相同。正式记录使用实际模型计数。

在上述 H100 80GB 和长度 2048 的前提下，初步验证优先采用 DP；614M/DP4 目前只是
可行性判断，尚未完成对应配置、峰值显存和吞吐验证。2.5B 不是并行方式的硬分界。

早期混合精度预算估计 NS97M/DP2、NS480M/DP4 的每卡静态状态约为 2.28/10.76 GiB，
包含辅助状态，采用 FP32 masters/梯度/Adam 加 BF16 model copy 的保守口径。
原生入口实际为 FP32 参数加 BF16 autocast；静态预算不等于峰值显存，还需计入激活值、
SVD 工作空间、结构候选状态、通信缓冲及评估副本。当前没有这些完整尺度的 GPU 实测。

### 5.1 语料来源、大小与本次下载范围

参考论文是 MirrorMoE（《Rethinking Routing Exploration in Modern Mixture-of-Experts》）。
正文第 4 节和附录 B.1 明确采用 DCLM，但没有公开可唯一确定原实验数据子集的完整 shard
manifest。当前方案具体选择 `mlfoundations/dclm-baseline-1.0`，这是实现中的固定版本选择。

2026-09-16 查询的 [DCLM 官方数据集页面](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
标注全库约 4T tokens、3B 文档、7.2 TB 文件总量；这些数字描述完整语料库。

| 讨论中的数据规模 | 含义 |
|---|---|
| 默认随机 smoke | train/validation/test 分别 128/32/32 条，序列长 32；三个 token 文件共 24,588 字节 |
| 97M 正式预算 | `2100 × 1024 × 2048 = 4,404,019,200` 个训练预测 tokens，通常简称 4.4B |
| 4.4B tokens 的存储估算 | 当前 uint32 格式每 token 4 字节，约 17.6 GB / 16.4 GiB；原始文本、缓存和 held-out 文件另计 |
| 本次云端下载 | 48 个原始压缩 JSONL 切片，共 10,502,832,978 字节（10.50 GB），已全部校验 |
| 本次本地真实文本 smoke | 原始切片 141,306,362 字节；准备 1,048,576 个训练预测 tokens，train token 文件约 4 MiB |

云端 10.50 GB 是压缩文本字节数，尚未分词确认总 token 数，不能记录为已经完成精确的
4.4B-token 数据集。后续实际分词达到目标后截取；不足时需继续准备分片。

固定数据 revision 为 `a3b142c183aebe5af344955ae20836eb34dcf69b`；tokenizer 为
`EleutherAI/pythia-12b`、revision `bb1e3e710cdf6b524461d543cfb5ba773f0a81b6`。
本次下载 tokenizer 文件，没有下载 Pythia 模型权重。完整路径及数据清单见
[DCLM 数据记录](../docs/moe_dclm_data.md)。

## 6. 入口和配置

默认 MoE 训练入口：[scripts/train_salad.py](../scripts/train_salad.py)，使用 `--cfg_version`。
MoE 命令别名：[scripts/train_moe.py](../scripts/train_moe.py)，调用同一个 `main()`。
训练器：[salaad_moe/trainer.py](../salaad_moe/trainer.py)。
共识更新：[salaad_moe/solver.py](../salaad_moe/solver.py)。

| 用途 | 配置 |
|---|---|
| 单进程 8 步小模型 | [smoke.yaml](../configs/smoke.yaml) |
| 双进程 DP 小模型 | [smoke_dp2.yaml](../configs/smoke_dp2.yaml) |
| 单 GPU BF16 + attention 重计算小模型 | [smoke_bf16.yaml](../configs/smoke_bf16.yaml) |
| 真实 DCLM 文本、Pythia 词表、单 GPU BF16 小模型 | [smoke_dclm_bf16.yaml](../configs/smoke_dclm_bf16.yaml) |
| 97M SALAAD / DP2 | [ns97m.yaml](../configs/ns97m.yaml) |
| 97M vanilla / DP2 | [ns97m_vanilla.yaml](../configs/ns97m_vanilla.yaml) |
| 97M independent L+S 消融 | [ns97m_independent_ls.yaml](../configs/ns97m_independent_ls.yaml) |
| 480M SALAAD / DP4 | [ns480m.yaml](../configs/ns480m.yaml) |

尚无独立的 480M vanilla 配置文件；进入该尺度配对实验前，应继承 `ns480m.yaml`
并仅修改实验名、关闭 SALAAD。`configs/` 是运行配置目录，早期讨论文件夹中的
`configs/` 是当时的方案规格，不应混用。

数据准备入口为 [prepare_moe_data.py](../scripts/prepare_moe_data.py)。
完整准备、短跑、恢复、导出及评估命令见 [运行说明](../docs/moe_salaad.md)。

### 6.1 本地测试与配置读取

本地默认真实文本测试可直接启动：

```bash
myenv/bin/python scripts/train_salad.py
```

如需使用随机 token 的单 GPU 测试，显式选择：

```bash
myenv/bin/torchrun --standalone --nproc-per-node=1 scripts/train_salad.py \
  --cfg_version smoke_bf16
```

配置继承链为 `configs/smoke_bf16.yaml` → `configs/smoke.yaml` → `configs/ns97m.yaml`，
三个文件均位于仓库 `configs/` 根目录。入口解析继承配置后直接创建原生 MoE
DP 训练器；模型结构直接来自 YAML，无需额外的 `<名称>_model.json`。
2026-09-16 起该入口不再支持旧 vision/dense 配置。

| 本地 BF16 smoke 设置 | 值 |
|---|---|
| DP / 层数 / hidden / expert FFN | 1 / 2 / 32 / 24 |
| experts / top-k / shared experts | 8 / 2 / 0 |
| 序列长度 / 全局 batch | 32 / 4 |
| 每 rank micro batch / 梯度累积 | 2 / 2 |
| 精度 / 重计算 | BF16 autocast，FP32 参数 / selective attention |
| optimizer steps / 总预测 tokens | 8 / 1024 |
| peak LR / min LR / rho | 0.001 / 0.0001 / 0.05 |
| 辅助状态初始化 / 结构更新间隔 | 第 0 步、训练前 / 每 2 步 |

smoke 的 `data.synthetic_smoke: true` 自动生成随机 token 语料。
默认输出为 `data/salaad_moe/smoke_bf16/<timestamp>/`，语料写在该次输出的 `data/` 下。
`--folder` 修改输出分组，`--output` 指定运行目录；正式配置需提供真实语料 manifest。
本地双进程 CPU 测试改用 `--cfg_version smoke_dp2 --device cpu`，进程数改为 2。

两个原生入口均支持 `--cfg_version` 或显式的 `--config`，二者择一。
MoE 的 `--num_total_iters N` 是 `--stop-after N` 的兼容别名，表示在第 N 个成功更新后暂停，
保留完整的 LR/结构调度。恢复时省略 `--output` 会沿用 checkpoint 所在运行目录，
语料 manifest 可从该运行的 `run_metadata.json` 恢复。

### 6.2 W&B 默认与恢复行为

基础配置 [ns97m.yaml](../configs/ns97m.yaml) 统一设置，当前八个 MoE 配置继承生效，
包括 vanilla 和 smoke：

```yaml
is_wandb: true
wandb_project: SALAAD-MoE
wandb_entity: hao-ma-eth-z-rich
```

原生 DP 训练器通过 [tracking.py](../salaad_moe/tracking.py) 接入真实 W&B SDK，
仅 rank 0 初始化、记录和结束 run。记录合并配置、训练 loss/NLL、学习率、梯度范数、
raw/reconstructed 验证结果、路由统计及 SALAAD 分解诊断；横轴为 `optimizer_step`。
逐 expert 原始路由计数保留在本地 `metrics.jsonl`，W&B 调用前后保存和恢复训练 RNG。

默认在线记录，使用现有 W&B 登录或 `WANDB_API_KEY`。`WANDB_MODE=offline` 或配置中的
`wandb_mode: offline` 可用于离线调试，环境变量优先；基础配置仍默认在线。
新增 `smoke_dclm_bf16.yaml` 显式设置离线模式，真实文本测试 run ID 为 `570b3895`。
密钥不写入 YAML 或讨论记录。

输出目录的 `wandb_run.json` 保存 run ID、项目及最后记录步数。
在原目录从最新 checkpoint 恢复时复用 run ID；回退到更早 checkpoint 或改用新输出目录时
创建新 run。W&B 初始化、训练日志错误会传播到全部 DP ranks。

### 6.3 当前数据位置与真实文本测试复用

云端主机：`hma2@login.cluster.is.localnet`。数据根目录：

```text
/lustre/home/hma2/projects/awd-LoRA/data/moe_corpora/dclm_20260916/
```

其中 `download_plan.json` 记录选定分片，`sources/shards.json` 记录已下载文件的绝对路径、
字节数和 SHA-256，`sources/tokenizer/` 保存分词器，`preparation_status.json` 保存完成状态。
下载前共享文件系统约有 19 TiB 可用，个人 quota 工具未给出可用额度；48 个文件写入和
校验均成功。云端代码仍为 `feature/salaad-vision`，本地为 `feature/moe-salaad`，未进行
跨分支代码同步。

本地数据 manifest：

```text
data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json
```

本地 `sources_local/shards.json` 记录完整原始切片的 HF 缓存路径；
`sources_local/tokenizer/` 保存分词器。下载前本地约 54 GiB 可用，完成后约 53 GiB。
真实文本语料的 train/validation/test 分别有 32,768/16/16 条 32-token 序列。

复用已准备的数据时，从主入口启动：

```bash
myenv/bin/python scripts/train_salad.py --cfg_version smoke_dclm_bf16 \
  --data-manifest data/moe_corpora/dclm_20260916/smoke_tokens/manifest.json
```

此次已完成的测试输出为 `data/salaad_moe/smoke_dclm_bf16/20260916_real_text/`，
第 8 步 checkpoint 完整。复跑时使用默认的新输出目录，避免覆盖此次结果。

### 6.4 本次代码阅读中的说明

- `validate_config(config)` 检查配置结构、参数范围、当前实现支持的模式及参数间的一致性；
  通过时返回 `None`，失败时抛异常，不修改配置或模型。传入 `world_size()` 后还检查实际
  启动进程数是否与配置一致。
- `prepare_corpus(...)` 优先采用显式 manifest，再读取配置路径或从 resume/branch 的实验
  记录恢复路径；仅 synthetic smoke 配置允许自动生成随机语料，最终返回 `TokenCorpus`。
- `corpus` 是只读 token 语料对象。`manifest.json` 描述文件、划分、长度和校验和；文件用
  `np.memmap` 访问，读取 batch 时才把对应输入和标签搬到设备上。
- `prepare_corpus` 的 device 参数用于集体错误检查；`allow_synthetic` 控制是否接受合成
  测试语料，不能据此把随机 token 当作真实 DCLM。
- 数据读取器从 T+1 个 token 生成 `inputs=tokens[:-1]`、`labels=tokens[1:]`，模型中无需
  再次 shift。`.zst/.zstd` 现已支持流式读取。
- `--max-train-tokens N` 是准备阶段的预测 token 上限，N 必须是序列长度的整数倍；
  train 文件最多保存 N+1 个 token。达到 train 上限后仍继续填充验证/测试集合。
- AdamW 参数分组中 `ndim >= 2` 包含矩阵和堆叠 expert 权重，施加 weight decay；
  `ndim < 2` 包含 normalization scale 等向量，不施加 weight decay。两组参数都会训练。

## 7. 已有验证与未完成工作

最新真实数据验证见 [moe_dclm_data_validation.json](../docs/moe_dclm_data_validation.json)：
15 项 workflow 检查通过，包含新增的多帧 Zstandard 读取和 token 上限检查；GPU BF16
完成 8 步、消费 1024 个预测 tokens、4 次 SALAAD sweep。最终 raw/reconstructed 验证
NLL 分别为 10.8092079163 / 10.8092093468，W&B 离线记录与 checkpoint 均完整。
这 15 项是本次执行的检查，不表示在真实数据准备改动后重跑了之前的全部 49 项。

独立框架重写验证见 [moe_framework_validation.json](../docs/moe_framework_validation.json)。
更早的历史验证见 [moe_salaad_validation.json](../docs/moe_salaad_validation.json)。

2026-09-09 配置路径与 W&B 修正后：44 项测试及双进程通信检查通过；覆盖原入口读取、
rank 0 日志、集体错误传播、恢复 run ID，以及日志开关不改变参数、Adam 和训练 RNG。
通过原有 `train_salad.py` 入口完成单 GPU BF16、双进程 CPU 各 8 步随机语料短跑，
真实 W&B SDK 的离线记录通过，项目名确认为 `SALAAD-MoE`；没有测试云端上传。

两次真实 SDK 短跑分别生成一个 run，离线文件均包含第 1–8 步的完整指标记录：

| 验证 | 本地输出目录 | 离线 run ID |
|---|---|---|
| 单 GPU BF16 | `/tmp/moe_root_wandb_bf16_20260909` | `105dd0c9` |
| 双进程 CPU DP | `/tmp/moe_root_wandb_dp2_20260909` | `23b58a80` |

这些 `/tmp` 路径是当时测试产物的位置，长期验证摘要保存在上面的 JSON 记录中。
2026-09-13 核对该记录中的 35 个源码/配置/文档 SHA-256，全部与当前文件一致；
本次没有重跑训练测试，也没有新增正式实验结果。

2026-09-08 初始实现阶段的历史验证：

- 35 项单元测试通过。
- 双进程 CPU/Gloo 检查通过，包括空 owner、SVD 全局失败传播和 owner checkpoint 恢复。
- 2 层、hidden 32、8 experts、top-2 的 8 步合成数据短跑：CPU 单进程、CPU DP2、
  单 GPU BF16 均通过。CPU DP1/DP2 对比，以及 CPU DP2、单 GPU BF16 从第 4 步恢复，
  对应模型、Adam 和辅助状态的最大绝对差异均为 0。
- 实际 BF16/factor/CSR 文件的导出、读取和物化评估已有小模型验证。
- 本地 GPU 是 RTX 3050 Ti Laptop 4 GiB；上述单 GPU 结果不是 H100 或多 GPU CUDA 结果。

这些是软件正确性检查，尚未获得完整 DCLM 预训练、97M/480M H100 质量/吞吐结果、
614M 实测、六个真实下游任务结果或原生 CSR/factor 推理 kernel 的加速结果。
现有 Megatron 路径只做过协议/映射测试，完整 GPU 路径未实测；它不阻塞当前原生 DP 目标。

下一步按依赖顺序推进：

1. 取得 CPU 预处理作业的 HTCondor bid。已发出询问，用户尚未提供数值。
2. 安排云端匹配的 MoE 代码版本；当前本地/云端分支不同，不能把本地已实现等同于云端
   已部署。本次未 commit/push MoE 工作区，也未切换云端原有分支。
3. 在计算节点对已下载分片进行分词、去重和划分，核对是否达到 4,404,019,200 个训练预测
   tokens 及所需 held-out 规模；不足时补充固定分片，完成后发布可训练的 manifest。
4. 数据与代码就绪后，再安排 97M vanilla/SALAAD 配对短跑到 150–200 步，覆盖第 0 步
   初始化及后续结构更新，测量峰值显存、NLL、SVD 耗时和吞吐，再考虑完整预算与 480M。

出价要求来自 [mpi-cluster-condor 技能](/home/hao/.codex/skills/mpi-cluster-condor/SKILL.md)：
“Do not invent a price if the user has not provided one and the current expected bid is unknown.”
截至本次归档，未提交 CPU 预处理或正式训练作业；“更新讨论”没有补充 bid 或新的训练参数。

更新时 MoE 代码、配置和测试仍是工作区未提交内容；已有分支跟踪关系不表示这些新增文件
已经提交或推送。本目录按当前 `.gitignore` 规则保留为本地讨论记录。

## 8. 历史材料

- [基础公式推导](moe_salaad_formulation_20260908/README.md)：LaTeX/PDF 和局部公式数值核验。
- [早期详细训练方案](moe_salaad_training_plan_20260908/README.md)：包含配置、来源和预算；
  其中要求对齐 FLAME/Megatron 的部分已被第 1 节的决定替代，原 PDF 保留为历史设计稿。
- [讨论索引](README.md)：其他 SALAAD/Vision 记录的入口。
