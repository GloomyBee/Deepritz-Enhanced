# 已有实验证据链草案 v1（2026-03-17）

## 目的

本文件用于把当前 `meshfree_kan_rkpm` 相关实验，从“若干脚本与输出目录”整理成一条更清晰的研究证据链，回答：

1. 现有实验各自在证明什么？
2. 合起来已经能支撑哪些主张？
3. 当前最关键的缺口是什么？
4. 下一步最应该补的是结果整理、对照重构，还是新实验？

本草案基于当前仓库中的实验代码结构撰写，尚未结合远端 `output/` 实际数值结果逐项审阅，因此以下判断属于“设计与证据逻辑层面的审阅”，不是最终结果解读。

---

## 一、当前总体证据链（工作版本）

当前实验主线实际上已经形成了一个较清楚的层次：

### 层 1：shape-function 结构层
回答：
- 学到的函数是否满足 PU？
- 是否具备一阶重构能力？
- 是否在边界和稀疏/危险区域保持稳定？
- 某些稳定机制（Softplus / Teacher / fallback / regularization）是否必要？

对应实验：
- `meshfree_kan_rkpm_1d_validation/*`
- `meshfree_kan_rkpm_2d_validation/patch_test/`
- `meshfree_kan_rkpm_2d_validation/stability_ablation/`

### 层 2：PDE 可用性层
回答：
- 这些 learned shape functions 作为 trial space 时，是否可用于变分求解？
- 其解误差与收敛行为是否合理？
- 结构正确性是否与 PDE 精度有联系？

对应实验：
- `meshfree_kan_rkpm_2d_validation/poisson_convergence/`

### 层 3：鲁棒性层
回答：
- 方法是否只在规则节点上有效？
- 节点扰动后，结构指标与 PDE 精度如何变化？

对应实验：
- `meshfree_kan_rkpm_2d_validation/irregular_nodes/`

### 层 4：迁移/更强泛化层（当前尚弱）
回答：
- 学到的 shape-function construction 是否能跨几何、跨节点集、跨问题迁移？
- teacher / offline pretraining 是否仅是辅助稳定训练，还是具有真正 transfer 含义？

当前状态：
- 有雏形（teacher、distillation、结构先行）
- 但尚未形成明确的迁移性实验链

---

## 二、逐实验审阅

---

## 1. 1D validation：代表性目标是“teacher-guided representational validation”

### 代表脚本
- `examples/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v5.py`

### 该实验在证明什么
这个实验不是 PDE 主实验，而是更基础的一层：

> 在 1D 设置下，当前参数化的 learned shape functions，是否能够逼近 analytical RKPM target？

### 当前已包含的关键设计
- 直接使用 analytical RKPM 作为 teacher target
- 训练目标以 `teacher loss` 为主
- 加入边界 anchor loss
- 加入 raw output regularization 抑制 cancellation / spike
- 将 `linear reproduction` 与 `PU` 作为诊断量输出

### 该实验的证据价值
它能很好地支持以下主张：
- 当前参数化器（当前为 KAN-like）具备逼近 RKPM shape family 的能力
- teacher distillation 在 shape-function 层面是可行的
- 边界锚定与温和正则化能改善形函数质量

### 它不能单独支持的主张
它不能单独证明：
- learned 方法优于 RKPM
- learned 方法天然具备 consistency，而非借 teacher 间接获得
- learned shape functions 一定能导向更好的 PDE 近似

### 当前定位建议
将它定位为：
- **表示能力 / teacher-guided shape-function validation**
- 而不是主要 PDE benchmark

---

## 2. 2D patch test：最核心的结构正确性实验

### 代表脚本
- `examples/meshfree_kan_rkpm_2d_validation/patch_test/deepritz_meshfree_kan_rkpm_2d_patch_test_v1.py`

### 当前输出指标
- `pu_rmse`
- `pu_linf`
- `linear_x_rmse`
- `linear_y_rmse`
- `lambda_h_max`
- `lambda_h_mean`
- PU residual heatmap
- linear reproduction heatmap
- `Lambda_h` heatmap

### 该实验在证明什么
这是当前最关键的基础实验，回答：

> 在 2D 中，learned shape functions 是否满足最基本的 meshfree shape-function structure？

具体来说，它试图证明：
- partition of unity 是否成立
- 一阶重构是否成立
- 形函数的放大/振荡是否受控（通过 `Lambda_h` 近似观测）

### 证据价值
如果结果足够好，这个实验能直接支撑：
- 当前方法不是普通 black-box neural ansatz
- 它确实在学习“具有 meshfree shape-function 意义”的函数族
- consistency 不是口头目标，而是在结构指标上被检验

### 当前局限
它目前仍然不是一个完全“纯诊断”的实验，因为不同 variant 的训练目标差异较大：
- 有的 variant 直接优化 `linear + pu + bd`
- 有的 variant 优化 `teacher + bd + reg`

所以 patch test 当前一方面在测结构结果，另一方面也在混合比较训练策略。

### 当前定位建议
将它定位为：
- **结构正确性核心实验**
- 是整个证据链中的第一支柱

---

## 3. stability ablation：不是普通消融，而是“trial-space stability mechanism study”

### 代表脚本
- `examples/meshfree_kan_rkpm_2d_validation/stability_ablation/deepritz_meshfree_kan_rkpm_2d_stability_ablation_v1.py`

### 当前关心的变体
从 `VARIANT_PRESETS` 看，当前至少涉及：
- `softplus_raw_pu_bd`
- `no_softplus_raw_pu_bd`
- `no_softplus_teacher`
- `no_softplus_teacher_reg`
- `no_softplus_raw_pu_bd_no_fallback`

### 该实验在证明什么
这个实验的真正问题不是“哪个 loss 更小”，而是：

> 哪些数值机制是在维持 learned shape functions 的结构稳定性？

具体包括：
- Softplus 是否有助于防止原始输出发生强 cancellation / sign oscillation
- teacher 是否能作为结构先验稳定训练
- fallback 是否是 orphan 区域的必要 safeguard
- regularization 是否能抑制 raw output 爆炸

### 证据价值
这是一个很有研究味道的实验，因为它不再只是看 PDE 误差，而是直接关注：
- learned trial space 能否保持数值可接受性
- 哪些机制是必要的结构 stabilizer

### 当前主要问题
当前 variant 设计存在**可归因性不足**：
- 某些 variant 同时改变了 softplus、teacher、linear loss、pu loss、regularization 等多个因素
- 因此，如果结果有差异，归因会变得不够干净

例如：
- `no_softplus_teacher` 不仅引入了 teacher，也同时关闭了 `linear_loss` 和 `pu_loss`
- 这会导致“teacher 是否有效”的结论不够纯粹

### 当前定位建议
保留它作为：
- **结构稳定性研究实验**

但后续应将其重构为更干净的 factorized ablation matrix。

---

## 4. 2D Poisson convergence：当前最接近“方法主实验”的部分

### 代表脚本
- `examples/meshfree_kan_rkpm_2d_validation/poisson_convergence/deepritz_meshfree_kan_rkpm_2d_poisson_convergence_v1.py`

### 当前输出指标
- patch metrics（结构层）
- `l2_error`
- `h1_semi_error`
- `boundary_l2`
- 多 `n_side` 的收敛图
- Phase A / Phase B 训练曲线

### 该实验在证明什么
它回答的是：

> learned shape functions 不仅结构上成立，而且作为 trial space 放入变分 Poisson 求解以后，能否给出合理的误差与收敛行为？

这实际上是在连接：
- shape-function structure
- variational approximation quality

### 证据价值
如果结果稳定且趋势良好，这个实验可以支撑：
- 当前方法不仅是在“学形函数”，而且这些形函数对 PDE 解是可用的
- patch-level 结构约束与 PDE-level 结果之间确实存在联系
- 该方法可以被视作一种真正的 learned meshfree trial-space construction，而不是纯可视化实验

### 当前局限
目前还需要结合实际结果确认：
- 误差收敛是否稳定
- 结构指标与 PDE 指标是否正相关
- 不同 variants 是否在 PDE 层也体现出与结构层一致的排序

### 当前定位建议
将它定位为：
- **当前论文主实验雏形**
- 是第二支柱（结构 → PDE 可用性）

---

## 5. irregular_nodes：鲁棒性实验已经有雏形

### 代表脚本
- `examples/meshfree_kan_rkpm_2d_validation/irregular_nodes/deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py`

### 当前输出指标
- jitter vs `L2 error`
- jitter vs `Lambda_h`
- patch metrics + solution metrics

### 该实验在证明什么
它回答的是：

> learned shape functions 是否只在规则网格/规则节点上成立，还是在节点扰动下仍然保持结构与 PDE 可用性？

### 证据价值
这个实验对你后续的“鲁棒性叙事”非常重要，因为它说明：
- 该方法不是纯规则格点上的巧合
- 结构约束与数值稳定性是否能跨节点布置保留

### 当前局限
目前它更像是鲁棒性初探，而不是更强的泛化/迁移实验。
它主要说明：
- node jitter robustness
而不是：
- cross-geometry transfer
- cross-problem transfer
- offline-online transfer reuse

### 当前定位建议
将它定位为：
- **鲁棒性实验雏形**
- 是第三支柱（不只在规则节点上成立）

---

## 三、合起来当前已经能支撑哪些主张

如果实际结果正常，并与代码意图一致，那么现有实验设计已经有潜力支撑以下主张：

### 主张 A：当前方法学习的是“具有结构意义的 shape functions”，而不是普通黑箱解表示
支撑证据：
- 1D teacher-guided RKPM shape matching
- 2D patch test (`PU`, linear reproduction, `Lambda_h`)

### 主张 B：某些数值机制对 learned shape-function stability 是必要的
支撑证据：
- stability ablation
- `Lambda_h` 与结构指标比较
- fallback / Softplus / teacher / regularization 的对照

### 主张 C：结构正确的 learned shape functions 可以作为变分 PDE 的 trial space 使用
支撑证据：
- 2D Poisson convergence
- patch metrics + PDE metrics 的联合记录

### 主张 D：该方法对节点扰动具有一定鲁棒性
支撑证据：
- irregular nodes experiment

---

## 四、当前还不能强说的主张

即使现有代码设计不错，以下主张目前仍不能在没有补强前提下强说：

### 1. “KAN 是不可替代的关键贡献”
原因：
- 当前定位本身就不应把贡献绑死在 KAN 上
- 现有实验虽有结构机制比较，但并未形成“KAN vs other parameterizer”的干净比较

### 2. “teacher 明确优于 consistency-style structure losses”
原因：
- 当前 teacher 变体与 raw/pu/linear 变体耦合过多
- 尚缺 clean comparison: `raw_pu_bd` vs `raw_pu_bd + teacher`

### 3. “方法已经具备明确迁移性”
原因：
- 当前 teacher/distillation 更像结构先验与训练稳定化
- 还没有真正意义上的 transfer experiment 链

### 4. “方法已经优于 classical RKPM / meshfree baseline”
原因：
- 当前 1D teacher experiment 是 shape-function matching，不是 superiority study
- 2D 主要对自身结构与变体做验证，还未形成与 classical method 的完整系统对照

---

## 五、当前最关键的证据缺口

### 缺口 1：结果层面尚未统一整理为“主结论表”
目前代码会输出：
- `metrics.json`
- `summary.txt`
- `curves.npz`
- figures

但研究层面还缺一份统一总表，例如：

| 实验 | 回答问题 | 主指标 | 当前最佳 variant | 支持的结论 | 还缺什么 |
|---|---|---|---|---|---|

这个总表非常关键，因为它能把“很多脚本”提升成“可复用科研证据”。

### 缺口 2：ablation matrix 还不够 clean
当前最明显的问题是：
- 一些 variant 同时改了多个因素
- 导致结论不够可归因

后续应至少拆清：
- Softplus on/off
- teacher on/off
- reg on/off
- fallback on/off
- 是否保留 linear/pu/bd losses

### 缺口 3：结构层与 PDE 层之间，还需要更明确的“因果解释”
当前代码已经同时记录：
- patch metrics
- PDE metrics

但后续需要明确回答：
- `pu_rmse` 降低是否真的对应 `L2/H1` 改善？
- `lambda_h_max` 升高是否对应 instability 或 error 上升？
- 哪个结构指标对 PDE 误差最敏感？

### 缺口 4：迁移性叙事还停留在雏形
当前更接近：
- distillation / structure prior
而不是：
- transfer learning evidence

---

## 六、下一步最合理的研究动作（基于当前已有资产）

### 动作 A：先整理现有结果，而不是立刻新增大量实验
优先建议：
1. 汇总远端 `output/` 中已有 `metrics.json / summary.txt`
2. 生成一份总表
3. 明确每个实验目前支持到什么程度

### 动作 B：重构最关键的 ablation 设计
优先补最关键的一类 clean comparison：
- `raw_pu_bd`
- `raw_pu_bd + teacher`
- `raw_pu_bd + teacher + reg`
- `raw_pu_bd` with / without fallback
- `softplus` on / off with everything else fixed

### 动作 C：在结果整理后，再决定是否补 classical baseline
这一步现在不应盲目开做，而应先看当前结果是否已经足够形成主张。

### 动作 D：暂不把迁移性写得过强
当前应更稳地写成：
- teacher / distillation 提供 structure prior 或 stabilizing guidance

而不是直接写成：
- strong transfer learning claim

---

## 七、当前一句话总结

当前 `meshfree_kan_rkpm` 实验体系已经具备较清楚的三层证据结构：

> 先在 1D/2D 上验证 learned shape functions 的结构正确性与稳定性，再在 2D Poisson 中验证其作为 trial space 的数值可用性，并通过 irregular nodes 初步检验鲁棒性。

当前最需要补的不是全新实验类型，而是：

> **把已有实验结果整理成统一结论，并将稳定性对照矩阵重构得更可归因。**
