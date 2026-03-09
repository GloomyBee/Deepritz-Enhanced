# Deep-RKPM-KAN Claim-Evidence Matrix (v0.1)

## One-Sentence Contribution

Deep-RKPM-KAN 通过共享 KAN 母核与 Phase-A 解耦约束，在免显式矩量矩阵逐点求逆的前提下，构建可解释且稳定的无网格 Deep Ritz 离散空间。

## Claim -> Evidence Mapping

| Claim ID | Claim | Evidence (Metric) | Value | Source |
|---|---|---|---|---|
| C1 | 去除正性先验会导致误差灾难性放大 | v4 global L_inf | 19.2948 | `test2/output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v4_output_no_softplus_v4/metrics.json` |
| C2 | Teacher 诊断可将 no-softplus 设定拉回稳定区间 | v5 global L_inf | 0.2518 | `test2/output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v5_output_teacher_distill_v5/metrics.json` |
| C3 | 相比 v4，v5 在核心误差指标显著恢复 | v4/v5 global L_2 | 3.4350 -> 0.0491 | same as above |
| C4 | v3/v5 保持稳定，而 v4 出现抵消放大 | Lambda_h | v3=1.000, v4=67.139, v5=1.000 | 由 `curves.npz` 统计结果写入论文表格 |
| C5 | 几何一致性与稳定性存在结构性权衡 | v3 linear reproduction RMSE | 0.0147 | `test2/output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v3_output_bd_anchor_v3/metrics.json` |

## Submission Risk Log

- R1: KAN 相关引用仍为占位符（`placeholder_kan2024_verify`），投稿前必须替换为程序化核验后的 BibTeX。
- R2: 当前版本以 1D 机制闭环为主，2D 收敛率与误差项分离尚未完成。
- R3: 目前结果重点是机制解释，SOTA 性能对比仍需补充。

## Pre-Submission Checklist (Current)

- [x] 一句话贡献已锁定
- [x] 主张与定量证据已对齐
- [x] 文内引用与参考文献链路已打通
- [ ] 所有引用完成 DOI/来源核验
- [ ] 2D 实验与误差分解实证补齐
- [ ] 审稿人视角自检（质量/清晰度/显著性/原创性）
