# Meshfree KAN-RKPM 汇报 PPT 规划（mtheme）

## 1. 目标与定位

- 模板：`paper/mtheme-master`
- 建议成品：一套 `16--18` 页的学术汇报型 beamer 幻灯片
- 目标听众：组会 / 中期汇报 / 论文投稿前内部汇报
- 主线：`1D 机制验证 -> 2D 形函数一致性 -> 2D 主实验收敛 -> 稳定性与鲁棒性`
- 设计原则：`少字、多图、每页一个明确结论`

## 2. 核心叙事

### 2.1 一句话主结论

> Meshfree KAN-RKPM 在 1D 中建立了可解释的形函数学习机制，在 2D 中进一步实现了稳定的 PU / 一次重构与有竞争力的 Poisson 误差表现；teacher-guided 变体在稳定性与精度之间给出了当前最优折中。

### 2.2 页面组织逻辑

1. 为什么需要学习型 meshfree shape function
2. 1D 先验证哪些机制是必要的
3. 2D patch test 说明 shape function 基础性质成立
4. 2D Poisson 主实验体现精度与收敛
5. 稳定性消融说明 teacher / fallback 的价值
6. 不规则节点说明鲁棒性

## 3. 逐页大纲（建议 18 页）

### Slide 1 标题页
- 标题建议：`From 1D Validation to 2D Poisson: Meshfree KAN-RKPM`
- 副标题：`Shape-function learning, stability, and convergence`
- 内容：作者、单位、日期
- 素材：无需实验图

### Slide 2 目录页
- 四节：`Background` / `1D Validation` / `2D Results` / `Conclusion`
- 使用 `mtheme` 自带 TOC 即可

### Slide 3 问题背景与目标
- 目标：说明我们不是单纯训练 PDE 解，而是在学习具有 meshfree/RKPM 结构的 shape function
- 要点：`PU`、`linear reproduction`、`boundary anchoring`、`teacher distillation`
- 形式：左侧公式或流程图，右侧三条 bullet
- 素材：后续可从论文公式整理，不依赖 output 图

### Slide 4 方法框架图
- 内容：`nodes -> local KAN basis -> windowing -> normalization/fallback -> shape functions -> Deep Ritz solver`
- 目标：建立后续所有实验共享的“总图”
- 形式：自绘流程图
- 素材来源：`examples/meshfree_kan_rkpm_2d_validation/common.py`

### Slide 5 为什么先做 1D
- 核心信息：1D 更容易直接观察 shape function、边界行为和不稳定现象
- 结论句：`1D is the microscope for mechanism validation.`
- 素材：1D 三组版本名与一句解释

### Slide 6 1D baseline：基本 shape function 行为
- 目标：展示基线版本已经具备较好的 PU 与线性重构
- 推荐图片：
  - `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_output/shape_compare.png`
  - `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_output/shape_subset_overlay.png`
- 推荐指标：
  - `global_l2 = 2.26e-2`
  - `pu_max_error = 1.47e-12`
  - `linear_reproduction_rmse = 2.56e-3`
- 数据来源：`output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_output/metrics.json`

### Slide 7 1D failure case：去掉 softplus 后的失稳
- 目标：展示“约束/结构不是可有可无”
- 推荐图片：
  - `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v4_output_no_softplus_v4/shape_compare.png`
  - `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v4_output_no_softplus_v4/shape_subset_overlay.png`
- 推荐指标：
  - `global_l2 = 3.435`
  - `global_linf = 19.295`
  - `pu_raw_sum_rmse = 1.033`
- 数据来源：`output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v4_output_no_softplus_v4/metrics.json`
- 结论句：`Removing the stabilizing mechanism can catastrophically destroy 1D shape quality.`

### Slide 8 1D teacher 版本：结构恢复与改进
- 目标：说明 teacher distillation 提供了更可控的 shape-function 学习路径
- 推荐图片：
  - `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v5_output_teacher_distill_v5/shape_compare.png`
  - `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v5_output_teacher_distill_v5/shape_subset_overlay.png`
- 推荐指标：
  - `global_l2 = 4.91e-2`
  - `linear_reproduction_rmse = 7.50e-3`
  - `phi0_at_x0 = 0.961`, `phiN_at_x1 = 0.954`
- 数据来源：`output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v5_output_teacher_distill_v5/metrics.json`
- 结论句：`Teacher guidance recovers structure and prepares the transition to 2D.`

### Slide 9 1D 小结：从机制到 2D
- 建议做成小结页
- 三条 bullet：
  - 1D 说明 shape-function learning 需要稳定化机制
  - teacher/fallback 是进入 2D 的关键线索
  - 后续 2D 关注的不只是精度，还有 `PU`、`Lambda_h` 与鲁棒性

### Slide 10 2D patch test：算例与目标
- 目标：说明 patch test 是 2D 线性重构一致性验证，而不是主 PDE 结果页
- 推荐图片：
  - `output/meshfree_kan_rkpm_2d_validation/patch_test/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/figures/pu_heatmap.png`
  - `output/meshfree_kan_rkpm_2d_validation/patch_test/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/figures/linear_x_heatmap.png`
  - `output/meshfree_kan_rkpm_2d_validation/patch_test/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/figures/lambda_h_heatmap.png`
- 数据来源：`output/meshfree_kan_rkpm_2d_validation/patch_test/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/summary.txt`
- 结论重点：`PU residual` 很小、`Lambda_h` 接近 1、一次重构误差可控

### Slide 11 2D patch test：结论解释
- 目标：把热图解释成“形函数合法性已经成立”
- 建议文字：
  - `sum phi_i ≈ 1`
  - `sum phi_i x_i ≈ x`
  - `sum phi_i y_i ≈ y`
- 这一页最好少图多解释，用于承接主实验

### Slide 12 2D Poisson 主实验设定
- 内容：
  - 问题：`sinusoidal_poisson_2d`
  - 节点加密：`n_side = 11, 15, 21`
  - 当前主变体：`no_softplus_teacher_reg`
  - 核心参数：`kappa=2.5, beta_bc=500, gamma_linear=1`
- 推荐附一行：主目标是让收敛图更像论文可投稿结果

### Slide 13 2D 主结果：收敛总览
- 推荐图片：`output/meshfree_kan_rkpm_2d_validation/poisson_convergence/convergence_summary.png`
- 推荐表格数据：
  - `n=11`: `L2 = 7.19e-3`, `H1 = 2.77e-1`
  - `n=15`: `L2 = 8.53e-3`, `H1 = 2.36e-1`
  - `n=21`: `L2 = 1.15e-2`, `H1 = 2.73e-1`
- 数据来源：
  - `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_no_softplus_teacher_reg_ns11_k2p5_seed42_paper_v3_main/metrics.json`
  - `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_no_softplus_teacher_reg_ns15_k2p5_seed42_paper_v3_main/metrics.json`
  - `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_no_softplus_teacher_reg_ns21_k2p5_seed42_paper_v3_main/metrics.json`
- 结论句：`The teacher-regularized 2D variant reaches a clearly better accuracy regime than the earlier baseline.`

### Slide 14 2D 主结果：旧版 vs 最优版
- 目标：直接体现改进幅度
- 建议做一个对比表：
  - `n=11`: `L2 2.58e-2 -> 7.19e-3`, `H1 5.58e-1 -> 2.77e-1`
  - `n=15`: `L2 3.02e-2 -> 8.53e-3`, `H1 5.40e-1 -> 2.36e-1`
  - `n=21`: `L2 2.73e-2 -> 1.15e-2`, `H1 6.18e-1 -> 2.73e-1`
- 对比来源：
  - baseline `paper_v1`：
    - `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_softplus_raw_pu_bd_ns11_k2p5_seed42_paper_v1/metrics.json`
    - `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/metrics.json`
    - `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_softplus_raw_pu_bd_ns21_k2p5_seed42_paper_v1/metrics.json`
  - 最优版 `paper_v3_main`：同上页
- 建议标题：`From baseline to paper-ready 2D configuration`

### Slide 15 2D 主结果：代表性解场
- 推荐图片：
  - `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_no_softplus_teacher_reg_ns15_k2p5_seed42_paper_v3_main/figures/solution_triplet.png`
  - 或 `ns21` 的 `solution_triplet.png`
- 目标：让观众直接看到真解/预测/误差场
- 辅助图片：`loss_curves.png`

### Slide 16 稳定性消融：为什么 teacher / regularization 值得保留
- 推荐图片：`output/meshfree_kan_rkpm_2d_validation/stability_ablation/stability_summary.png`
- 推荐核心数字：
  - `softplus_raw_pu_bd`: `lambda_h_max ≈ 1.00`
  - `no_softplus_raw_pu_bd`: `lambda_h_max ≈ 14.09`
  - `no_softplus_teacher`: `lambda_h_max ≈ 1.00`
  - `no_softplus_teacher_reg`: `lambda_h_max ≈ 1.01`
- 数据来源：`output/meshfree_kan_rkpm_2d_validation/stability_ablation/stability_summary.txt`
- 结论句：`Teacher guidance restores stability that is otherwise lost in raw unconstrained variants.`

### Slide 17 不规则节点鲁棒性
- 推荐图片：
  - `output/meshfree_kan_rkpm_2d_validation/irregular_nodes/irregular_summary.png`
  - `output/meshfree_kan_rkpm_2d_validation/irregular_nodes/variant_softplus_raw_pu_bd_ns15_k2p5_jit0p1_seed42_paper_v1/figures/solution_triplet.png`
- 推荐数字：
  - `jitter=0.0`: `L2 = 2.41e-2`, `H1 = 6.64e-1`
  - `jitter=0.1`: `L2 = 2.65e-2`, `H1 = 6.26e-1`
  - `jitter=0.2`: `L2 = 2.63e-2`, `H1 = 7.58e-1`
- 数据来源：`output/meshfree_kan_rkpm_2d_validation/irregular_nodes/irregular_summary.txt`
- 结论句：`The method remains usable under moderate node perturbation.`

### Slide 18 总结与下一步
- 建议三条总结：
  - 1D 明确了形函数学习的关键机制
  - 2D patch test + Poisson 结果说明方法可推广且有效
  - teacher-guided regularized variant 是当前最优投稿候选
- 下一步：更系统的收敛实验、更多 PDE、与 RKPM/FEM 更直接对比

## 4. 强烈建议保留的素材清单

### 4.1 必选图片
- `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_output/shape_compare.png`
- `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v4_output_no_softplus_v4/shape_compare.png`
- `output/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v5_output_teacher_distill_v5/shape_compare.png`
- `output/meshfree_kan_rkpm_2d_validation/patch_test/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/figures/pu_heatmap.png`
- `output/meshfree_kan_rkpm_2d_validation/patch_test/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/figures/lambda_h_heatmap.png`
- `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/convergence_summary.png`
- `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_no_softplus_teacher_reg_ns15_k2p5_seed42_paper_v3_main/figures/solution_triplet.png`
- `output/meshfree_kan_rkpm_2d_validation/stability_ablation/stability_summary.png`
- `output/meshfree_kan_rkpm_2d_validation/irregular_nodes/irregular_summary.png`

### 4.2 备选图片
- `output/meshfree_kan_rkpm_2d_validation/patch_test/variant_softplus_raw_pu_bd_ns15_k2p5_seed42_paper_v1/figures/linear_x_heatmap.png`
- `output/meshfree_kan_rkpm_2d_validation/poisson_convergence/variant_no_softplus_teacher_reg_ns15_k2p5_seed42_paper_v3_main/figures/loss_curves.png`
- `output/meshfree_kan_rkpm_2d_validation/irregular_nodes/variant_softplus_raw_pu_bd_ns15_k2p5_jit0p1_seed42_paper_v1/figures/loss_curves.png`

## 5. 文案风格建议

- 每页标题尽量写成结论句，而不是中性标题
- 每页 bullet 控制在 `3--4` 条
- 每页只保留 `1` 个主结论
- 所有数字统一保留 `2--3` 位有效数字
- 颜色建议：
  - baseline：灰色
  - best variant：蓝绿
  - failure/unstable：红色

## 6. mtheme 实施建议

### 6.1 目录建议
- 后续正式制作时新建：`paper/presentation_mtheme/`
- 推荐文件：
  - `paper/presentation_mtheme/main.tex`
  - `paper/presentation_mtheme/figures/`
  - `paper/presentation_mtheme/refs.bib`
  - `paper/presentation_mtheme/README.md`

### 6.2 技术建议
- 直接基于 `paper/mtheme-master/demo/demo.tex` 改造
- 优先用 `pdfLaTeX` 起步；若字体效果需要提升，再切 `XeLaTeX`
- 前期先不追求复杂动画，先把版式与结论锁住

## 7. 下一步执行顺序

1. 先搭 `main.tex` 骨架与 section 结构
2. 复制必选图片到 `paper/presentation_mtheme/figures/`
3. 先完成 `Slide 1--9` 的 1D 与 patch test 部分
4. 再完成 `Slide 10--18` 的 2D 主结果与总结
5. 最后补 appendix 页与答疑备用图
