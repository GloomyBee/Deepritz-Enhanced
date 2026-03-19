# 第一部分实验规划：形函数特性验证

## 1. 目标与边界

第一部分实验只回答一个核心问题：

> 共享 KAN 参数化得到的 meshfree shape functions，是否具备后续作为无网格 trial space 使用所需的基本结构性质与几何鲁棒性？

这一部分**不以 PDE 最终误差为主目标**，PDE 求解结果只作为辅助参考。主线应放在：

- 形函数是否可表达
- 一致性 / 积分约束是否成立或近似成立
- 节点分布变化后结构性质是否退化
- 大变形 / 重拓扑后重新计算形函数时，结构性质是否还能保留

与此对应，后续“trial space 数值价值”相关实验，应归到第二部分，而不是混在本部分里。

## 2. 当前状态

当前仓库已经完成或基本具备的内容：

- `1D` / `2D` 形函数可表达性验证已经跑过
- `patch_test` 已验证：
  - partition of unity 残差
  - 线性重构残差
  - `Lambda_h`
- `irregular_nodes` 已有 2D 雏形：
  - 基于 `jitter` 扰动节点
  - 输出 `L2 error` 与 `Lambda_h`
- `trial_space_value` 已单独形成第二部分风格：
  - 固定 learned basis 后，比较 classical / frozen-`w` / joint 的数值求解效果

因此，第一部分实验不应继续向 `trial_space_value` 聚焦，而应沿着 `meshfree_kan_rkpm_2d_validation` 这条结构验证主线扩展。

## 3. 代码组织原则

后续第一部分实验统一参考：

- `examples/meshfree_kan_rkpm_2d_validation/common.py`
- 每个子实验目录下只有“参数解析 + sweep 逻辑 + 本实验专属汇总图”
- 共享逻辑全部放 `common.py`

### 3.1 `common.py` 负责内容

- 节点生成与节点几何变换
- `MeshfreeKAN2D` 模型定义
- Phase A 训练逻辑
- 公共结构指标计算
- 公共绘图函数
- 标准输出目录和落盘函数

### 3.2 具体实验脚本负责内容

- 参数解析
- 单一实验主题的 sweep
- 汇总图绘制
- 指标汇总与批量输出

### 3.3 标准产物

每个 case 统一输出：

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/*.png`

每个 group 根目录统一输出：

- `*_summary.json`
- `*_summary.txt`
- `*_summary.png`

## 4. 第一部分实验建议结构

建议继续沿用现有目录，不另开新的顶层主线：

```text
examples/meshfree_kan_rkpm_2d_validation/
├── common.py
├── README.md
├── patch_test/
├── consistency_constraints/
├── irregular_nodes/
└── large_deformation_retopology/
```

其中：

- `patch_test/`：保留为最基础的点值一致性与线性重构验证
- `consistency_constraints/`：新增，专门做积分约束 / 一致性实验
- `irregular_nodes/`：保留并扩展，负责非均布节点实验
- `large_deformation_retopology/`：新增，负责大变形 / 重拓扑后形函数重新计算实验

## 5. 第一部分实验的统一指标

建议把指标分成四类。

### 5.1 点值结构指标

- `pu_max_abs`
- `pu_rmse`
- `linear_x_max_abs`
- `linear_y_max_abs`
- `linear_x_rmse`
- `linear_y_rmse`
- `lambda_h_max`
- `lambda_h_mean`

这类指标当前 `patch_test` 已基本覆盖。

### 5.2 覆盖与稳定性指标

- `orphan_ratio`
- `phi_sum_min`
- `phi_sum_p01`
- `support_coverage_min`
- `support_coverage_mean`

说明：

- 这类指标用于描述变形、非均布或重拓扑后，节点覆盖是否断裂
- 当前代码里已有 `phi_sum` / fallback 机制的实现基础，但需要整理成正式指标输出

### 5.3 积分一致性指标

这类指标是第一部分尚未独立成组的关键内容。

建议统一记录：

- `int_pu_residual`
- `int_x_repro_residual`
- `int_y_repro_residual`
- `mass_sum_residual`
- `moment_x_residual`
- `moment_y_residual`

建议定义方式：

- `int_pu_residual = | ∫(Σ_i φ_i - 1) dΩ |`
- `int_x_repro_residual = | ∫(Σ_i φ_i x_i - x) dΩ |`
- `int_y_repro_residual = | ∫(Σ_i φ_i y_i - y) dΩ |`
- `mass_sum_residual = | Σ_i ∫φ_i dΩ - |Ω| |`
- `moment_x_residual = | Σ_i x_i ∫φ_i dΩ - ∫x dΩ |`
- `moment_y_residual = | Σ_i y_i ∫φ_i dΩ - ∫y dΩ |`

备注：

- 这里优先使用“全局积分一致性”写法，便于与后续理论表述对齐
- 若后续理论要求的是更细的 moment 条件，可在此基础上扩展，不必推翻目录结构

### 5.4 变形迁移指标

用于第三类实验：

- `transfer_pu_max_abs`
- `transfer_linear_rmse`
- `transfer_lambda_h_max`
- `transfer_orphan_ratio`
- `transfer_support_gap_count`

含义：

- 先在参考节点构型上训练或固定 learned mechanism
- 再在变形后 / 重拓扑后的节点构型上直接重新计算形函数
- 衡量结构性质保留了多少

## 6. 子实验规划

### 6.1 `patch_test/`

作用：

- 作为第一部分实验的基础入口
- 不承担所有结构验证，只负责最直接的点值型诊断

当前内容：

- `PU residual`
- `Linear reproduction x/y`
- `Lambda_h heatmap`

后续建议：

- 保持为最轻量、最快速的 sanity check
- 不在这里堆太多积分约束逻辑

### 6.2 `consistency_constraints/`

这是第一部分里最应该新增的实验组。

目标：

- 用统一积分规则验证 learned shape functions 是否满足全局一致性约束
- 把“理论上的 consistency”转成可直接引用的实验指标

建议脚本：

```text
examples/meshfree_kan_rkpm_2d_validation/consistency_constraints/
└── deepritz_meshfree_kan_rkpm_2d_consistency_constraints_v1.py
```

输入 sweep 建议：

- `n_side`
- `kappa`
- `variant`
- `quadrature_order`
- `grid_resolution`

核心流程建议：

1. 跑 Phase A，得到 learned shape generator
2. 在高精度 domain quadrature 点上评估 `phi`
3. 计算全局积分一致性指标
4. 生成：
   - consistency summary
   - residual bar chart
   - 必要时的 residual field heatmap

输出重点：

- `metrics.json` 中必须明确积分残差
- `figures/` 中至少包含一张 consistency 汇总图

### 6.3 `irregular_nodes/`

这是第一部分里已经有基础、但还需要扩展的问题组。

目标：

- 检查节点非均布后，形函数结构性质如何退化

当前版本：

- 主要是 `jitter` 扰动

建议不要只停留在“随机抖动”。

建议拆成两个层次：

1. `jitter sweep`
- 轻量、可快速批量扫描
- 用于建立与现有结果连续的基线

2. `distribution shift`
- 用于更接近真实非均布场景
- 可选类型：
  - 局部稀疏 / 局部加密
  - 密度梯度
  - 中心聚集 / 边界聚集

建议脚本演进：

```text
deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py
deepritz_meshfree_kan_rkpm_2d_irregular_nodes_density_v1.py
```

统一关注指标：

- `lambda_h`
- `PU`
- 线性重构误差
- orphan / coverage 指标

若需要附带 PDE 误差，可以保留，但放在次要位置。

### 6.4 `large_deformation_retopology/`

这是第一部分里最容易被误解、但也是后续理论最关键的一组。

本组**不是训练节点怎么移动**，而是：

> 为了验证大变形或重拓扑场景下，给定新的节点构型后，这套 learned mechanism 是否还能重新计算出可用的形函数。

建议脚本：

```text
examples/meshfree_kan_rkpm_2d_validation/large_deformation_retopology/
└── deepritz_meshfree_kan_rkpm_2d_large_deformation_retopology_v1.py
```

建议分三类构型变化：

1. **大位移但拓扑未变**
- 仿射拉伸
- 剪切
- 非线性平滑形变

2. **邻域关系显著变化**
- 大幅随机位移
- 局部压缩 / 拉伸

3. **重拓扑**
- 删除部分节点后重采样
- 局部加密 / 局部稀疏
- 重新生成节点云但保持同一区域

建议比较模式：

1. `reference`
- 原始节点构型上计算形函数

2. `transfer`
- 不重新训练 KAN，只替换节点构型并重新计算形函数

3. `optional_refresh`
- 在变形后构型上做少量 Phase A refresh
- 这个模式不是第一优先级，但为后续论文可以预留

第一版最重要的是 `reference vs transfer`。

本组核心问题不是最终 PDE 精度，而是：

- 变形后 `phi` 是否还能计算
- `PU` / 一致性 / `Lambda_h` 是否还能维持
- 是否出现明显 coverage failure 或 orphan collapse

## 7. 与第二部分实验的边界

为了避免实验目的混淆，建议明确：

### 第一部分：形函数特性验证

- `patch_test`
- `consistency_constraints`
- `irregular_nodes`
- `large_deformation_retopology`

重点是结构正确性、几何鲁棒性、变形迁移性。

### 第二部分：trial space 数值价值验证

- `trial_space_value/poisson_compare`
- 后续可能扩展的 PDE solve comparisons

重点是 learned basis 作为 trial space 是否“数值上值得用”。

## 8. 实现顺序建议

建议按下面顺序推进：

1. `consistency_constraints`
- 先把积分一致性补齐，形成第一部分最缺的证据

2. 扩展 `irregular_nodes`
- 先保留现有 jitter 结果，再加一种更明确的非均布分布

3. `large_deformation_retopology`
- 先做 `reference vs transfer`
- 再看是否需要 `optional_refresh`

## 9. 近期落地建议

如果下一轮开始写代码，建议按以下拆分执行：

1. 在 `common.py` 增加统一结构指标与积分一致性计算函数
2. 新增 `consistency_constraints/` 入口脚本
3. 复用现有 `irregular_nodes/`，补更明确的非均布构型生成
4. 新增 `large_deformation_retopology/`，先只做 `transfer` 模式

这样推进，既不破坏现有目录，也能把第一部分实验闭环补完整。
