# shape_validation one_d

1D 形函数验证主线目录，输出写入 `output/shape_validation/one_d/`。

当前活跃实验只保留两类节点分布：

- `uniform_nodes/`
- `nonuniform_nodes/`

根目录共享层按职责拆分：

- `common.py`：路径、artifact、轻量工具与共享样式常量
- `basis.py`：1D 节点、RKPM 与形函数评估基础
- `training.py`：训练 variant 与训练循环
- `plotting.py`：case 级主图与 diagnostics
- `summary_plotting.py`：group 级汇总图

旧的稳定性消融入口已经迁移到 `archive/shape_validation/one_d/`，默认不再作为主视图入口扩展。
