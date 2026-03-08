# presentation_mtheme

这是基于 `paper/mtheme-master` 风格搭建的 beamer 汇报骨架，主题是：

- `1D 机制验证`
- `2D patch test`
- `2D Poisson 主实验`
- `稳定性消融`
- `不规则节点鲁棒性`

## 文件说明

- `main.tex`：主幻灯片源文件
- `../presentation_mtheme_plan.md`：逐页策划与素材映射文档
- `figures/`：后续若需要复制/裁剪图片，可放在这里

## 编译建议

优先使用 `XeLaTeX`，因为 `main.tex` 使用了中文排版：

```powershell
cd "paper/presentation_mtheme"
xelatex main.tex
xelatex main.tex
```

如果本机 TeX 环境尚未安装 `metropolis` 主题，可以考虑两种方式：

1. 在本地 TeX 发行版中安装 `metropolis`
2. 或者先把 `mtheme-master` 生成/安装到本地 TeX 搜索路径，再编译

## 当前状态

当前版本已经完成：

- 章节结构
- 18 页左右的页面骨架
- 直接引用 `output/` 中现有 1D/2D 图片
- 关键结论与代表性指标的初步填充

当前版本还可以继续增强：

- 替换为更精炼的英文/中英混排标题
- 把表格进一步美化为投稿风格
- 加入方法总图、自绘示意图与 appendix 页
- 视需要复制图片到 `figures/` 并统一裁剪

## 推荐下一步

1. 先尝试编译 `main.tex`
2. 检查哪些图片尺寸在 beamer 中显示过大或过小
3. 再做第二轮版式微调与文案压缩
