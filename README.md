# Conformal-RVFL

在三头 edRVFL 上比较三种构造**类别预测集**（共形预测）的方法。网络结构与输出权重一次解出的方式都不变，区别只在“何时合并三个头”和“用什么校准”。

## 方法

| 名称 | 分数 | 校准 | 输出 |
|---|---|---|---|
| **aggregate**（本文） | 三头分数平均，$s=\sum_j w_j(1-p_j)$ | 训练折内 10 折 CV+，一次 | 一个集合 |
| **per-head union**（消融） | 每个头各自 $s_j=1-p_j$ | 同样的 CV+，每头一次 | 三个集合取并 |
| **split**（基线） | 同 aggregate | 训练折内留出 25% | 一个集合 |

aggregate 与 union 共用同一批折模型和校准点，分类准确率完全相同；比较的是“先聚合再校准”还是“先校准再合并”。split 是校准协议的对照，注意它的模型只见到 75% 训练折（CV+ 为 90%）。

## 25 个任务

唯一来源是 `src/data_loader.py` 的 `MULTICLASS_TASKS`，只含多分类 UCI 任务，样本量 47–67,557，类别数 3–10：

Abalone(1)、Annealing(3)、Cardiotocography_10classes(193)、Car Evaluation(19)、Connect-4(26)、Dermatology(33)、Ecoli(39)、
Energy Efficiency(242)、Glass Identification(42)、Hayes-Roth(44)、Heart_cleveland(45)、lymphography(63)、Molecular Biology(69)、
Nursery(76)、Optical Recognition of Handwritten Digits(80)、Soybean(91)、Statlog (Shuttle)(148)、Statlog (Landsat Satellite)(146)、
Statlog (Vehicle Silhouettes)(149)、Vertebral Column(212)、Waveform(107)、Yeast(110)、Wine(109)、Wine Quality(186)、Zoo(111)。

标签口径：Abalone 按环数合并为 3 类（≤8、9–10、≥11）；Energy Efficiency 取 Y1 按分位数分 5 类；Heart_cleveland 用原生五类；Cardiotocography 用 10 类的 `CLASS` 列。

## 协议

- 外层 5 折 `StratifiedKFold`（shuffle，`random_state=42`）。
- 5 个固定种子：1618、3979、6187、7333、8443。
- $\alpha \in \{0.05, 0.10, 0.15, 0.20, 0.30\}$。
- 模型：3 层，每层 50 个隐藏节点，$\lambda=0.1$，sigmoid，三头均匀加权。
- 内层 10 折 CV+ 全部在训练折内；缺失值填充与标准化只在训练折上拟合（`fold_preprocess`）。
- 主结局：把覆盖率插值到 0.90 处的平均集合大小；任务层 Wilcoxon 符号秩检验、bootstrap 95% CI、两个对比的 Holm 校正。

## 运行

在仓库根目录执行：

```bash
python src/analyze.py                 # 只汇总已有的 results/conformal.csv，不跑实验
python src/experiment.py --help
python src/experiment.py              # 重跑全部 25 个任务 × 5 个种子（需要 GPU）
python src/experiment.py --tasks Wine --out /tmp/wine.csv --seeds 1618
python src/main.py                    # 交互式入口；选 3 只做汇总
```

⚠️ `experiment.py` 每次启动都会先清空 `results/`。

依赖：numpy、scipy、scikit-learn、pandas、torch；`ucimlrepo` 可选，仅在缓存缺失需要下载时使用。

数据缓存位于 `data/dataset_cache/uci_<id>.pkl`（25 个文件），加载器按仓库内的相对路径查找，与当前工作目录无关。

## 结果

- `results/conformal.csv`：9,375 行 = 25 任务 × 5 种子 × 5 折 × 3 方法 × 5 个 α，0 错误行。
- `results/analysis.txt`：`analyze.py` 的输出。
- 匹配 90% 覆盖时：aggregate 比 union 小 0.023（1.6%，22/1/2，Holm $p=3.2\times10^{-4}$）；比 split 小 0.046（2.7%，23/0/2，Holm $p=1.7\times10^{-6}$）。
- $\alpha=0.10$ 时覆盖率 0.906（aggregate）/ 0.940（union）/ 0.905（split）；aggregate 与 union 平均准确率相同（0.831）。

## 文件

| 内容 | 路径 |
|---|---|
| edRVFL + CV+ 实现 | `src/models/RVFLC.py` |
| 数据加载、预处理、25 题清单 | `src/data_loader.py` |
| 实验驱动（三种方法） | `src/experiment.py` |
| 结果分析 | `src/analyze.py` |
| 交互式入口 | `src/main.py` |
| 结果 | `results/conformal.csv`、`results/analysis.txt` |
| 数据缓存 | `data/dataset_cache/` |
