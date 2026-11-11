# Conformal-RVFL

在三头 edRVFL 上构造**类别预测集**（共形预测），比较三种把多个读出头的概率变成一个集合的规则。网络结构与岭回归读出保持不变，区别只在“何时合并三个头”和“用什么校准”。

## 方法

| 名称 | 分数 | 校准 | 输出 |
|---|---|---|---|
| **aggregate**（本文） | 三头平均，$s=\sum_j w_j(1-p_j)$，$w_j=1/3$ | 训练折内 10 折 CV+，一次 | 一个集合 |
| **per-head union**（消融） | 每头各自 $s_j=1-p_j$ | 同样的 CV+，每头一次 | 三个集合取并 |
| **split**（基线） | 同 aggregate | 训练折内留出 25% | 一个集合 |

CV+ 的 p 值与本项目采用的集合为

$$
p(y)=\frac{1+\sum_{k=1}^{K}\#\{i\in S_k:\ s_i\ge s^{(k)}(x,y)\}}{n+1},
\qquad
\Gamma_\alpha(x)=\{y:\ p(y)>\alpha\},
$$

其中 $s^{(k)}(x,y)$ 是第 $k$ 个折模型对候选标签的分数。aggregate 与 union 共用同一批折模型和校准点，因此分类准确率完全相同；比较的是“先聚合再校准”还是“先校准再合并”。split 是校准协议对照，模型只见到 75% 训练折，CV+ 的每个折模型见到 90%。

## 25 个任务

唯一来源是 `src/data_loader.py` 的 `MULTICLASS_TASKS`，全部为多分类 UCI 任务，按样本量排序：

| ID | Task | $n$ | $K$ |
|---|---|---|---|
| 91 | Soybean | 47 | 4 |
| 111 | Zoo | 101 | 7 |
| 44 | Hayes-Roth | 132 | 3 |
| 63 | Lymphography | 148 | 4 |
| 109 | Wine | 178 | 3 |
| 42 | Glass Identification | 214 | 6 |
| 45 | Heart Cleveland | 303 | 5 |
| 212 | Vertebral Column | 310 | 3 |
| 39 | Ecoli | 336 | 8 |
| 33 | Dermatology | 366 | 6 |
| 242 | Energy Efficiency | 768 | 5 |
| 149 | Statlog (Vehicle Silhouettes) | 845 | 4 |
| 3 | Annealing | 898 | 5 |
| 110 | Yeast | 1,484 | 10 |
| 19 | Car Evaluation | 1,728 | 4 |
| 193 | Cardiotocography | 2,126 | 10 |
| 69 | Molecular Biology | 3,190 | 3 |
| 1 | Abalone | 4,177 | 3 |
| 107 | Waveform | 5,000 | 3 |
| 80 | Optical Recognition of Handwritten Digits | 5,620 | 10 |
| 146 | Statlog (Landsat Satellite) | 6,435 | 6 |
| 186 | Wine Quality | 6,497 | 7 |
| 76 | Nursery | 12,960 | 5 |
| 148 | Statlog (Shuttle) | 58,000 | 7 |
| 26 | Connect-4 | 67,557 | 3 |

标签口径：Abalone 按环数合并为 3 类（≤8、9–10、≥11）；Energy Efficiency 取 Y1 按分位数分 5 类；Heart Cleveland 用原生五类；Cardiotocography 用 10 类的 `CLASS` 列。

## 协议

- 外层 5 折 `StratifiedKFold`（shuffle，`random_state=42`），测试折不参与任何拟合或校准。
- 5 个固定种子：1618、3979、6187、7333、8443；每个（种子，外层折）组合抽一组隐藏权重，同折内的 10 个 CV+ 模型共享这一组。
- $\alpha \in \{0.05, 0.10, 0.15, 0.20, 0.30\}$。
- 模型：3 层，每层 50 个隐藏节点，$\lambda=0.1$，sigmoid，三头均匀加权。
- 内层 10 折 CV+ 全部在训练折内；缺失值用训练折均值填充，标准化按 `should_standardize` 决定（Connect-4、Heart Cleveland、Waveform、Zoo 不标准化），统计量只在训练折上拟合。
- 主结局：把覆盖率插值到 0.90 处的平均集合大小；任务层 Wilcoxon 符号秩检验、bootstrap 95% CI（10,000 次，按任务重采样）、两个对比的 Holm 校正。
- 测试点标签若未在训练部分出现，按未覆盖计。

## 运行

在仓库根目录执行：

```bash
python src/analyze.py                 # 只汇总已有的 results/conformal.csv，不跑实验
python src/experiment.py --help
python src/experiment.py              # 重跑全部 25 个任务 × 5 个种子（需要 GPU）
python src/experiment.py --tasks Wine --out /tmp/wine.csv --seeds 1618
python src/main.py                    # 交互式入口：主实验 / 指定任务 / 只汇总
```

⚠️ `experiment.py` 每次启动都会先清空 `results/`。

依赖：numpy、scipy、scikit-learn、pandas、torch；`ucimlrepo` 可选，仅在缓存缺失需要下载时使用。数据缓存位于 `data/dataset_cache/uci_<id>.pkl`（25 个文件），加载器按仓库内的相对路径查找，与当前工作目录无关。

## 结果

`results/conformal.csv` 共 9,375 行 = 25 任务 × 5 种子 × 5 折 × 3 方法 × 5 个 α，0 错误行。

| $\alpha$ | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| 覆盖：aggregate | 0.955 | 0.906 | 0.856 | 0.807 | 0.705 |
| 覆盖：union | 0.974 | 0.940 | 0.907 | 0.871 | 0.790 |
| 覆盖：split | 0.956 | 0.905 | 0.857 | 0.804 | 0.704 |
| 大小：aggregate | 1.641 | 1.332 | 1.170 | 1.050 | 0.859 |
| 大小：union | 2.025 | 1.552 | 1.353 | 1.221 | 1.020 |
| 大小：split | 1.913 | 1.381 | 1.191 | 1.065 | 0.871 |

- 匹配 90% 覆盖：aggregate 1.323，union 1.347（差 −0.023，−1.6%，22/1/2，Holm $p=3.2\times10^{-4}$）；split 1.369（差 −0.046，−2.7%，23/0/2，Holm $p=1.7\times10^{-6}$）。
- 平均准确率：aggregate 与 union 均为 0.831（任务范围 0.555–0.993），split 为 0.827。
- $\alpha=0.10$ 时空集比例：aggregate 2.8%，union 1.4%。

## 文件

| 内容 | 路径 |
|---|---|
| edRVFL 与 CV+ 实现 | `src/models/RVFLC.py` |
| 数据加载、预处理、25 题清单 | `src/data_loader.py` |
| 实验驱动（三种方法、写 CSV） | `src/experiment.py` |
| 结果汇总与统计检验 | `src/analyze.py` |
| 交互式入口 | `src/main.py` |
| 结果 | `results/conformal.csv`、`results/analysis.txt` |
| 数据缓存 | `data/dataset_cache/` |
