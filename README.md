# Conformal-RVFL

在三头 edRVFL（`src/models/RVFLC.py`）上比较三种共形预测集方法：
- `J3`：聚合分数，10 折 CV+ 校准（本文方法）；
- `PH-UNION-J3`：逐头校准后取并集（消融）；
- `SPLIT-J3`：聚合分数，25% 留出校准（split 基线）。

实验只做 **25 个多分类 UCI 任务**。

## 保留的 25 个任务（唯一来源：`src/data_loader.py`）
Abalone(1)、Annealing(3)、Cardiotocography_10classes(193)、Car Evaluation(19)、Connect-4(26)、Dermatology(33)、Ecoli(39)、
Energy Efficiency(242)、Glass Identification(42)、Hayes-Roth(44)、Heart_cleveland(45，原生五类)、lymphography(63)、Molecular Biology(69)、
Nursery(76)、Optical Recognition of Handwritten Digits(80)、Soybean(91)、Statlog (Shuttle)(148)、Statlog (Landsat Satellite)(146)、
Statlog (Vehicle Silhouettes)(149)、Vertebral Column(212)、Waveform(107)、Yeast(110)、Wine(109)、Wine Quality(186)、Zoo(111)。

## 协议（未改动）
- 外层用 5 折 StratifiedKFold（shuffle，random_state=42）。
- 5 个固定种子：1618、3979、6187、7333、8443。
- α ∈ {0.05, 0.10, 0.15, 0.20, 0.30}。
- 模型为 3 层，每层 50 个隐藏节点，λ=0.1，默认设备 `DEVICE="cuda"`。
- 预处理只在训练折上拟合（`fold_preprocess`）。是否标准化由 `should_standardize` 决定：Connect-4、Heart、Waveform、Zoo 不做标准化。

## 运行（在仓库根目录）
```bash
python src/analyze.py                 # 只汇总已有的 results/conformal.csv，不跑实验
python src/experiment.py --help
python src/experiment.py              # ⚠️ 会先清空 results/ 再重跑全部 25 个任务 × 5 个种子（需要 GPU）
python src/experiment.py --tasks Wine --out /tmp/wine.csv --seeds 1618   # ⚠️ 同样会先清空 results/
python src/main.py                    # 交互式入口；选 3 只做汇总
```
依赖：numpy、scipy、scikit-learn、pandas、torch。ucimlrepo 是可选的，只有缓存缺失需要下载时才用到。
数据缓存位于 `data/dataset_cache/uci_<id>.pkl`（25 个文件），加载器按仓库内的相对位置找到它们，与当前工作目录无关。

## 结果
- `results/conformal.csv`：原始结果，9375 行，正好是这 25 个任务 × 5 个种子 × 5 折 × 3 种方法 × 5 个 α，没有错误行。
- `results/analysis.txt`：原分析输出，保持不变。
- `analyze.py` 现在会显式筛选到 `src/data_loader.py` 里的 25 个任务。CSV 中若有其他任务的行，会被列为"不在范围内"，不会被删除。

## 2026-11-11 清理说明
清理时只删去了无用代码和数据，没有改动任务定义、标签、协议、种子、模型或算法。

**删除的代码**
- 二分类任务清单，以及只服务二分类的强制二值化分支（这 25 个任务一直以 `force_binary=False` 运行，这个分支从未被执行）。
- MONK、LED Display、Low Resolution Spectrometer 的加载函数。
- 回归相关的全部代码：UCI 回归处理、ARFF 回归加载与配置、回归任务清单。
- `experiment.py` 中的 `FIXED_PARTITION_IDS`、`SCOPE_GROUP`、`EXCLUDE_TASKS`（Flags）筛选逻辑，现在直接使用 `data_loader.MULTICLASS_TASKS`。
- ID 表中只属于已删任务的条目：Musk、Flags、Parkinsons 的列与目标设置，以及二分类组的免标准化条目。

**删除的数据（35 个文件，约 18 MB）**
- 11 个 ARFF 回归文件和 `regression_sources.json`；
- MONK 的 6 个文件；
- `led+display+domain/`（5 个文件）和 `low+resolution+spectrometer/`（6 个文件）；
- `uci_13`、`uci_40`（Flags）、`uci_53`、`uci_162`、`uci_189`、`uci_291` 六个 pickle。

**保留并共用的部分**
- `fold_preprocess`、`should_standardize`、UCI 分类数据清洗（`_process_classification_data`、`_clean_and_convert_features`）；
- 多分类标签映射（Abalone、Energy Efficiency）、目标列选择（Cardiotocography、Energy Efficiency）、Hayes-Roth 的列删除规则。

**行为上唯一的变化**
- `CACHE_DIR` 改为相对本仓库的路径，`experiment.py` 不再 `chdir`；导入时也不再新建目录。

**验证**
- 25 个任务清理前后逐个加载，X/y 指纹完全一致。
- `analyze.py` 的输出与原 `analysis.txt` 只差第一行的范围说明文字。
- 修改前的备份在本机仓库之外，没有提交。
