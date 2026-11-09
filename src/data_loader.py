import os
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

# 缓存目录
CACHE_DIR = "dataset_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ----------------------------
# 二分类强制白名单：UCI ID → 是否强制二分类
# 仅当调用方显式传入 force_binary=True 时生效：这些数据集即使原始标签 >2 类，
# 也视为二分类（0 vs >0）；不传时按原始多分类标签处理。
# ----------------------------
BINARY_DATASETS_FORCE_BINARY = {
    45,  # Heart Disease
}


# ----------------------------
# 需要显式删除列的白名单：UCI ID → 列名列表
# 这些列是标识符或元数据，不应作为特征
# ----------------------------
COLUMNS_TO_DROP = {
    74: ['molecule_name', 'conformation_name'],  # Musk (Version 1)
    75: ['molecule_name', 'conformation_name'],  # Musk (Version 2)
    44: ['hobby'],
    40: ['name'],  # Flags: country identifier, not a predictive attribute
}


# ----------------------------
# 不应标准化的数据集白名单：UCI ID → True
# 这些数据集跳过 StandardScaler 处理
# ----------------------------
NON_STANDARDIZED_DATASETS = {
    # 示例：MAGIC Gamma Telescope 特征为物理探测器计数，标准化可能不利
    14, # Breast Cancer
    22, # Chess (King-Rook vs. King-Pawn)
    105, # Congressional Voting Records
    26, # Connect-4
    32, # Cylinder Bands
    43, # Haberman's Survival
    45, # Heart Disease (cleveland)
    47, # Horse Colic
    144, # Statlog (German Credit Data)
    151, # Connectionist Bench (Sonar, Mines vs. Rocks)
    52, # Ionosphere
    107, # Waveform
    111 # Zoo
}


# ----------------------------
# 多分类强制映射白名单：UCI ID → 映射函数
# 用于将原始数值标签按自定义规则转为多分类标签
# ----------------------------
MULTICLASS_LABEL_MAPPINGS = {
    1: lambda y: np.select(
        condlist=[
            y <= 8,
            (y >= 9) & (y <= 10),
            y >= 11
        ],
        choicelist=[0, 1, 2],
        default=-1  # 理论上不会出现，用于调试
    ).astype(int),

    242: lambda y_continuous: pd.qcut(
        y_continuous,
        q=5,  # 分成 5 类
        labels=False,  # 返回 0~4 的整数标签
        duplicates='drop'  # 安全处理重复值（理论上不会触发）
    ).astype(int)
}


# ----------------------------
# 多目标数据集的标签列指定：UCI ID → 目标列名
# 用于当 dataset.data.targets 有多个列时，明确使用哪一列
# ----------------------------
TARGET_COLUMN_SELECTION = {
    193: 'CLASS',  # Cardiotocography: use 10-class 'CLASS' instead of 'NSP'
    242: 'Y1',
    40: 'religion',
    189: 'motor_UPDRS'  # Parkinsons Telemonitoring: 使用主要的运动评分作为目标
}


# ----------------------------
# 回归任务固定划分白名单：UCI ID
# 这些UCI数据集虽然是正数ID，但使用固定划分而非交叉验证
# ----------------------------
REGRESSION_FIXED_SPLIT_IDS = {
    1,    # Abalone
    291,  # Airfoil Self-Noise
    162,  # Forest Fires
    189,  # Parkinsons Telemonitoring
}


def should_standardize(uci_id):
    """UCI 数据集是否做标准化（NON_STANDARDIZED_DATASETS 白名单之外才标准化）。"""
    return uci_id not in NON_STANDARDIZED_DATASETS


def fold_preprocess(X_train, X_test, standardize=True, impute='mean'):
    """防泄漏预处理：所有统计量只在训练部分上计算，再应用到测试部分。

    1. 缺失值：用训练部分各列的均值（impute='mean'）或中位数（impute='median'）填充；
       训练部分全为缺失的列填 0。
    2. standardize=True 时：在训练部分上拟合 StandardScaler，再应用到测试部分。

    必须在切分之后调用；不得在包含测试行的全数据上计算任何统计量。
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        stat = np.nanmedian if impute == 'median' else np.nanmean
        with np.errstate(all='ignore'):
            fill = stat(X_train, axis=0)
        fill = np.where(np.isnan(fill), 0.0, fill)
        X_train = np.where(np.isnan(X_train), fill, X_train)
        X_test = np.where(np.isnan(X_test), fill, X_test)
    if not standardize:
        return X_train, X_test
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def load_monk_datasets(one_hot_encode=True, monk_subset=None):
    """加载所有MONK数据集子集或指定的单个子集"""
    from sklearn.preprocessing import OneHotEncoder

    monk_datasets = []
    # 如果没有指定子集，则加载全部三个
    if monk_subset is None:
        monk_ids = [1, 2, 3]
    elif isinstance(monk_subset, int):
        monk_ids = [monk_subset]
    elif isinstance(monk_subset, list):
        monk_ids = monk_subset
    else:
        raise ValueError("monk_subset 必须是整数、整数列表或 None")
    
    for monk_id in monk_ids:
        if monk_id not in [1, 2, 3]:
            print(f"⚠️  无效的MONK数据集ID: {monk_id}，跳过该数据集")
            continue
            
        print(f"📂 正在加载 MONK-{monk_id} 数据集...")
        
        # MONK数据集文件路径
        train_file = f'dataset_cache/monks-{monk_id}.train'
        test_file = f'dataset_cache/monks-{monk_id}.test'
        
        # 检查文件是否存在
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"⚠️  MONK-{monk_id} 数据集文件未找到，跳过该数据集")
            continue
        
        # MONK数据集列名
        feature_names = [f'a{i}' for i in range(1, 7)]
        
        # 加载训练集
        train_data = pd.read_csv(train_file, sep=r'\s+', header=None, 
                                 names=['target'] + feature_names + ['id'])
        X_train = train_data[feature_names]
        y_train = train_data['target'].values
        
        # 加载测试集
        test_data = pd.read_csv(test_file, sep=r'\s+', header=None, 
                                names=['target'] + feature_names + ['id'])
        X_test = test_data[feature_names]
        y_test = test_data['target'].values
        
        # 应用独热编码
        if one_hot_encode:
            encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'避免多重共线性
            X_train_encoded = encoder.fit_transform(X_train)
            X_test_encoded = encoder.transform(X_test)
            
            # 更新特征名以反映独热编码后的特征
            new_feature_names = []
            for i, feature in enumerate(feature_names):
                categories = encoder.categories_[i][1:]  # 排除第一个被drop的类别
                new_feature_names.extend([f'{feature}_{cat}' for cat in categories])
            
            X_train = X_train_encoded
            X_test = X_test_encoded
            feature_names = new_feature_names
        else:
            X_train = X_train.values
            X_test = X_test.values
        
        monk_datasets.append({
            'dataset_name': f'MONK-{monk_id}',
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names
        })
        
        print(f"✅ 成功加载 MONK-{monk_id}: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    
    return monk_datasets


def load_led_dataset(one_hot_encode=False):
    """
    加载 LED Display 数据集
    
    参数:
        one_hot_encode: 是否对特征进行独热编码 (默认 False，因为特征已经是 0/1)
    
    返回:
        包含训练集和测试集的字典列表
    """
    from sklearn.model_selection import train_test_split
    
    print("📂 正在加载 LED Display 数据集...")
    
    # 这些是从 C 代码中提取的 10 个基础 LED 模式 (数字 0-9)
    base_patterns = [
        [1, 1, 1, 0, 1, 1, 1],  # 0
        [0, 0, 1, 0, 0, 1, 0],  # 1
        [1, 0, 1, 1, 1, 0, 1],  # 2
        [1, 0, 1, 1, 0, 1, 1],  # 3
        [0, 1, 1, 1, 0, 1, 0],  # 4
        [1, 1, 0, 1, 0, 1, 1],  # 5
        [1, 1, 0, 1, 1, 1, 1],  # 6
        [1, 0, 1, 0, 0, 1, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1]   # 9
    ]
    
    # 生成带噪声的完整数据集
    from evaluation import set_random_seed
    set_random_seed(42)
    n_samples_per_digit = 200  # 每个数字生成 200 个样本
    noise_rate = 0.1  # 10% 的噪声
    
    X_list = []
    y_list = []
    
    for digit in range(10):
        base_pattern = base_patterns[digit]
        for _ in range(n_samples_per_digit):
            # 添加噪声
            noisy_pattern = np.array(base_pattern).copy()
            for i in range(7):
                if np.random.random() < noise_rate:
                    noisy_pattern[i] = 1 - noisy_pattern[i]  # 反转
            
            X_list.append(noisy_pattern)
            y_list.append(digit)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    feature_names = [f'led_{i}' for i in range(7)]
    
    dataset = {
        'dataset_name': 'LED Display',
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }
    
    print(f"✅ 成功加载 LED Display: 训练集 {X_train.shape}, 测试集 {X_test.shape}, 10 类别")
    
    return [dataset]


def load_lrs_dataset():
    """
    加载 Low Resolution Spectrometer 数据集
    
    返回:
        包含训练集和测试集的字典列表
    """
    from sklearn.model_selection import train_test_split
    
    print("📂 正在加载 Low Resolution Spectrometer 数据集...")
    
    # LRS 数据集文件路径
    data_file = 'dataset_cache/low+resolution+spectrometer/lrs.data'
    names_file = 'dataset_cache/low+resolution+spectrometer/lrs.names'
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"⚠️  LRS 数据集文件未找到：{data_file}")
        return []
    
    # 读取数据
    X_list = []
    y_list = []
    
    with open(data_file, 'r') as f:
        content = f.read()
    
    # 解析数据 - 每个样本用括号包围
    samples = content.strip().split('\n\n')
    
    for sample in samples:
        sample = sample.strip()
        if not sample or not sample.startswith('('):
            continue
        
        # 移除括号
        sample = sample[1:].strip()
        if sample.endswith(')'):
            sample = sample[:-1]
        
        # 分割成字段
        parts = sample.split()
        
        if len(parts) < 103:
            continue
        
        try:
            # 第 2 列是 LRS-class (十位数字作为基本类别)
            lrs_class = int(float(parts[1]))
            basic_class = lrs_class // 10  # 取十位数作为类别 (0-9)
            
            # 提取特征 (第 11 列开始是光谱数据，共 93 个特征)
            features = []
            for i in range(10, min(103, len(parts))):
                val = float(parts[i])
                features.append(val)
            
            if len(features) >= 90:  # 确保有足够的特征
                X_list.append(features)
                y_list.append(basic_class)
        except (ValueError, IndexError) as e:
            continue
    
    if len(X_list) == 0:
        print("⚠️  无法解析 LRS 数据")
        return []
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # 检查类别分布
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"📊 LRS 原始类别分布：{dict(zip(unique_classes, class_counts))}")
    
    # 标准化：已移到划分之后，只在训练集上拟合（防泄漏）
    # 检查是否有类别样本数少于 2 个
    min_samples_per_class = min(class_counts)
    if min_samples_per_class < 2:
        print(f"⚠️  警告：某些类别样本数少于 2 个，使用简单划分而非分层抽样")
        # 使用简单随机划分，不使用 stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # 使用分层抽样
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    X_train, X_test = fold_preprocess(X_train, X_test)

    # 创建特征名
    feature_names = [f'blue_channel_{i}' for i in range(44)] + \
                   [f'red_channel_{i}' for i in range(49)]
    
    dataset = {
        'dataset_name': 'Low Resolution Spectrometer',
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }
    
    unique_classes = len(np.unique(y))
    print(f"✅ 成功加载 LRS: 训练集 {X_train.shape}, 测试集 {X_test.shape}, {unique_classes} 类别")
    
    return [dataset]


def load_uci_dataset(uci_id, dataset_name=None, standardize=True, handle_missing=True, use_cache=True, task='classification', force_binary=False):
    cache_path = os.path.join(CACHE_DIR, f"uci_{uci_id}.pkl")

    # 尝试从缓存加载
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                if cached.get('version') == '1.0':
                    X, y, feature_names, name = cached['X'], cached['y'], cached['feature_names'], cached['name']
                    print(f"✅ 从缓存加载数据集：{name} (ID: {uci_id})")
                    if task == 'regression':
                        # 检查是否在固定划分白名单中
                        if uci_id in REGRESSION_FIXED_SPLIT_IDS:
                            return _process_regression_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id)
                        else:
                            # 其他回归任务（理论上不应该有）
                            return _process_regression_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id)
                    else:
                        return _process_classification_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id, force_binary=force_binary)
        except Exception as e:
            print(f"⚠️ 缓存损坏，重新下载 (ID: {uci_id}): {e}")
            os.remove(cache_path)

    # 缓存未命中，从网络下载
    try:
        print(f"🌐 正在下载 UCI 数据集 (ID: {uci_id})...")
        dataset = fetch_ucirepo(id=uci_id)
        X = dataset.data.features
        y = dataset.data.targets
        if y is None: y = X

        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        name = dataset_name if dataset_name else f"UCI_{uci_id}"

        # 保存原始数据到缓存
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'version': '1.0',
                        'X': X.copy(),
                        'y': y.copy(),
                        'feature_names': feature_names,
                        'name': name
                    }, f)
                print(f"💾 已缓存数据集到 {cache_path}")
            except Exception as e:
                print(f"⚠️ 缓存保存失败：{e}")
        
        if task == 'regression':
            # 检查是否在固定划分白名单中
            if uci_id in REGRESSION_FIXED_SPLIT_IDS:
                return _process_regression_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id)
            else:
                # 其他回归任务（理论上不应该有）
                return _process_regression_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id)
        else:
            return _process_classification_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id, force_binary=force_binary)

    except Exception as e:
        print(f"❌ 无法加载UCI数据集 (ID: {uci_id}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def _process_classification_data(X, y, feature_names, name, standardize=True, handle_missing=True, uci_id=None, force_binary=False):
    """
    分类任务专用数据处理函数
    对数据执行预处理（缺失值、编码、标准化等），并自动移除全空列和全常量列。
    """
    # --------------------------
    # 1. 安全提取 y 为 1D numpy array，支持多目标列选择
    # --------------------------
    target_column = getattr(y, 'name', None)
    if hasattr(y, 'columns'):
        target_column = y.columns[0]
    if hasattr(y, 'columns') and len(y.columns) > 1:
        if uci_id in TARGET_COLUMN_SELECTION:
            col_name = TARGET_COLUMN_SELECTION[uci_id]
            if col_name not in y.columns:
                raise ValueError(f"UCI ID {uci_id}: 指定的目标列 '{col_name}' 不存在。可用列: {list(y.columns)}")
            print(f"🎯 UCI ID {uci_id}: 使用指定目标列 '{col_name}'")
            target_column = col_name
            y = y[col_name]
        else:
            print(f"⚠️ UCI ID {uci_id} 有多个目标列 ({list(y.columns)})，默认使用第一列: '{y.columns[0]}'")
            y = y.iloc[:, 0]

    # Some UCI datasets (notably Flags) expose the target among the features.
    # Remove it before encoding, including when targets were cached as X itself.
    X = X.copy()
    if hasattr(X, 'columns') and target_column in X.columns:
        print(f"🗑️  移除输入中的目标列: {target_column}")
        X = X.drop(columns=[target_column])
        feature_names = X.columns.tolist()

    # 转为 1D numpy array
    if hasattr(y, 'values'):
        y = y.values
    else:
        y = np.asarray(y)

    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            raise ValueError(f"标签 y 维度异常: {y.shape}，无法转为 1D")

    # A missing target is an unlabeled observation, never an extra class.
    missing_tokens = {'', '?', 'nan', 'none', 'null', '<na>'}
    valid_mask = np.array([
        not pd.isna(value) and str(value).strip().lower() not in missing_tokens
        for value in y
    ], dtype=bool)
    if not valid_mask.all():
        print(f"🗑️  移除缺失标签样本: {int((~valid_mask).sum())}")
        X = X.iloc[valid_mask].copy() if hasattr(X, 'iloc') else X[valid_mask]
        y = y[valid_mask]
    if not len(y):
        raise ValueError(f"UCI ID {uci_id}: 没有有效标签样本")

    # --------------------------
    # 2. 多分类强制映射（优先于二分类）
    # --------------------------
    if uci_id in MULTICLASS_LABEL_MAPPINGS:
        y_numeric = pd.to_numeric(y, errors='coerce')
        if np.any(np.isnan(y_numeric)):
            raise ValueError(f"UCI ID {uci_id} 的标签包含无法转为数值的内容，无法应用多分类映射。")
        y = MULTICLASS_LABEL_MAPPINGS[uci_id](y_numeric)
        unique_labels = np.unique(y)
        if np.any(unique_labels == -1):
            raise ValueError(f"UCI ID {uci_id}: 多分类映射产生无效标签（-1），请检查映射逻辑。")
        print(f"🔄 UCI ID {uci_id}: 应用多分类映射，标签范围 → {sorted(unique_labels)}")

    # --------------------------
    # 3. 强制二分类处理（仅当调用方要求二分类场景时生效）
    # --------------------------
    elif force_binary and uci_id in BINARY_DATASETS_FORCE_BINARY:
        y_numeric = pd.to_numeric(y, errors='coerce')
        if not np.all(np.isnan(y_numeric)):
            y = np.array((y_numeric > 0)).astype(int)
        else:
            y_clean = np.array([str(val).strip() for val in y])
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_clean)
            unique_vals = np.unique(y_encoded)
            if len(unique_vals) == 2:
                y = y_encoded
            else:
                raise ValueError(
                    f"UCI ID {uci_id} is in BINARY_DATASETS_FORCE_BINARY but has >2 unique labels after encoding!")

    # --------------------------
    # 4. 通用标签清洗 + 编码
    # --------------------------
    else:
        y = np.asarray(y, dtype=object)

        def clean_label(val):
            if isinstance(val, str):
                return val.strip().rstrip('.').strip()
            else:
                return str(val)

        # 149 脏数据集 特殊处理
        if uci_id == 149:
            valid_categories = ['van', 'saab', 'bus', 'opel']
            valid_mask = np.isin(y, valid_categories)
            X = X[valid_mask]
            y = y[valid_mask]

        y_clean = np.array([clean_label(val) for val in y])
        le = LabelEncoder()
        y = le.fit_transform(y_clean)

    X = X.copy()

    # --------------------------
    # 特殊列删除：根据 COLUMNS_TO_DROP 白名单
    # --------------------------
    if uci_id in COLUMNS_TO_DROP:
        cols_to_drop = [col for col in COLUMNS_TO_DROP[uci_id] if col in X.columns]
        if cols_to_drop:
            print(f"🗑️  根据白名单移除列 (UCI ID {uci_id}): {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
            feature_names = [f for f in feature_names if f not in cols_to_drop]

    # --------------------------
    # 处理分类特征（OneHotEncoder）
    # --------------------------
    if hasattr(X, 'columns'):
        categorical_cols = []
        if X.columns.duplicated().any():
            print(f"⚠️ 检测到重复列名 (数据集: {name})，自动去重（保留首次出现）...")
            X = X.loc[:, ~X.columns.duplicated()]
            feature_names = X.columns.tolist()
        for col in X.columns:
            # pandas 3.0 起字符串列默认 str dtype（不再是 object），一并纳入分类特征
            if (X[col].dtype == 'object' or X[col].dtype.name == 'category'
                    or pd.api.types.is_string_dtype(X[col])):
                categorical_cols.append(col)

        if categorical_cols:
            from sklearn.preprocessing import OneHotEncoder
            # 使用独热编码处理分类特征
            encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'避免多重共线性
            X_categorical = X[categorical_cols]
            X_encoded = encoder.fit_transform(X_categorical)
            
            # 获取编码后的特征名
            new_categorical_feature_names = []
            for i, col in enumerate(categorical_cols):
                categories = encoder.categories_[i][1:]  # 排除第一个被drop的类别
                new_categorical_feature_names.extend([f'{col}_{cat}' for cat in categories])
            
            # 处理数值特征
            numerical_cols = [col for col in X.columns if col not in categorical_cols]
            if numerical_cols:
                X_numerical = X[numerical_cols]
                # 合并数值特征和编码后的分类特征
                X_encoded = np.concatenate([X_numerical.values, X_encoded], axis=1)
                feature_names = numerical_cols + new_categorical_feature_names
            else:
                feature_names = new_categorical_feature_names
                
            # 转换为DataFrame便于后续处理
            X = pd.DataFrame(X_encoded, columns=feature_names)

    # 缺失值不在此处填充（会把全数据均值写进训练特征），由 fold_preprocess 在训练折内完成

    # --------------------------
    # 转为数值型并清理
    # --------------------------
    X, feature_names = _clean_and_convert_features(X, feature_names, uci_id)
    
    # 移除全常量列（唯一值 ≤ 1）
    X_df = pd.DataFrame(X, columns=feature_names)
    n_unique = X_df.nunique()
    constant_cols = n_unique[n_unique <= 1].index.tolist()
    if constant_cols:
        print(f"🗑️  移除全常量列 (UCI ID {uci_id}): {constant_cols}")
        X_df = X_df.drop(columns=constant_cols)
        feature_names = [f for f in feature_names if f not in constant_cols]
    
    if X_df.empty:
        raise ValueError(f"UCI ID {uci_id}: 所有特征列均被移除（全空或全常量），无法继续处理！")
    X = X_df.values

    # --------------------------
    # 标准化不在此处进行：必须在训练折上拟合 scaler（见 fold_preprocess），否则测试折统计量会泄漏进训练特征

    return X, y, feature_names, name


def _process_regression_data(X, y, feature_names, name, standardize=True, handle_missing=True, uci_id=None):
    """
    回归任务专用数据处理函数（带固定划分）
    返回字典格式，包含训练集和测试集
    """
    from sklearn.model_selection import train_test_split
    
    # 1. 安全提取 y 为 1D numpy array
    if hasattr(y, 'columns') and len(y.columns) > 1:
        if uci_id in TARGET_COLUMN_SELECTION:
            col_name = TARGET_COLUMN_SELECTION[uci_id]
            if col_name not in y.columns:
                print(f"⚠️ UCI ID {uci_id}: 指定的目标列 '{col_name}' 不存在，使用第一列")
                y = y.iloc[:, 0]
            else:
                print(f"🎯 UCI ID {uci_id}: 使用指定目标列 '{col_name}'")
                y = y[col_name]
        else:
            print(f"⚠️ UCI ID {uci_id} 有多个目标列 ({list(y.columns)})，默认使用第一列: '{y.columns[0]}'")
            y = y.iloc[:, 0]
    
    if hasattr(y, 'values'):
        y = y.values
    else:
        y = np.asarray(y)
    
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            y = y[:, 0].ravel()
    
    y = pd.to_numeric(y, errors='coerce').astype(np.float64)
    
    # 2. 处理 X 的列删除
    X = X.copy()
    if uci_id in COLUMNS_TO_DROP:
        cols_to_drop = [col for col in COLUMNS_TO_DROP[uci_id] if col in X.columns]
        if cols_to_drop:
            print(f"🗑️  根据白名单移除列 (UCI ID {uci_id}): {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
            feature_names = [f for f in feature_names if f not in cols_to_drop]
    
    # 3. 缺失值不在此处填充（会把全数据中位数写进训练特征），由 fold_preprocess 在训练折内完成

    # 4. 转为数值型并清理
    X, feature_names = _clean_and_convert_features(X, feature_names, uci_id)
    
    # 5. 固定比例划分（先划分，再只在训练集上计算统计量，防泄漏）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if standardize:
        X_train, X_test = fold_preprocess(X_train, X_test, True, impute='median')
    
    print(f"✅ 数据集 {name} 已划分为: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    
    return {
        'dataset_name': name,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }


def _clean_and_convert_features(X, feature_names, uci_id=None):
    """
    通用特征清理和转换函数
    移除全空/全无穷列，转换为数值类型
    """
    if hasattr(X, 'values'):
        X = X.values
    try:
        X = X.astype(float)
    except (ValueError, TypeError):
        X_array = np.asarray(X, dtype=object)
        X_df_temp = pd.DataFrame(X_array)
        for col in X_df_temp.columns:
            X_df_temp[col] = pd.to_numeric(X_df_temp[col], errors='coerce')
        X = X_df_temp.fillna(0).values
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 移除全空/全无穷列
    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        all_inf_cols = X_df.columns[np.isinf(X_df[numeric_cols]).all(axis=0)].tolist()
    else:
        all_inf_cols = []
    invalid_cols = set(all_nan_cols + all_inf_cols)
    
    if invalid_cols:
        print(f"🗑️  移除全空/全无穷列 (UCI ID {uci_id}): {sorted(invalid_cols)}")
        X_df = X_df.drop(columns=invalid_cols)
        feature_names = [f for f in feature_names if f not in invalid_cols]
    
    if X_df.empty:
        raise ValueError(f"UCI ID {uci_id}: 所有特征列均被移除，无法继续处理！")
    
    return X_df.values, feature_names


def get_classification_datasets():
    binary_datasets = [
        (2, "Adult"),
        (222, "Bank Marketing"),
        (176, "Blood Transfusion Service Center"),
        (14, "Breast Cancer"),
        (17, "Breast Cancer Wisconsin (Diagnostic)"),
        (22, "Chess (King-Rook vs. King-Pawn)"),
        (105, "Congressional Voting Records"),
        (27, "Credit Approval"),
        (32, "Cylinder Bands"),
        (43, "Haberman's Survival"),
        (45, "Heart_cleveland"),
        (47, "Horse Colic"),
        (70, "MONK-1"),
        (70, "MONK-2"),
        (70, "MONK-3"),
        (73, "Mushroom"),
        (74, "Musk_1"),
        (75, "Musk_2"),
        (172, "Ozone Level Detection"),
        (174, "Parkinsons"),
        (94, "Spambase"),
        (144, "Statlog (German Credit Data)"),
        (46, "Hepatitis"),
        (151, "Connectionist Bench (Sonar, Mines vs. Rocks)"),
        (159, "MAGIC Gamma Telescope"),
        (52, "Ionosphere"),
        (198, "Steel Plates Faults")
    ]

    multiclass_datasets = [
        (1, "Abalone"),
        (3, "Annealing"),
        (193, "Cardiotocography_10classes"),
        (19, "Car Evaluation"),
        (26, "Connect-4"),
        (33, "Dermatology"),
        (39, "Ecoli"),
        (242, "Energy Efficiency"),
        # Flags (UCI 40) is out of this project's benchmark.
        (42, "Glass Identification"),
        # 本地数据集
        (-1, "LED Display"),  # 特殊标记，使用自定义加载函数
        (-2, "Low Resolution Spectrometer"),  # 特殊标记，使用自定义加载函数
        (44, "Hayes-Roth"),
        (45, "Heart_cleveland"),
        (63, "lymphography"),
        (69, "Molecular Biology"),
        (76, "Nursery"),
        (80, "Optical Recognition of Handwritten Digits"),
        (91, "Soybean"),
        (148, "Statlog (Shuttle)"),
        (146, "Statlog (Landsat Satellite)"),
        (149, "Statlog (Vehicle Silhouettes)"),
        (212, "Vertebral Column"),
        (107, "Waveform"),
        (110, "Yeast"),
        (109, "Wine"),
        (186, "Wine Quality"),
        (111, "Zoo")
    ]

    return binary_datasets, multiclass_datasets


def load_arff_regression_dataset(dataset_name, arff_filename, n_features, feature_prefix='feature', 
                                  target_index=None, feature_names=None, label_mapping=None, 
                                  categorical_features=None, standardize=True):
    """
    通用 ARFF 格式回归数据集加载函数
    
    参数:
        dataset_name: 数据集名称（用于显示）
        arff_filename: ARFF 文件路径（相对于项目根目录）
        n_features: 特征数量
        feature_prefix: 特征名前缀（默认 'feature'，当 feature_names 为 None 时使用）
        target_index: 目标变量列索引（默认为 n_features，即最后一列之前的列）
        feature_names: 可选，自定义特征名称列表
        label_mapping: 兼容旧调用的数值目标映射；回归目标声明仍必须为数值型，禁止分类派生版本
        categorical_features: 可选，分类特征的列索引列表（例如 [0, 1] 表示前两列是分类特征）
    
    返回:
        包含训练集和测试集的字典列表
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    
    print(f"📂 正在加载 {dataset_name} 数据集...")
    
    # 检查文件是否存在
    if not os.path.exists(arff_filename):
        print(f"⚠️  {dataset_name} 数据集文件未找到：{arff_filename}")
        return []
    
    # 读取 ARFF 文件
    X_list = []
    y_list = []
    in_data_section = False
    attribute_types = []
    
    # 默认目标变量在特征之后
    if target_index is None:
        target_index = n_features
    
    with open(arff_filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('%'):
                continue
            
            # 检测数据部分开始
            if line.lower().startswith('@attribute'):
                match = re.match(r'''@attribute\s+(?:'[^']+'|"[^"]+"|\S+)\s+(.+)''', line, re.I)
                if match:
                    attribute_types.append(match.group(1).lower())
            if line.lower().startswith('@data'):
                if target_index >= len(attribute_types) or not attribute_types[target_index].startswith(('numeric', 'real', 'integer')):
                    raise ValueError(f"{dataset_name}: 回归目标必须为数值型；请勿使用分类派生版本")
                in_data_section = True
                continue
            
            # 处理数据行
            if in_data_section and line and not line.startswith('@'):
                try:
                    # 解析带引号的值（处理分类特征）
                    values = _parse_arff_line(line)
                    
                    # 提取特征（前 n_features 列）
                    features = []
                    for i in range(n_features):
                        val = values[i].strip("'\"")
                        # 如果是分类特征，保留原始值；否则转为数值
                        if categorical_features is not None and i in categorical_features:
                            features.append(val)
                        else:
                            features.append(float(val))
                    
                    # 提取目标变量
                    target_val = values[target_index].strip("'\"")
                    
                    # 应用标签映射（如果提供）
                    if label_mapping is not None:
                        target = label_mapping.get(target_val)
                        if target is None:
                            continue  # 跳过无效标签
                    else:
                        target = float(target_val)
                    
                    X_list.append(features)
                    y_list.append(target)
                except (ValueError, IndexError) as e:
                    continue
    
    if len(X_list) == 0:
        print(f"⚠️  无法解析 {dataset_name} 数据")
        return []
    
    X_raw = np.array(X_list)
    y = np.array(y_list)
    
    # 处理分类特征
    if categorical_features is not None and len(categorical_features) > 0:
        # 分离分类特征和数值特征
        categorical_cols = [i for i in range(n_features) if i in categorical_features]
        numerical_cols = [i for i in range(n_features) if i not in categorical_features]
        
        # 对分类特征进行独热编码
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        categorical_data = X_raw[:, categorical_cols]
        categorical_encoded = encoder.fit_transform(categorical_data)
        
        # 提取数值特征
        if numerical_cols:
            numerical_data = X_raw[:, numerical_cols].astype(float)
            X = np.concatenate([categorical_encoded, numerical_data], axis=1)
        else:
            X = categorical_encoded
        
        # 生成特征名
        if feature_names is None:
            feature_names = []
            # 为每个分类特征生成编码后的特征名
            for i, col_idx in enumerate(categorical_cols):
                categories = encoder.categories_[i][1:]  # 排除第一个
                if categorical_features[col_idx] if isinstance(categorical_features, dict) else f'cat_{col_idx}':
                    base_name = feature_names[col_idx] if feature_names and col_idx < len(feature_names) else f'cat_{col_idx}'
                else:
                    base_name = f'cat_{col_idx}'
                feature_names.extend([f'{base_name}_{cat}' for cat in categories])
            # 添加数值特征名
            for col_idx in numerical_cols:
                feature_names.append(f'{feature_prefix}{col_idx}')
        else:
            # 使用提供的特征名，但需要扩展分类特征
            original_names = feature_names.copy()
            feature_names = []
            for i, col_idx in enumerate(categorical_cols):
                categories = encoder.categories_[i][1:]
                base_name = original_names[col_idx] if col_idx < len(original_names) else f'cat_{col_idx}'
                feature_names.extend([f'{base_name}_{cat}' for cat in categories])
            for col_idx in numerical_cols:
                feature_names.append(original_names[col_idx] if col_idx < len(original_names) else f'{feature_prefix}{col_idx}')
    else:
        # 没有分类特征，直接转为数值数组
        X = X_raw.astype(float)
        if feature_names is None:
            feature_names = [f'{feature_prefix}{i}' for i in range(1, n_features + 1)]
    
    print(f"📊 {dataset_name} 数据集：{X.shape[0]} 样本，{X.shape[1]} 特征")
    if label_mapping is not None:
        unique_labels = np.unique(y)
        label_counts = {label: np.sum(y == label) for label in unique_labels}
        print(f"📊 目标变量分布：{label_counts}")
    else:
        print(f"📊 目标变量范围：[{y.min():.4f}, {y.max():.4f}], 均值：{y.mean():.4f}")
    
    # 划分训练集和测试集（先划分，再只在训练集上计算统计量，防泄漏）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if standardize:
        X_train, X_test = fold_preprocess(X_train, X_test)
    
    dataset = {
        'dataset_name': dataset_name,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }
    
    print(f"✅ 成功加载 {dataset_name}: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    
    return [dataset]


def _parse_arff_line(line):
    """
    解析 ARFF 数据行，正确处理带引号的字符串值
    
    返回:
        list: 分割后的值列表
    """
    values = []
    current = ""
    in_quotes = False
    quote_char = None
    
    for char in line:
        if char in ("'", '"') and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == ',' and not in_quotes:
            values.append(current.strip())
            current = ""
        else:
            current += char
    
    # 添加最后一个值
    if current:
        values.append(current.strip())
    
    return values


# ARFF 回归数据集配置
ARFF_REGRESSION_CONFIGS = {
    -3: {'name': '2D Planes', 'file': 'dataset_cache/2dplanes.arff', 'n_features': 10, 'feature_prefix': 'x'},
    -4: {'name': 'Ailerons', 'file': 'dataset_cache/ailerons.arff', 'n_features': 40, 'feature_prefix': 'feature_'},
    -5: {'name': 'California Housing', 'file': 'dataset_cache/california_housing.arff', 'n_features': 8, 
         'target_index': 8, 'feature_names': ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                                              'total_bedrooms', 'population', 'households', 'median_income']},
    -6: {'name': 'Delta Ailerons', 'file': 'dataset_cache/delta_ailerons.arff', 'n_features': 5,
         'feature_names': ['RollRate', 'PitchRate', 'currPitch', 'currRoll', 'diffRollRate']},
    -7: {'name': 'Kin8nm', 'file': 'dataset_cache/kin8nm.arff', 'n_features': 8,
         'feature_names': ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8']},
    -8: {'name': 'Servo', 'file': 'dataset_cache/servo.arff', 'n_features': 4,
         'feature_names': ['motor', 'screw', 'pgain', 'vgain'],
         'categorical_features': [0, 1]},
    -9: {'name': 'Triazines', 'file': 'dataset_cache/triazines.arff', 'n_features': 60, 'feature_prefix': 'p'},
    -10: {'name': 'Yacht Hydrodynamics', 'file': 'dataset_cache/yacht_hydrodynamics.arff', 'n_features': 6,
          'feature_names': ['Logitudinal.position', 'Prismatic.coefficient', 'Length.displacement.ratio',
                           'Beam.draught.ratio', 'Length.beam.ratio', 'Froude.number']},
    -11: {'name': 'Elevators', 'file': 'dataset_cache/elevators.arff', 'n_features': 18,
          'feature_names': ['climbRate', 'Sgz', 'p', 'q', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate',
                           'diffDiffClb', 'SaTime1', 'SaTime2', 'SaTime3', 'SaTime4', 'diffSaTime1',
                           'diffSaTime2', 'diffSaTime3', 'diffSaTime4', 'Sa']},
    -12: {'name': 'Pol', 'file': 'dataset_cache/pol.arff', 'n_features': 48, 'feature_prefix': 'f'},
    -13: {'name': 'Stock', 'file': 'dataset_cache/stock.arff', 'n_features': 9, 'target_index': 9,
          'feature_names': [f'company{i}' for i in range(1, 10)]},
}


def load_arff_regression_by_id(dataset_id, standardize=True):
    """根据数据集ID加载ARFF格式的回归数据集"""
    if dataset_id not in ARFF_REGRESSION_CONFIGS:
        print(f"⚠️  未知的数据集ID: {dataset_id}")
        return []
    
    config = ARFF_REGRESSION_CONFIGS[dataset_id]
    feature_prefix_val = config.get('feature_prefix')
    categorical_features_val = config.get('categorical_features')
    
    return load_arff_regression_dataset(
        dataset_name=config['name'],
        standardize=standardize,
        arff_filename=config['file'],
        n_features=config['n_features'],
        feature_prefix=str(feature_prefix_val) if feature_prefix_val is not None else 'feature',
        target_index=config.get('target_index'),
        feature_names=config.get('feature_names'),
        label_mapping=config.get('label_mapping'),
        categorical_features=categorical_features_val if isinstance(categorical_features_val, (list, tuple)) or categorical_features_val is None else None
    )


def get_regression_datasets():
    """获取回归数据集列表"""
    regression_datasets = [
        (-3, "2D Planes"),  # ARFF本地数据集
        (1, "Abalone"),  # UCI数据集（固定划分）
        (-4, "Ailerons"),  # ARFF本地数据集
        (291, "Airfoil Self-Noise"),  # UCI数据集（固定划分）
        (-5, "California Housing"),  # ARFF本地数据集
        (-6, "Delta Ailerons"),  # ARFF本地数据集
        (-11, "Elevators"),  # ARFF本地数据集
        (162, "Forest Fires"),  # UCI数据集（固定划分）
        (-7, "Kin8nm"),  # ARFF本地数据集
        (-12, "Pol"),  # ARFF本地数据集
        (189, "Parkinsons Telemonitoring"),  # UCI数据集（固定划分）
        (-8, "Servo"),  # ARFF本地数据集
        (-13, "Stock"),  # ARFF本地数据集
        (-9, "Triazines"),  # ARFF本地数据集
        (-10, "Yacht Hydrodynamics"),  # ARFF本地数据集
    ]
    return regression_datasets
