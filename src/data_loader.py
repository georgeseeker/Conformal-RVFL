import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
try:
    from ucimlrepo import fetch_ucirepo  # 仅在缓存缺失需要下载时使用
except ImportError:  # pragma: no cover
    fetch_ucirepo = None

# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_cache")


# ----------------------------
# 需要显式删除列的白名单：UCI ID → 列名列表
# 这些列是标识符或元数据，不应作为特征
# ----------------------------
COLUMNS_TO_DROP = {
    44: ['hobby'],  # Hayes-Roth
}


# ----------------------------
# 不应标准化的数据集白名单：UCI ID → True
# 这些数据集跳过 StandardScaler 处理
# ----------------------------
NON_STANDARDIZED_DATASETS = {
    26,   # Connect-4
    45,   # Heart Disease (cleveland)
    107,  # Waveform
    111,  # Zoo
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


def load_uci_dataset(uci_id, dataset_name=None, standardize=True, handle_missing=True, use_cache=True):
    """加载一个多分类 UCI 任务（本项目只保留 tasks.MULTICLASS_TASKS 中的 25 个）。"""
    cache_path = os.path.join(CACHE_DIR, f"uci_{uci_id}.pkl")

    # 尝试从缓存加载
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                if cached.get('version') == '1.0':
                    X, y, feature_names, name = cached['X'], cached['y'], cached['feature_names'], cached['name']
                    print(f"✅ 从缓存加载数据集：{name} (ID: {uci_id})")
                    return _process_classification_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id)
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
        
        return _process_classification_data(X, y, feature_names, name, standardize, handle_missing, uci_id=uci_id)

    except Exception as e:
        print(f"❌ 无法加载UCI数据集 (ID: {uci_id}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def _process_classification_data(X, y, feature_names, name, standardize=True, handle_missing=True, uci_id=None):
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
    """兼容旧接口：返回 (二分类列表, 多分类列表)。本项目只保留多分类，二分类恒为空。"""
    from tasks import MULTICLASS_TASKS
    return [], list(MULTICLASS_TASKS)
