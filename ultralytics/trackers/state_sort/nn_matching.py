import numpy as np


def _pdist(a, b):
    """
    计算 `a` 和 `b` 中点的两两对应的平方距离。

    参数
    ----------
    a : 一个 NxM 矩阵，包含 N 个维度为 M 的样本。
    b : 一个 LxM 矩阵，包含 L 个维度为 M 的样本。

    返回
    -------
    (ndarray) 返回一个大小为 [len(a), len(b)] 的矩阵，其中元素 (i, j)，包含 `a[i]` 和 `b[j]` 之间的平方距离。
    通俗讲就是：(i， j) 表示 a 的第 i 个行向量（样本）与 b 的第 j 个行向量（样本）的欧式距离平方。
    """
    a, b = np.asarray(a), np.asarray(b)  # 将输入转换为 numpy 数组
    if len(a) == 0 or len(b) == 0:  # 如果 a 或 b 为空，返回全零矩阵
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)  # 得到a^2 和 b^2
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]  # (a - b)^2 == a^2 + b^2 - 2ab
    r2 = np.clip(r2, 0., float(np.inf))  # 将结果裁剪到 [0, +∞]
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """
    计算 `a` 和 `b` 中点的两两余弦距离。

    参数
    ----------
    a : 一个 NxM 矩阵，包含 N 个维度为 M 的样本。
    b : 一个 LxM 矩阵，包含 L 个维度为 M 的样本。
    data_is_normalized : Optional[bool] 如果为 True，假设 a 和 b 中的行是单位长度的向量。否则，a 和 b 会被显式地归一化为长度为 1。

    返回
    -------
    (ndarray) 返回一个大小为 [len(a), len(b)] 的矩阵，其中元素 (i, j)，包含 `a[i]` 和 `b[j]` 之间的余弦距离。
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """
    最近邻距离度量的辅助函数（欧氏距离）。

    参数
    ----------
    x : (ndarray) 一个包含 N 个行向量（样本点）的矩阵。
    y : (ndarray) 一个包含 M 个行向量（查询点）的矩阵。

    返回
    -------
    (ndarray) 一个长度为 M 的向量，其中每个条目包含 `y` 中的每个条目到 `x` 中某个样本的最小欧氏距离。
    通俗讲就是：向量矩阵改编成(1， M)， 只保留每个y向量和x所有向量距离的最大值。
    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """
    最近邻距离度量的辅助函数（余弦距离）。

    参数
    ----------
    x : (ndarray) 一个包含 N 个行向量（样本点）的矩阵。
    y : (ndarray) 一个包含 M 个行向量（查询点）的矩阵。

    返回
    -------
    (ndarray) 一个长度为 M 的向量，其中每个条目包含 `y` 中的每个条目到 `x` 中某个样本的最小余弦距离。
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
     一个最近邻距离度量，对于每个目标，返回到目前为止观察到的任何样本的最近距离。

    参数
    ----------
    metric : (str) "euclidean" 或 "cosine"。
    matching_threshold: (float) 匹配阈值。距离较大的样本被认为是无效匹配。
    budget : (Optional[int]) 如果不为 None，则将每类样本的数量固定为最多此数值。当达到预算时，移除最旧的样本。
    属性
    ----------
    samples : (Dict[int -> List[ndarray]]) 一个字典，将目标身份映射到到目前为止观察到的样本列表。
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """
        用新数据更新距离度量。

        参数
        ----------
        features : (ndarray) 一个 NxM 矩阵，包含 N 个维度为 M 的特征。
        targets : (ndarray) 一个包含相关目标身份的整数数组。
        active_targets : (List[int]) 场景中当前存在的目标列表。
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """
        计算特征和目标之间的距离。

        参数
        ----------
        features : (ndarray) 一个 NxM 矩阵，包含 N 个维度为 M 的特征。
        targets : (List[int]) 一个目标列表，用于匹配给定的 `features`。

        返回
        -------
        (ndarray) 返回一个形状为 [len(targets), len(features)] 的成本矩阵，
        其中元素 (i, j) 包含 `targets[i]` 和 `features[j]` 之间的最小距离。
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)

        return cost_matrix
