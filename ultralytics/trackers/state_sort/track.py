from collections import deque


class MoneyState:
    """
    钱的状态描述。
    """
    unconfirmed = 0     # 还未确认为有效目标
    unchecked = 1       # 未验钞
    checking = 2        # 验钞中/在验钞机中
    checked = 3         # 验完钞
    occlusion = 4       # 发生遮挡
    in_boxing = 5       # 在款箱中
    missing = 6         # 钱从视野消失


class MoneyTrack:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None, min_confidence=0.4, max_history=15):
        # deepsort中原始的参数
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self._n_init = n_init
        self._max_age = max_age
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = MoneyState.unchecked       # 修改成自定义状态
        self.features = []
        if feature is not None:
            self.features.append(feature)

        # 自己额外增加的参数
        self.min_confidence = min_confidence    # 设一个最小置信度阈值
        self.max_history = max_history          # 最大保留的历史帧数
        self.desc = [0, None]                   # 描述信息：[时间信息，字符串信息]
        self.history = deque(maxlen=self.max_history)     # 保存的历史信息

    def to_xywh(self):
        # xyah --> xywh
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_xyxy(self):
        # xyah --> xyxy
        ret = self.to_xywh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        # 使用卡尔曼滤波器预测步骤，将状态分布传播到当前时间步骤。
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        # 执行卡尔曼滤波器测量更新步骤并更新特征缓存。
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        if detection.feature is not None:
            self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == MoneyState.unconfirmed and self.hits >= self._n_init:      # 确认是有效检测框后状态默认为未验钞
            self.state = MoneyState.unchecked

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.time_since_update > self._max_age:
            self.state = MoneyState.missing

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state != MoneyState.unconfirmed
