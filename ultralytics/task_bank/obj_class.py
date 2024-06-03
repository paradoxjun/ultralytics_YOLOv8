from collections import deque

class MoneyCounter:
    def __init__(self, idx_frame, track_id, bbox, confidence, max_history=5):
        self.track_id = track_id        # 当前跟踪ID
        self.bbox = bbox                # 检测框
        self.confidence = confidence    # 置信度
        self.max_history = max_history  # 最大保留的历史帧数
        self.history = deque([(idx_frame, bbox, confidence)], maxlen=self.max_history)     # 保存历史信息

    def update(self, idx_frame, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.history.append((idx_frame, bbox, confidence))
        self.cleanup_history(idx_frame)
        return self.get_stable_bbox()

    def cleanup_history(self, current_frame):
        # 删除超过30帧的历史信息
        self.history = deque([info for info in self.history if current_frame - info[0] <= 30], maxlen=self.max_history)

    def get_stable_bbox(self):
        # 删除低置信度的历史信息
        valid_history = [info for info in self.history if info[2] > 0.5]
        if not valid_history:
            return self.bbox

        # 根据置信度加权计算预测位置
        total_weight = sum(info[2] for info in valid_history)
        if total_weight == 0:
            return self.bbox

        x1 = sum(info[1][0] * info[2] for info in valid_history) / total_weight
        y1 = sum(info[1][1] * info[2] for info in valid_history) / total_weight
        x2 = sum(info[1][2] * info[2] for info in valid_history) / total_weight
        y2 = sum(info[1][3] * info[2] for info in valid_history) / total_weight

        return [x1, y1, x2, y2]

    def get_latest_bbox(self):
        return self.bbox

    def get_latest_confidence(self):
        return self.confidence

    def get_track_id(self):
        return self.track_id