from collections import deque
import numpy as np


class MoneyState:
    """
    钱的状态描述。
    """
    new = 0             # 新目标
    unchecked = 1       # 未验钞
    checking = 2        # 验钞中/在验钞机中
    checked = 3         # 验完钞
    in_boxing = 4       # 在款箱中
    occlusion = 5       # 发生遮挡
    missing = 6         # 钱从视野消失


class MoneyTrack:
    """
    钱的追踪类，提供基本的属性和方法。

    属性：
        _count (int)：用于唯一追踪 ID 的类级计数器。
        track_id (int)：追踪的唯一标识符。
        is_activated (bool)：指示当前追踪是否处于活动状态的标志。
        state (TrackState)：追踪的当前状态。
        history (deque)：追踪状态的有序历史记录。yolov8未实现，这里修改为双端队列实现。
        features (deque)：从对象中提取最特征列表。
        curr_feature (any)：被追踪对象的当前特征。
        score (float)：追踪的置信度分数。
        start_frame (int)：追踪开始的帧号。
        frame_id (int)：轨道处理的最近帧 ID。
        time_since_update (int)：自上次更新以来经过的帧数。
        location (tuple)：多摄像机追踪上下文中对象的位置。

    方法：
        end_frame：返回追踪对象的最后一帧的 ID。
        next_id：增加并返回下一个全局追踪 ID。
        activate：激活追踪的抽象方法。
        predict：预测追踪下一个状态的抽象方法。
        update：使用新数据更新追踪的抽象方法。
        reset_id：重置全局轨道 ID 计数器。
    """

    _count = 0

    def __init__(self, xywh, score, cls):
        """Initializes a new track with unique ID and foundational tracking attributes."""
        self.track_id = 0
        self.is_activated = False
        self.state = MoneyState.new
        self.history = deque()
        self.features = deque()
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)
        self.start_location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """Return the last frame ID of the track."""
        return self.frame_id

    @staticmethod
    def next_id():
        """Increment and return the global track ID counter."""
        MoneyTrack._count += 1
        return MoneyTrack._count

    def activate(self, *args):
        """Abstract method to activate the track with provided arguments."""
        raise NotImplementedError

    def predict(self):
        """Abstract method to predict the next state of the track."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Abstract method to update the track with new observations."""
        raise NotImplementedError

    @staticmethod
    def reset_id():
        """Reset the global track ID counter."""
        MoneyTrack._count = 0
