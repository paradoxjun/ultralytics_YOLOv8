import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class STrack(BaseTrack):
    """
    单目标跟踪表示，使用卡尔曼滤波进行状态估计。该类负责存储有关单个跟踪轨迹的所有信息，并基于卡尔曼滤波执行状态更新和预测。

    属性：
        shared_kalman (KalmanFilterXYAH): 在所有STrack实例中共享的卡尔曼滤波器，用于预测。
        _tlwh (np.ndarray): 用于存储边界框的左上角坐标和宽高的私有属性。
        kalman_filter (KalmanFilterXYAH): 用于该特定目标跟踪的卡尔曼滤波器实例。
        mean (np.ndarray): 状态估计的均值向量。
        covariance (np.ndarray): 状态估计的协方差。
        is_activated (bool): 标志跟踪是否已激活的布尔值。
        score (float): 跟踪的置信得分。
        tracklet_len (int): 轨迹的长度。
        cls (any): 目标的类别标签。
        idx (int): 目标的索引或标识符。
        frame_id (int): 当前帧的ID。
        start_frame (int): 目标首次检测到的帧。

    方法：
        predict(): 使用卡尔曼滤波预测目标的下一个状态。
        multi_predict(stracks): 使用卡尔曼滤波对多个轨迹进行预测。
        multi_gmc(stracks, H): 使用单应性矩阵更新多个轨迹的状态。
        activate(kalman_filter, frame_id): 激活新的轨迹。
        re_activate(new_track, frame_id, new_id): 重新激活之前丢失的轨迹。
        update(new_track, frame_id): 更新匹配轨迹的状态。
        convert_coords(tlwh): 将边界框转换为 x-y-纵横比-高度 格式。
        tlwh_to_xyah(tlwh): 将tlwh边界框转换为xyah格式。
    """

    shared_kalman = KalmanFilterXYAH()          # 共享的卡尔曼滤波器实例

    def __init__(self, xywh, score, cls):       # 初始化新的 STrack 实例
        super().__init__()                      # 调用父类 BaseTrack 的初始化方法
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"   # xywh+idx or xywha+idx
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)              # xywh 转换为左上角坐标和宽高格
        self.kalman_filter = None                   # 初始化卡尔曼滤波器为 None
        self.mean, self.covariance = None, None     # 初始化均值和协方差为 None
        self.is_activated = False                   # 初始化激活状态为 False

        self.score = score          # 存储置信度得分
        self.tracklet_len = 0       # 初始化轨迹长度为 0
        self.cls = cls              # 存储目标类别
        self.idx = xywh[-1]         # 存储目标的索引或标识符
        self.angle = xywh[4] if len(xywh) == 6 else None    # 如果 xywh 长度为 6，则存储宽高比a，否则为 None

    def predict(self):              # 使用卡尔曼滤波器预测均值和协方差
        mean_state = self.mean.copy()               # 复制当前的均值状态
        if self.state != TrackState.Tracked:        # 如果当前状态不是 Tracked
            mean_state[7] = 0                       # 将 mean_state 的第 7 个元素设置为 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)    # 使用卡尔曼滤波器进行预测

    @staticmethod
    def multi_predict(stracks):     # 对给定的 stracks 使用卡尔曼滤波进行多目标预测
        if len(stracks) <= 0:       # 如果 stracks 的长度小于等于 0
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])         # 复制所有轨迹的均值并转换为数组
        multi_covariance = np.asarray([st.covariance for st in stracks])    # 将所有轨迹的协方差转换为数组
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:          # 如果当前轨迹的状态不是 Tracked
                multi_mean[i][7] = 0                    # 将 multi_mean 的第 i 行第 7 个元素设置为 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)  # 使用共享的卡尔曼滤波器进行多目标预测
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):  # 遍历预测的均值和协方差
            stracks[i].mean = mean                      # 更新轨迹的均值
            stracks[i].covariance = cov                 # 更新轨迹的协方差

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):      # 使用单应矩阵更新轨迹位置和协方差，主要处理相机视角变化
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])         # 复制所有轨迹的均值并转换为数组
            multi_covariance = np.asarray([st.covariance for st in stracks])    # 将所有轨迹的协方差转换为数组

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)   # 计算 R 的克罗内克积
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):  # 遍历均值和协方差
                mean = R8x8.dot(mean)       # 计算新的均值
                mean[:2] += t               # 更新均值的前两元素
                cov = R8x8.dot(cov).dot(R8x8.transpose())  # 计算新的协方差

                stracks[i].mean = mean          # 更新轨迹的均值
                stracks[i].covariance = cov     # 更新轨迹的协方差

    def activate(self, kalman_filter, frame_id):    # 开始一个新的轨迹
        self.kalman_filter = kalman_filter          # 存储卡尔曼滤波器实例
        self.track_id = self.next_id()              # 获取下一个轨迹ID
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))  # 初始化均值和协方差

        self.tracklet_len = 0                       # 重置轨迹长度
        self.state = TrackState.Tracked             # 设置轨迹状态为 Tracked
        if frame_id == 1:                           # 如果当前帧ID为 1
            self.is_activated = True                # 设置激活状态为 True
        self.frame_id = frame_id                    # 存储当前帧ID
        self.start_frame = frame_id                 # 设置开始帧为当前帧ID

    def re_activate(self, new_track, frame_id, new_id=False):       # 用新的检测重新激活之前丢失的轨迹
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh))  # 更新均值和协方差
        self.tracklet_len = 0                       # 重置轨迹长度
        self.state = TrackState.Tracked             # 设置轨迹状态为 Tracked
        self.is_activated = True                    # 设置激活状态为 True
        self.frame_id = frame_id                    # 存储当前帧ID
        if new_id:
            self.track_id = self.next_id()          # 获取新的轨迹ID
        self.score = new_track.score                # 更新置信度得分
        self.cls = new_track.cls                    # 更新类别
        self.angle = new_track.angle                # 更新角度
        self.idx = new_track.idx                    # 更新索引

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.
        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        self.frame_id = frame_id    # 存储当前帧ID
        self.tracklet_len += 1      # 增加跟踪的目标数

        new_tlwh = new_track.tlwh   # 获取新的 tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh))  # 更新均值和协方差
        self.state = TrackState.Tracked         # 设置轨迹状态为 Tracked
        self.is_activated = True    # 设置激活状态为 True

        self.score = new_track.score    # 更新置信度得分
        self.cls = new_track.cls        # 更新类别
        self.angle = new_track.angle    # 更新角度
        self.idx = new_track.idx        # 更新索引

    def convert_coords(self, tlwh):         # 将边界框的左上角-宽-高格式转换为中心x-y-纵横比-高度格式
        return self.tlwh_to_xyah(tlwh)      # 调用 tlwh_to_xyah 函数进行转换

    @property
    def tlwh(self):         # 以左上角x, 左上角y, 宽, 高的格式获取当前位置的边界框 (top left x, top left y, width, height)
        if self.mean is None:               # 如果均值为 None
            return self._tlwh.copy()        # 返回复制的 _tlwh
        ret = self.mean[:4].copy()          # 复制均值的前四个元素
        ret[2] *= ret[3]                    # 更新宽度
        ret[:2] -= ret[2:] / 2              # 更新左上角坐标
        return ret

    @property
    def xyxy(self):         # 将边界框转换为(min x, min y, max x, max y)格式，即(左上, 右下)
        ret = self.tlwh.copy()              # 复制 tlwh
        ret[2:] += ret[:2]                  # 更新宽高为右下坐标
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):     # 将边界框转换为(center x, center y, aspect ratio, height)格式，其中纵横比为宽/高
        ret = np.asarray(tlwh).copy()       # 将 tlwh 转换为数组并复制
        ret[:2] += ret[2:] / 2              # 更新中心坐标
        ret[2] /= ret[3]                    # 计算纵横比
        return ret

    @property
    def xywh(self):         # 以中心x, 中心y, 宽, 高的格式获取当前位置的边界框。(center x, center y, width, height)
        ret = np.asarray(self.tlwh).copy()  # 复制 tlwh 并转换为数组
        ret[:2] += ret[2:] / 2              # 更新中心坐标
        return ret

    @property
    def xywha(self):        # 以中心x, 中心y, 宽, 高, 角度的格式获取当前位置的边界框。(center x, center y, width, height, angle)
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ `angle` attr not found, returning `xywh` instead.")  # 记录警告信息
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):       # 获取当前的跟踪结果
        coords = self.xyxy if self.angle is None else self.xywha        # 根据是否有角度选择坐标格式
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):     # 返回包含起始帧和结束帧以及轨迹ID的字符串表示
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"  # 返回字符串表示


class MoneyStateTracker:
    """
    StateTracker: 基于YOLOv8的目标检测和跟踪算法。
    该类负责初始化、更新和管理视频序列中检测到的目标的轨迹。它维护了跟踪、丢失和移除轨迹的状态，
    使用卡尔曼滤波器预测新的目标位置，并执行数据关联。

    属性：
        tracked_stracks (list[STrack]): 成功激活的轨迹列表。
        lost_stracks (list[STrack]): 丢失的轨迹列表。
        removed_stracks (list[STrack]): 被移除的轨迹列表。
        frame_id (int): 当前帧的ID。
        args (namespace): 命令行参数。
        max_time_lost (int): 轨迹被认为是“丢失”的最大帧数。
        kalman_filter (object): 卡尔曼滤波器对象。

    方法：
        update(results, img=None): 使用新检测更新目标跟踪器。
        get_kalmanfilter(): 返回用于跟踪边界框的卡尔曼滤波器对象。
        init_track(dets, scores, cls, img=None): 使用检测初始化目标跟踪。
        get_dists(tracks, detections): 计算轨迹和检测之间的距离。
        multi_predict(tracks): 预测轨迹的位置。
        reset_id(): 重置STrack的ID计数器。
        joint_stracks(tlista, tlistb): 合并两个STrack列表。
        sub_stracks(tlista, tlistb): 从第一个列表中过滤出第二个列表中存在的STrack。
        remove_duplicate_stracks(stracksa, stracksb): 基于IoU移除重复的STrack。
    """

    def __init__(self, args, frame_rate=30):
        """初始化一个YOLOv8对象，以给定的参数和帧速率跟踪目标。"""
        self.tracked_stracks = []  # 存储成功激活的轨迹
        self.lost_stracks = []  # 存储丢失的轨迹
        self.removed_stracks = []  # 存储被移除的轨迹

        self.frame_id = 0  # 当前帧的ID
        self.args = args  # 命令行参数
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)  # 计算轨迹被认为是“丢失”的最大帧数
        self.kalman_filter = self.get_kalmanfilter()  # 获取卡尔曼滤波器对象
        self.reset_id()  # 重置轨迹ID计数器

    def update(self, results, img=None):
        """使用新检测更新目标跟踪器，并返回跟踪目标的边界框。"""
        self.frame_id += 1  # 增加当前帧ID
        activated_stracks = []  # 存储激活的轨迹
        refind_stracks = []  # 存储重新找到的轨迹
        lost_stracks = []  # 存储丢失的轨迹
        removed_stracks = []  # 存储被移除的轨迹

        scores = results.conf  # 获取检测结果的置信度
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh  # 获取检测框
        # 添加索引
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls  # 获取类别

        remain_inds = scores >= self.args.track_high_thresh  # 获取置信度大于高阈值的检测框索引
        inds_low = scores > self.args.track_low_thresh  # 获取置信度大于低阈值的检测框索引
        inds_high = scores < self.args.track_high_thresh  # 获取置信度小于高阈值的检测框索引

        inds_second = inds_low & inds_high  # 获取置信度介于低阈值和高阈值之间的检测框索引
        dets_second = bboxes[inds_second]  # 获取低置信度检测框
        dets = bboxes[remain_inds]  # 获取高置信度检测框
        scores_keep = scores[remain_inds]  # 获取高置信度得分
        scores_second = scores[inds_second]  # 获取低置信度得分
        cls_keep = cls[remain_inds]  # 获取高置信度类别
        cls_second = cls[inds_second]  # 获取低置信度类别

        detections = self.init_track(dets, scores_keep, cls_keep, img)  # 初始化高置信度轨迹
        # 将新检测到的轨迹添加到 tracked_stracks 中
        unconfirmed = []  # 存储未确认的轨迹
        tracked_stracks = []  # 存储已跟踪的轨迹
        for track in self.tracked_stracks:  # 遍历所有已跟踪的轨迹
            if not track.is_activated:  # 如果轨迹未激活
                unconfirmed.append(track)  # 添加到未确认轨迹列表
            else:
                tracked_stracks.append(track)  # 添加到已跟踪轨迹列表
        # 步骤2：第一次关联，高置信度检测框
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)  # 合并已跟踪轨迹和丢失轨迹
        # 使用卡尔曼滤波器预测当前位置
        self.multi_predict(strack_pool)  # 对多个轨迹进行预测
        if hasattr(self, "gmc") and img is not None:  # 如果存在全局运动补偿（gmc）且图像不为空
            warp = self.gmc.apply(img, dets)  # 应用全局运动补偿
            STrack.multi_gmc(strack_pool, warp)  # 使用全局运动补偿更新轨迹
            STrack.multi_gmc(unconfirmed, warp)  # 使用全局运动补偿更新未确认的轨迹

        dists = self.get_dists(strack_pool, detections)  # 计算轨迹和检测框之间的距离
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)  # 使用匈牙利算法进行数据关联

        for itracked, idet in matches:  # 遍历所有匹配
            track = strack_pool[itracked]  # 获取匹配的轨迹
            det = detections[idet]  # 获取匹配的检测框
            if track.state == TrackState.Tracked:  # 如果轨迹状态为已跟踪
                track.update(det, self.frame_id)  # 更新轨迹
                activated_stracks.append(track)  # 添加到激活轨迹列表
            else:
                track.re_activate(det, self.frame_id, new_id=False)  # 重新激活轨迹
                refind_stracks.append(track)  # 添加到重新找到的轨迹列表
        # 步骤3：第二次关联，低置信度检测框与未跟踪的轨迹进行关联
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)  # 初始化低置信度轨迹
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]  # 获取未跟踪的已跟踪轨迹
        dists = matching.iou_distance(r_tracked_stracks, detections_second)  # 计算轨迹和低置信度检测框之间的IoU距离
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)  # 使用匈牙利算法进行数据关联
        for itracked, idet in matches:  # 遍历所有匹配
            track = r_tracked_stracks[itracked]  # 获取匹配的轨迹
            det = detections_second[idet]  # 获取匹配的检测框
            if track.state == TrackState.Tracked:  # 如果轨迹状态为已跟踪
                track.update(det, self.frame_id)  # 更新轨迹
                activated_stracks.append(track)  # 添加到激活轨迹列表
            else:
                track.re_activate(det, self.frame_id, new_id=False)  # 重新激活轨迹
                refind_stracks.append(track)  # 添加到重新找到的轨迹列表

        for it in u_track:  # 遍历所有未匹配的已跟踪轨迹
            track = r_tracked_stracks[it]  # 获取未匹配的已跟踪轨迹
            if track.state != TrackState.Lost:  # 如果轨迹状态不是丢失
                track.mark_lost()  # 标记为丢失
                lost_stracks.append(track)  # 添加到丢失轨迹列表
        # 处理未确认的轨迹，通常是只有一帧的轨迹
        detections = [detections[i] for i in u_detection]  # 获取未匹配的检测框
        dists = self.get_dists(unconfirmed, detections)  # 计算未确认轨迹和检测框之间的距离
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)  # 使用匈牙利算法进行数据关联
        for itracked, idet in matches:  # 遍历所有匹配
            unconfirmed[itracked].update(detections[idet], self.frame_id)  # 更新未确认轨迹
            activated_stracks.append(unconfirmed[itracked])  # 添加到激活轨迹列表
        for it in u_unconfirmed:  # 遍历所有未匹配的未确认轨迹
            track = unconfirmed[it]  # 获取未确认轨迹
            track.mark_removed()  # 标记为移除
            removed_stracks.append(track)  # 添加到移除轨迹列表
        # 步骤4：初始化新轨迹
        for inew in u_detection:  # 遍历所有未匹配的检测框
            track = detections[inew]  # 获取检测框
            if track.score < self.args.new_track_thresh:  # 如果置信度低于新轨迹阈值
                continue  # 跳过
            track.activate(self.kalman_filter, self.frame_id)  # 激活新轨迹
            activated_stracks.append(track)  # 添加到激活轨迹列表
        # 步骤5：更新状态
        for track in self.lost_stracks:  # 遍历所有丢失轨迹
            if self.frame_id - track.end_frame > self.max_time_lost:  # 如果丢失时间超过最大丢失时间
                track.mark_removed()  # 标记为移除
                removed_stracks.append(track)  # 添加到移除轨迹列表

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]  # 保留所有已跟踪的轨迹
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)  # 合并已跟踪轨迹和激活轨迹
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)  # 合并已跟踪轨迹和重新找到的轨迹
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)  # 从丢失轨迹中移除已跟踪的轨迹
        self.lost_stracks.extend(lost_stracks)  # 添加新的丢失轨迹
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)  # 从丢失轨迹中移除已移除的轨迹
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)  # 移除重复轨迹
        self.removed_stracks.extend(removed_stracks)  # 添加新的移除轨迹
        if len(self.removed_stracks) > 1000:  # 如果移除轨迹超过1000
            self.removed_stracks = self.removed_stracks[-999:]  # 保留最新的999个移除轨迹

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)  # 返回已激活轨迹的结果

    def get_kalmanfilter(self):
        """返回用于跟踪边界框的卡尔曼滤波器对象。"""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """使用检测结果和得分初始化目标跟踪。"""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []

    def get_dists(self, tracks, detections):
        """使用IoU计算轨迹和检测之间的距离，并融合得分。"""
        dists = matching.iou_distance(tracks, detections)  # 计算轨迹和检测框之间的IoU距离
        dists = matching.fuse_score(dists, detections)  # 融合得分
        return dists  # 返回距离矩阵

    def multi_predict(self, tracks):
        """使用YOLOv8网络预测轨迹的位置。"""
        STrack.multi_predict(tracks)  # 调用STrack类的multi_predict方法

    @staticmethod
    def reset_id():
        """重置STrack的ID计数器。"""
        STrack.reset_id()  # 调用STrack类的reset_id方法

    def reset(self):
        """重置跟踪器。"""
        self.tracked_stracks = []  # 清空已跟踪轨迹列表
        self.lost_stracks = []  # 清空丢失轨迹列表
        self.removed_stracks = []  # 清空移除轨迹列表
        self.frame_id = 0  # 重置帧ID
        self.kalman_filter = self.get_kalmanfilter()  # 获取新的卡尔曼滤波器对象
        self.reset_id()  # 重置ID计数器

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """合并两个STrack列表。"""
        exists = {}  # 用于存储已经存在的轨迹ID
        res = []  # 结果列表
        for t in tlista:  # 遍历第一个列表
            exists[t.track_id] = 1  # 将轨迹ID标记为已存在
            res.append(t)  # 添加到结果列表
        for t in tlistb:  # 遍历第二个列表
            tid = t.track_id  # 获取轨迹ID
            if not exists.get(tid, 0):  # 如果轨迹ID不存在
                exists[tid] = 1  # 将轨迹ID标记为已存在
                res.append(t)  # 添加到结果列表
        return res  # 返回合并后的列表

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """从第一个列表中过滤出第二个列表中存在的STrack。"""
        track_ids_b = {t.track_id for t in tlistb}  # 获取第二个列表中的所有轨迹ID
        return [t for t in tlista if t.track_id not in track_ids_b]  # 返回第一个列表中不在第二个列表中的轨迹

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """移除重复的STrack，基于非最大IoU距离。"""
        pdist = matching.iou_distance(stracksa, stracksb)  # 计算两个列表之间的IoU距离
        pairs = np.where(pdist < 0.15)  # 找到IoU距离小于0.15的匹配对
        dupa, dupb = [], []  # 存储重复的轨迹索引
        for p, q in zip(*pairs):  # 遍历所有匹配对
            timep = stracksa[p].frame_id - stracksa[p].start_frame  # 计算第一个列表中的轨迹生存时间
            timeq = stracksb[q].frame_id - stracksb[q].start_frame  # 计算第二个列表中的轨迹生存时间
            if timep > timeq:  # 如果第一个轨迹生存时间更长
                dupb.append(q)  # 将第二个轨迹标记为重复
            else:
                dupa.append(p)  # 将第一个轨迹标记为重复
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]  # 移除第一个列表中的重复轨迹
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]  # 移除第二个列表中的重复轨迹
        return resa, resb  # 返回去重后的两个列表
