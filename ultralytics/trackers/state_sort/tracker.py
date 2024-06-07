from __future__ import absolute_import
import numpy as np
import ultralytics.trackers.state_sort.linear_assignment as linear_assignment
import ultralytics.trackers.state_sort.iou_matching as iou_matching
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
from track import MoneyTrack


class MoneyTracker:
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, label=None, confs=None):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.label = label
        self.confs = confs

        self.kf = KalmanFilterXYAH()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        # 步骤1：基于 T-1 时刻，使用卡尔曼滤波对 T 时刻的状态进行预测。
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        pass

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(MoneyTrack(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
        self._next_id += 1

    def _match(self, detections):
        # 基于外观信息和马氏距离，计算卡尔曼滤波预测的tracks和当前时刻检测到的detections的代价矩阵
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            # 基于外观信息，计算tracks和detections的余弦距离代价矩阵
            cost_matrix = self.metric.distance(features, targets)

            # 基于马氏距离，过滤掉代价矩阵中一些不合适的项 (将其设置为一个较大的值)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices)

            return cost_matrix

        """
        KF predict 
            -- confirmed 
                Matching_Cascade (appearance feature + distance)
                    -- matched Tracks
                    -- unmatched tracks
                        -- 
                    -- unmatched detection
            -- unconfirmed 
        """

        # Split track set into confirmed and unconfirmed tracks. ********************************************
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if
                            t.is_confirmed()]  # confirmed: directly apply Matching_Cascade
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if
                              not t.is_confirmed()]  # unconfirmed: directly go to IOU match

        # Associate confirmed tracks using appearance features.(Matching_Cascade) ***************************
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU *****************
        # for IOU match: unconfirmed + u
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]  # 刚刚没有匹配上

        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        # IOU matching *************************************************************************************
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost, self.max_iou_distance, self.tracks,
            detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections
