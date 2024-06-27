from ultralytics.task_bank.byte_tracker_modify import BYTETracker
from ultralytics.task_bank.pose.ops import filter_boxes_ioa
from ultralytics.task_bank.pose.utils import get_upper_body_keypoint
from ultralytics.task_bank.pose.one_euro_filter import OneEuroFilter
from easydict import EasyDict

bytetrack_config = {
    'track_high_thresh': 0.5,   # threshold for the first association
    'track_low_thresh': 0.1,    # threshold for the second association
    'new_track_thresh': 0.6,    # threshold for init new track if the detection does not match any tracks
    'track_buffer': 30,         # buffer to calculate the time when to remove tracks
    'match_thresh': 0.8         # threshold for matching tracks
}

bytetrack_args = EasyDict(bytetrack_config)


class Person:
    def __init__(self):
        self.bytetrack = BYTETracker(bytetrack_args)
        self.filter = dict()    # 一欧元滤波器字典{track_id: filter}

    def update(self, idx_frame, pose):
        xyxy = pose.boxes.xyxy.cpu().numpy()
        conf = pose.boxes.conf.view(-1, 1).cpu().numpy()
        valid_index = filter_boxes_ioa(xyxy, conf)

        xyxy = xyxy[valid_index]
        conf = conf[valid_index]
        xywh = pose.boxes.xywh.cpu().numpy()[valid_index]
        cls = pose.boxes.cls.view(-1, 1).cpu().numpy()[valid_index]

        track_res = self.bytetrack.update(xywh, conf.reshape(-1), cls.reshape(-1))
        if track_res.shape[0] > 0:
            track_index = track_res[:, -1].astype(int)
        else:
            track_index = []

        keypoint_data = get_upper_body_keypoint(pose.keypoints.xy.cpu().numpy()[valid_index])
        # pre = keypoint_data.copy()
        for det in track_res:
            track_id, order = int(det[4]), int(det[-1])

            if track_id not in self.filter:
                self.filter[track_id] = OneEuroFilter(idx_frame, keypoint_data[order])
            else:
                keypoint_data[order] = self.filter[track_id](idx_frame, keypoint_data[order])

        # print(idx_frame)
        # print(pre - keypoint_data)
        return track_res, keypoint_data[track_index]
