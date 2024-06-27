from ultralytics.utils import yaml_load
from ultralytics.utils.torch_utils import time_sync
from ultralytics.task_bank.byte_tracker_modify import BYTETracker
from ultralytics.task_bank.pose.predict import PosePredictor
from ultralytics.task_bank.pose.utils import get_upper_body_keypoint, plot_keypoint
from ultralytics.task_bank.pose.ops import filter_boxes_ioa
from ultralytics.task_bank.pose.ren_filter_track_pose import Person
from ultralytics.task_bank.utils import resize_and_pad, get_config
from pathlib import Path
from datetime import datetime

import os
import time
import cv2
import numpy as np
import torch


class VideoTracker:
    def __init__(self, track_cfg, predictors):
        self.track_cfg = yaml_load(track_cfg)       # v8内置方法读取track.yaml文件为字典
        self.predictors = predictors                # 检测器列表
        byte_sort_config = get_config(self.track_cfg["config_byte_sort"])  # 读取byte_sort.yaml为EasyDict类
        self.bytesort_ren = BYTETracker(byte_sort_config)
        self.people = Person()
        self.save_dir = self.make_save_dir()

        print("INFO: Tracker init finished...")

    def make_save_dir(self):    # 创建保存文件的文件夹
        root_dir = Path(self.track_cfg["save_option"]["root"])      # 保存根路径

        if not root_dir.exists():   # 根路径一定要自己指定
            raise ValueError(f"设置存储根目录失败，不存在根路径：{root_dir}")

        save_dir = os.path.join(root_dir, self.track_cfg["save_option"]["dir"])     # 实际保存路径

        if os.path.exists(save_dir):    # 存在也保存到这里
            print(f"INFO: 当前保存路径 [{save_dir}] 已经存在。")
        else:
            os.makedirs(save_dir)
            print(f"INFO: 当前保存路径 [{save_dir}] 不存在，已创建。")

        for sub_dir in ["image_plot", "txt_pose", "txt_xyxy"]:     # 分目录保存不同结果
            sub = os.path.join(save_dir, sub_dir)
            if not os.path.exists(sub):
                os.makedirs(sub)

        return save_dir

    def get_video(self, video_path=None):           # 获取视频流（优先级：指定文件路径 > 摄像头 > 配置文件路径）
        if video_path is None:                      # 读取输入
            if self.track_cfg["camera"] != -1:      # 使用摄像头获取视频
                print("INFO: Using webcam " + str(self.track_cfg["camera"]))
                v_cap = cv2.VideoCapture(self.track_cfg["camera"])
            else:                                           # 使用文件路径获取
                assert os.path.isfile(self.track_cfg["input_path"]), "Video path in *.yaml is error. "
                v_cap = cv2.VideoCapture(self.track_cfg["input_path"])
        else:
            assert os.path.isfile(video_path), "Video path in method get_video() is error. "
            v_cap = cv2.VideoCapture(video_path)

        return v_cap

    def image_pose(self, idx_frame, img):     # 生成追踪目标的id
        t1 = time_sync()
        person_pose = self.predictors(source=img)[0].cpu()  # 官方预训练权重，检测人的姿态

        det_res, keypoint_data = self.people.update(idx_frame, person_pose)

        # xyxy = person_pose.boxes.xyxy.cpu().numpy()
        # xywh = person_pose.boxes.xywh.cpu().numpy()
        # conf = person_pose.boxes.conf.view(-1, 1).cpu().numpy()
        # cls = person_pose.boxes.cls.view(-1, 1).cpu().numpy()
        # data = person_pose.keypoints.data.cpu().numpy()
        #
        # valid_index = filter_boxes_ioa(xyxy, conf)
        #
        # det_res = np.concatenate([xyxy[valid_index], cls[valid_index], conf[valid_index]], axis=1)
        # print(det_res)
        # # x1,y1,x2,y2,track_id,confs,label,order
        # det_res = self.bytesort_ren.update(xywh[valid_index], conf[valid_index].reshape(-1), cls[valid_index].reshape(-1))
        # if det_res.shape[0] > 0:
        #     track_index = det_res[:, -1].astype(int)
        #     keypoint_data = get_upper_body_keypoint(data[valid_index][track_index])
        # else:
        #     keypoint_data = get_upper_body_keypoint(data[valid_index])
        # print(det_res)

        t2 = time_sync()

        return keypoint_data, det_res, t2 - t1

    def plot_pose(self, img, keypoint_data, det_res):
        print('*' * 20)
        img = plot_keypoint(img, keypoint_data)

        for i, box in enumerate(det_res):
            x1, y1, x2, y2, track_id = list(map(int, box[:5]))       # 将结果均映射为整型
            confidence = float(box[5])

            color = (255, 0, 0)    # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)    # 基本矩形检测框
            label_text = f'ren:{round(confidence, 2)}, id:{track_id}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img

    def save_pose(self, i=0, img=None, pose=None, det_res=None):    # 传入帧数，绘制结果，追踪结果，检测结果
        if not self.track_cfg["save_option"]["save"]:
            return

        if img is not None and self.track_cfg["save_option"]["img"]:
            img_save = os.path.join(self.save_dir, "image_plot", "img_" + str(i).zfill(5) + ".jpg")
            cv2.imwrite(img_save, img)

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{img_save}].")

        if pose is not None and self.track_cfg["save_option"]["txt"]:
            sort_save = os.path.join(self.save_dir, "txt_pose", "sort_" + str(i).zfill(5) + ".txt")
            np.savetxt(sort_save, pose, fmt='%.6f')

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{sort_save}].")

        if det_res is not None and self.track_cfg["save_option"]["txt"]:
            xyxy_save = os.path.join(self.save_dir, "txt_xyxy", "xyxy_" + str(i).zfill(5) + ".txt")
            np.savetxt(xyxy_save, det_res, fmt='%.6f')

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{xyxy_save}].")

    def det_pose_pipline(self, video_path=None):    # 读取视频，检测，追踪，绘制，保存全流程
        cap = self.get_video(video_path=video_path)
        if not cap.isOpened():
            print("INFO: 无法获取视频，退出！")
            exit()

        if self.track_cfg["video_shape"][0] > 32 and self.track_cfg["video_shape"][1] > 32:
            width = self.track_cfg["video_shape"][0]
            height = self.track_cfg["video_shape"][1]
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 获取视频的宽度、高度和帧率
        if self.track_cfg["save_option"]["save"]:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
            video_plot_save_path = os.path.join(self.save_dir, "video_plot_" + current_time + ".mp4")
            out = cv2.VideoWriter(video_plot_save_path, fourcc, fps, (width, height))   # 初始化视频写入器

        yolo_time, pose_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_pose = None    # 跳过的帧不绘制，会导致检测框闪烁
        last_det = None

        while idx_frame < 1800:
            ret, frame = cap.read()
            t0 = time.time()

            if not ret or cv2.waitKey(1) & 0xFF == ord('q'):    # 结束 或 按 'q' 键退出
                break

            if self.track_cfg["video_shape"][0] > 32 and self.track_cfg["video_shape"][1] > 32:
                frame = resize_and_pad(frame, self.track_cfg["video_shape"])

            if idx_frame % self.track_cfg["vid_stride"] == 0:
                pose, det_res, cost_time = self.image_pose(idx_frame, frame)       # 追踪结果，检测结果，消耗时间
                last_pose = pose
                last_det = det_res
                yolo_time.append(cost_time)          # yolo推理时间

                if self.track_cfg["verbose"]:
                    print('INFO: Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, *cost_time))

                plot_img = self.plot_pose(frame, pose, det_res)         # 绘制加入追踪框的图片
                self.save_pose(idx_frame, plot_img, pose, det_res)      # 保存跟踪结果
            else:
                plot_img = self.plot_pose(frame, last_pose, last_det)   # 帧间隔小，物体运动幅度小，就用上一次结果

            if self.track_cfg["save_option"]["save"]:
                out.write(plot_img)     # 将处理后的帧写入输出视频

            t1 = time.time()
            avg_fps.append(t1 - t0)     # 第1帧包含了模型加载时间要删除

            # add FPS information on output video
            text_scale = max(1, plot_img.shape[1] // 1000)
            cv2.putText(plot_img, 'frame: %d fps: %.2f ' % (idx_frame, (len(avg_fps) - 1) / (sum(avg_fps[1:]) + 1e-6)),
                        (10, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=1)
            cv2.imshow('Frame', plot_img)

            idx_frame += 1

        cap.release()   # 释放读取资源
        if self.track_cfg["save_option"]["save"]:
            out.release()  # 释放写入资源
        cv2.destroyAllWindows()

        avg_yolo_t = sum(yolo_time[1:]) / (len(yolo_time) - 1)
        print(f'INFO: Avg YOLO time ({avg_yolo_t:.3f}s) per frame')
        total_t, avg_fps = time.time() - t_start, (len(avg_fps) - 1) / (sum(avg_fps[1:]) + 1e-6)
        print('INFO: Total Frame: %d, Total time (%.3fs), Avg fps (%.3f)' % (idx_frame, total_t, avg_fps))


if __name__ == '__main__':
    track_cfg = r'/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/bank_monitor/pose.yaml'

    overrides = {"task": "pose",
                 "mode": "predict",
                 "model": r'../../../weights/yolov8m-pose.pt',
                 "verbose": False,
                 }

    predictor = PosePredictor(overrides=overrides)

    vt = VideoTracker(track_cfg=track_cfg, predictors=predictor)
    vt.det_pose_pipline()
