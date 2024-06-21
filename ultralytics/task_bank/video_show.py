from ultralytics.utils.torch_utils import time_sync
from ultralytics.utils import yaml_load
from ultralytics.utils.plotting import colors as set_color
from ultralytics.task_bank.byte_tracker_modify import BYTETracker
from ultralytics.task_bank.predict import BankDetectionPredictor
from ultralytics.task_bank.utils import get_config, resize_and_pad, transform_and_concat_tensors
from pathlib import Path
from datetime import datetime

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

currentUrl = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(currentUrl)))

cudnn.benchmark = True


class VideoTracker:
    def __init__(self, track_cfg, predictors):
        self.track_cfg = yaml_load(track_cfg)       # v8内置方法读取track.yaml文件为字典
        self.predictors = predictors                # 检测器列表
        self.bytesort_arg = get_config(self.track_cfg["config_byte_sort"])      # 读取xxxx_sort.yaml为EasyDict类
        self.bytesort = BYTETracker(self.bytesort_arg)     # 实例化xxxx_sort类
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

        for sub_dir in ["image_plot", "txt_track", "txt_xyxy", "txt_xywh", "txt_xywhn"]:     # 分目录保存不同结果
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

    def image_track(self, img, idx_frame=0):     # 生成追踪目标的id
        t1 = time_sync()

        det_res_xyxy = []
        det_res_xywh = []
        det_res_conf = []
        det_res_cls = []
        det_res_names = []
        for predictor in self.predictors:
            det_res = predictor(source=img)[0]
            det_res_xyxy.append(det_res.boxes.xyxy)
            det_res_xywh.append(det_res.boxes.xywh)
            det_res_conf.append(det_res.boxes.conf)
            det_res_cls.append(det_res.boxes.cls)
            det_res_names.append(det_res.names)

        bbox_xywh = torch.cat(det_res_xywh).cpu().numpy()
        bbox_xyxy = torch.cat(det_res_xyxy).cpu().numpy()
        confs = torch.cat(det_res_conf).cpu().numpy()
        cls = transform_and_concat_tensors(det_res_cls, det_res_names, self.track_cfg['class_name']).cpu().numpy()

        t2 = time_sync()

        if len(cls) > 0:
            bytesort_outputs = self.bytesort.update(bbox_xywh, confs, cls)   # x1,y1,x2,y2,track_id,confs,label,order
        else:
            bytesort_outputs = np.zeros((0, 8), dtype=np.float32)               # 或者返回空

        t3 = time.time()
        return bytesort_outputs, [bbox_xywh, bbox_xyxy, cls, confs], [t2 - t1, t3 - t2]

    def plot_track(self, img, sort_output):      # 在一帧上绘制检测结果（类别+置信度+追踪ID）
        for i, box in enumerate(sort_output):
            x1, y1, x2, y2, track_id, _, label, _ = list(map(int, box))       # 将结果均映射为整型
            confidence = float(box[5])

            # 设置显示内容：文本框左上角为“标签名：置信度”，右上角为“跟踪id”，文本框颜色由类别决定
            color = set_color(label * 2)    # 设置颜色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)    # 基本矩形检测框
            label_text = f'{self.track_cfg["class_name"][label]}:{round(confidence, 2)}'  # 左上角标签+置信度文字
            cv2.putText(img, label_text, (x1 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            track_text = f"ID: {track_id}"  # 右上角追踪ID文字
            cv2.putText(img, track_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img

    def save_track(self, i=0, img=None, sort_output=None, det_res=None):    # 传入帧数，绘制结果，追踪结果，检测结果
        if not self.track_cfg["save_option"]["save"]:
            return

        if img is not None and self.track_cfg["save_option"]["img"]:
            img_save = os.path.join(self.save_dir, "image_plot", "img_" + str(i).zfill(5) + ".jpg")
            cv2.imwrite(img_save, img)

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{img_save}].")

        if sort_output is not None and self.track_cfg["save_option"]["txt"]:
            sort_save = os.path.join(self.save_dir, "txt_track", "sort_" + str(i).zfill(5) + ".txt")
            np.savetxt(sort_save, sort_output, fmt='%d')

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{sort_save}].")

        if det_res is not None and self.track_cfg["save_option"]["txt"]:
            xywh, xyxy, cls, confs = det_res    # torch.Size([n, 4]) torch.Size([n, 4]) torch.Size([n]) torch.Size([n])
            xywh_save = os.path.join(self.save_dir, "txt_xywh", "xywh_" + str(i).zfill(5) + ".txt")
            xyxy_save = os.path.join(self.save_dir, "txt_xyxy", "xyxy_" + str(i).zfill(5) + ".txt")
            xywh_np = torch.cat([xywh, cls.view(-1, 1), confs.view(-1, 1)], dim=1).numpy()
            xyxy_np = torch.cat([xyxy, cls.view(-1, 1), confs.view(-1, 1)], dim=1).numpy()
            np.savetxt(xywh_save, xywh_np, fmt='%.6f')

            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{xywh_save}].")
            np.savetxt(xyxy_save, xyxy_np, fmt='%.6f')
            if self.track_cfg["verbose"]:
                print(f"INFO: 已经保存[{xyxy_save}].")

    def det_track_pipline(self, video_path=None):    # 读取视频，检测，追踪，绘制，保存全流程
        cap = self.get_video(video_path=video_path)
        if not cap.isOpened():
            print("INFO: 无法获取视频，退出！")
            exit()

        # 获取视频的宽度、高度和帧率
        if self.track_cfg["save_option"]["save"]:
            if self.track_cfg["video_shape"][0] > 32 and self.track_cfg["video_shape"][1] > 32:
                width = self.track_cfg["video_shape"][0]
                height = self.track_cfg["video_shape"][1]
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
            video_plot_save_path = os.path.join(self.save_dir, "video_plot_" + current_time + ".mp4")
            out = cv2.VideoWriter(video_plot_save_path, fourcc, fps, (width, height))   # 初始化视频写入器

        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_sort = None    # 跳过的帧不绘制，会导致检测框闪烁

        while True:
            ret, frame = cap.read()
            t0 = time.time()

            if not ret or cv2.waitKey(1) & 0xFF == ord('q'):    # 结束 或 按 'q' 键退出
                break

            if self.track_cfg["video_shape"][0] > 32 and self.track_cfg["video_shape"][1] > 32:
                frame = resize_and_pad(frame, self.track_cfg["video_shape"])

            if idx_frame % self.track_cfg["vid_stride"] == 0:
                sort, det_res, cost_time = vt.image_track(frame, idx_frame)       # 追踪结果，检测结果，消耗时间
                last_sort = sort
                yolo_time.append(cost_time[0])          # yolo推理时间
                sort_time.append(cost_time[1])          # sort跟踪时间

                if self.track_cfg["verbose"]:
                    print('INFO: Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, *cost_time))

                plot_img = vt.plot_track(frame, sort)                  # 绘制加入追踪框的图片
                vt.save_track(idx_frame, plot_img, sort, det_res)      # 保存跟踪结果
            else:
                plot_img = vt.plot_track(frame, last_sort)              # 帧间隔小，物体运动幅度小，就用上一次结果

            if self.track_cfg["save_option"]["save"]:
                out.write(plot_img)         # 将处理后的帧写入输出视频

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

        avg_yolo_t, avg_sort_t = sum(yolo_time[1:]) / (len(yolo_time) - 1), sum(sort_time[1:]) / (len(sort_time) - 1)
        print(f'INFO: Avg YOLO time ({avg_yolo_t:.3f}s), Sort time ({avg_sort_t:.3f}s) per frame')
        total_t, avg_fps = time.time() - t_start, (len(avg_fps) - 1) / (sum(avg_fps[1:]) + 1e-6)
        print('INFO: Total Frame: %d, Total time (%.3fs), Avg fps (%.3f)' % (idx_frame, total_t, avg_fps))


if __name__ == '__main__':
    track_cfg = r'/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/bank_monitor/track.yaml'
    overrides_1 = {"task": "detect",
                   "mode": "predict",
                   "model": r'/home/chenjun/code/ultralytics_YOLOv8/weights/yolov8m.pt',
                   "verbose": False,
                   "classes": [0],
                   "iou": 0.4
                   }

    overrides_2 = {"task": "detect",
                   "mode": "predict",
                   "model": r'/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_06_20_m_ycj_kx/weights/best.pt',
                   "verbose": False,
                   "classes": [0, 1, 2]
                   }

    overrides_3 = {"task": "detect",
                   "mode": "predict",
                   "model": r'/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_06_20_m_qian/weights/best.pt',
                   "verbose": False,
                   "classes": [3]
                   }

    overrides_4 = {"task": "detect",
                   "mode": "predict",
                   "model": r'/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_06_20_m_zbm/weights/best.pt',
                   "verbose": False,
                   "classes": [4]
                   }

    predictor_1 = BankDetectionPredictor(overrides=overrides_1)
    predictor_2 = BankDetectionPredictor(overrides=overrides_2)
    predictor_3 = BankDetectionPredictor(overrides=overrides_3)
    predictor_4 = BankDetectionPredictor(overrides=overrides_4)
    predictors = [predictor_1, predictor_2, predictor_3, predictor_4]

    vt = VideoTracker(track_cfg=track_cfg, predictors=predictors)
    vt.det_track_pipline()
