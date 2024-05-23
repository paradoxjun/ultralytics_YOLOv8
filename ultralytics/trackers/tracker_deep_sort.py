"""
代码参考DeepSORT_YOLOv5_Pytorch
"""
from ultralytics.utils.torch_utils import time_sync
from ultralytics.utils import yaml_load
from ultralytics.trackers.deep_sort import build_tracker
from ultralytics.task_bank.predict import BankDetectionPredictor
from ultralytics.task_bank.utils import draw_boxes, get_config

import os
import time
import cv2
import torch
import torch.backends.cudnn as cudnn

import sys

currentUrl = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(currentUrl)))

cudnn.benchmark = True

model = '/home/chenjun/code/ultralytics_YOLOv8/weights/yolov8m.pt'
model_2 = '/home/chenjun/code/ultralytics_YOLOv8/runs/detect/train_bank_05_21_m/weights/best.pt'

overrides = {"task": "detect",
             "mode": "predict",
             "model": model,
             "verbose": False,
             "classes": 0}

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": model_2,
               "verbose": False,
               }

predictor = BankDetectionPredictor(overrides=overrides)
predictor_2 = BankDetectionPredictor(overrides=overrides_2)


class VideoTracker:
    def __init__(self, track_cfg):
        self.track_cfg = yaml_load(track_cfg)
        self.deepsort_arg = get_config(self.track_cfg["config_deep_sort"])
        self.predictor = predictor
        self.predictor_2 = predictor_2
        self.img_size = self.track_cfg["imgsz"]

        if self.track_cfg['display']['turn_on']:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)

            if self.track_cfg['display']["width"] >= 32 and self.track_cfg['display']["height"] >= 32:
                cv2.resizeWindow("test", self.track_cfg['display']["width"], self.track_cfg['display']["height"])

        if self.track_cfg["camera"] != -1:
            print("Using webcam " + str(self.track_cfg["camera"]))
            self.vdo = cv2.VideoCapture(self.track_cfg["camera"])
        else:
            self.vdo = cv2.VideoCapture()

        use_cuda = self.track_cfg["device"] != "cpu" and torch.cuda.is_available()
        self.deepsort = build_tracker(self.deepsort_arg, use_cuda=use_cuda)
        self.names = self.track_cfg["class_name"]

        print("Tracker init finished...")

    def __enter__(self):
        # ************************* Load video from camera *************************
        if self.track_cfg["camera"] != -1:
            print('Camera ...')
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"

        # ************************* Load video from file *************************
        else:
            assert os.path.isfile(self.track_cfg["input_path"]), "Path error"
            self.vdo.open(self.track_cfg["input_path"])
            assert self.vdo.isOpened()
            print('Done. Load video file ', self.track_cfg["input_path"])

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ************************* create output *************************
        if self.track_cfg["save_path"]:
            os.makedirs(self.track_cfg["save_path"], exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.track_cfg["save_path"], "results.mp4")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.track_cfg["fourcc"])
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
            print('Done. Create output file ', self.save_video_path)

        if self.track_cfg["save_txt"]:
            os.makedirs(self.track_cfg["save_txt"], exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_out = None
        while self.vdo.grab():
            # Inference *********************************************************************
            t0 = time.time()
            _, img0 = self.vdo.retrieve()

            if idx_frame % self.track_cfg["vid_stride"] == 0:
                outputs, cls, yt, st = self.image_track(img0)  # (#ID, 5) x1,y1,x2,y2,id
                last_out = outputs
                yolo_time.append(yt)
                sort_time.append(st)
                print('Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
            else:
                outputs = last_out  # directly use prediction in last frames
            t1 = time.time()
            avg_fps.append(t1 - t0)

            # post-processing ***************************************************************
            # visualize bbox  ********************************
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                img0 = draw_boxes(img0, bbox_xyxy, cls, identities)  # BGR

                # add FPS information on output video
                text_scale = max(1, img0.shape[1] // 1600)
                cv2.putText(img0, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)),
                            (20, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

            # display on window ******************************
            if self.track_cfg["display"]:
                cv2.imshow("test", img0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    break

            # save to video file *****************************
            if self.track_cfg["save_path"]:
                self.writer.write(img0)

            if self.track_cfg["save_txt"]:
                with open(self.track_cfg["save_txt"] + str(idx_frame).zfill(4) + '.txt', 'a') as f:
                    for i in range(len(outputs)):
                        x1, y1, x2, y2, idx = outputs[i]
                        f.write('{}\t{}\t{}\t{}\t{}\n'.format(x1, y1, x2, y2, idx))

            idx_frame += 1

        print('Avg YOLO time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),
                                                                      sum(sort_time) / len(sort_time)))
        t_end = time.time()
        print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

    def image_track(self, img):
        # Detection time *********************************************************
        # Inference

        t1 = time_sync()
        pred = self.predictor(source=img)  # list: bz * [ (#obj, 6)]
        pred_2 = self.predictor_2(source=img)
        t2 = time_sync()

        # get all obj ************************************************************
        det = pred[0]  # for video, bz is
        det_2 = pred_2[0]
        if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls

            # Rescale boxes from img_size to original im0 size

            # Print results. statistics of number of each obj
            bbox_xywh = torch.cat((det.boxes.xywh, det_2.boxes.xywh)).cpu()
            confs = torch.cat((det.boxes.conf, det_2.boxes.conf)).cpu()
            cls = torch.cat((det.boxes.cls + 4, det_2.boxes.cls)).cpu()

            print(f"bbox_xywh: {bbox_xywh}, confs: {confs}, cls: {cls}")
            # ****************************** deepsort ****************************
            outputs = self.deepsort.update(bbox_xywh, confs, img)
            print(f"outputs: {outputs}")
            # (#ID, 5) x1,y1,x2,y2,track_ID
        else:
            outputs = torch.zeros((0, 5))
            cls = torch.zeros((0, 1))

        t3 = time.time()
        return outputs, cls, t2 - t1, t3 - t2


if __name__ == '__main__':
    model_cfg = '/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/default.yaml'
    track_cfg = '/home/chenjun/code/ultralytics_YOLOv8/ultralytics/cfg/bank_monitor/track.yaml'

    with VideoTracker(track_cfg=track_cfg) as vdo_trk:
        vdo_trk.run()
