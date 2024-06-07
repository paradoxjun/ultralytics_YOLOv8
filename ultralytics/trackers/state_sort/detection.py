import numpy as np


class Detection:
    """
    定义一张图片中一个检测框的数据结构。
    xywh：deepsort中的tlwh，这里取名与YOLOv8统一
    label: 检测框的标签
    confidence: 检测框的置信度
    feature：检测框的标签，这里默认为None，因为有部分目标不需要使用外观特征区分。
    """
    def __init__(self, tlwh, label, confidence, feature=None):
        self.tlwh = tlwh                        # xywh检测框             #
        self.label = label                      # 标签
        self.confidence = confidence            # 置信度
        if feature is not None:                 # 需要用特征再使用
            self.feature = np.asarray(feature, dtype=np.float32)
        else:
            self.feature = None

    def to_tlbr(self):
        # tlwh --> tlbr(xyxy)。（注意区分：tlwh是左上角坐标，xywh是中心点坐标）
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        # xywh --> wyah
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


if __name__ == '__main__':
    from ultralytics.utils.ops import xyxy2ltwh
    def run_tests():
        # 测试1：初始化和属性检查
        xywh = np.array([100, 50, 20, 10], dtype=np.float32)
        label = "person"
        confidence = 0.9
        feature = [0.1, 0.2, 0.3]

        det = Detection(xywh, label, confidence, feature)
        det_no_feature = Detection(xywh, label, confidence)
        assert det_no_feature.feature is None, "feature属性应为None"

        print("测试1通过: 初始化和属性检查")

        # 测试2：to_xyxy 方法
        expected_xyxy = np.array([100, 50, 120, 60])
        print(xyxy2ltwh(expected_xyxy))
        assert np.array_equal(det.to_tlbr(), expected_xyxy), f"to_xyxy方法不正确: {det.to_xyxy()} != {expected_xyxy}"

        print("测试2通过: to_xyxy 方法")

        # 测试3：to_xyah 方法
        expected_xyah = np.array([110, 55, 2, 10], dtype=np.float32)
        assert np.array_equal(det.to_xyah(), expected_xyah), f"to_xyah方法不正确: {det.to_xyah()} != {expected_xyah}"

        print("测试3通过: to_xyah 方法")

    run_tests()
