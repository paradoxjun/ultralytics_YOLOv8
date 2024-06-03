import torch
import numpy as np
import cv2


def image_show(image, title="Image"):
    """
    读取并显示一张图片。YOLOv8中默认用cv2读取为BGR通道顺序。
    见 ultralytics/engine/predictor.py下 BasePredictor.preprocess()。
    Args:
        image: 路径 or np数组 or tensor张量。
        title: 显示窗口名。
    Returns: None
    """
    try:
        need_show = True                # 存在batch时，循环里显示，最后不显示
        if isinstance(image, str):      # 从文件路径读取图片
            img = cv2.imread(image)
            if not img:
                raise ValueError("Error path: Could not read the image.")
        elif isinstance(image, np.ndarray):     # image 是 ndarray
            img = image
        elif isinstance(image, torch.Tensor):   # image 是 PyTorch Tensor
            if image.dim() == 3:                # C,H,W to H,W,C
                img = image.permute(1, 2, 0).cpu().numpy() if image.shape[0] == 3 else image
            elif image.dim() == 4:              # B,C,H,W or B,H,W,C
                for i in range(image.size(0)):
                    img = image[i].permute(1, 2, 0).cpu().numpy() if image[i].shape[0] == 3 else image[i]
                    cv2.imshow(title + "_" + str(i + 1), img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    need_show = False
            else:
                raise ValueError(f"Error tensor: Tensor dimensions are incorrect. The d is {image.dim()}.")
        else:
            raise ValueError(f"Error format: Unsupported image type: {type(image)}.")

        if img.dtype == np.float32 or img.dtype == np.float64:  # 如果是浮点数
            if img.max() > 1:
                img = img.astype(np.uint8)
            else:
                img = (img * 255).astype(np.uint8)  # 假定范围是0到1，转换回0到255的范围

        if need_show:
            cv2.imshow(title, img)      # 使用 OpenCV 显示图片
            cv2.waitKey(0)              # 等待直到任何键被按下
            cv2.destroyAllWindows()     # 关闭所有 OpenCV 创建的窗口

    except Exception as e:
        print(f"Error: 读取图片发生错误：{e}")


if __name__ == '__main__':
    tensor_example = torch.rand(2, 3, 640, 640)  # 创建一个随机的四维Tensor
    image_show(tensor_example, "Random Tensor Image")
