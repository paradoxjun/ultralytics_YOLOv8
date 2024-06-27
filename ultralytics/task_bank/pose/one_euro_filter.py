import numpy as np


def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.1, beta=0.0, d_cutoff=0.1):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.zeros_like(self.x_prev) if dx0 is None else np.array(dx0, dtype=np.float32)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (np.array(x) - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # 确保原始输入为0的值在结果中也保持为0。0表示无效，平滑后会出问题。
        zero_rows = np.all(np.array(x) == 0, axis=1)
        x_hat[zero_rows] = 0

        # 如果 self.x_prev 中某一行全部为0，则 x_hat 中的这一行设为输入 x 对应的值
        prev_zero_rows = np.all(self.x_prev == 0, axis=1)
        x_hat[prev_zero_rows] = np.array(x)[prev_zero_rows]

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


if __name__ == '__main__':
    # 示例关键点数据
    initial_keypoints = [
        [100, 150], [200, 250], [300, 350]
    ]
    t0 = 0

    # 初始化一欧元滤波器
    keypoint_filter = OneEuroFilter(t0, initial_keypoints)

    # 模拟随时间变化的关键点数据
    timesteps = np.linspace(1, 5, num=5)
    keypoints_data = [
        [[102, 148], [198, 252], [305, 348]],
        [[104, 149], [197, 250], [310, 346]],
        [[101, 151], [196, 249], [308, 349]],
        [[103, 150], [195, 248], [307, 350]],
        [[105, 152], [194, 247], [306, 351]]
    ]

    # 平滑关键点数据
    for t, keypoints in zip(timesteps, keypoints_data):
        smoothed_keypoints = keypoint_filter(t, keypoints)
        print(f"Time {t}: {smoothed_keypoints}")
