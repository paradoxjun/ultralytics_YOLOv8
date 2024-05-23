from ultralytics import YOLO


class object_info:
    def __init__(self, id, xywhn):
        pass





action = {
    1: "从款箱拿钱",
    2: "放入点钞机",
    3: "放入另一边"
}


class StateMachine:
    def __init__(self):
        self.state_list = ["1", "2", "3", "11", "12", "13", "21", "22", "23", "31", "32", "33"]
        self.state_illegal = ["13"]

    @staticmethod
    def state_transition(state_pre, state_now):
        if state_pre is None:
            return state_now
        else:
            return state_pre[-1] + state_now

    def check_step(self, state_pre, state_now):
        state = self.state_transition(state_pre, state_now)
        if state in self.state_illegal:
            print(f"当前操作不规范！，缺少点钱步骤。执行的步骤为：{state}")
        else:
            print(f"当前正在执行步骤：{state}")


class GetAction:
    def __init__(self, data_path, model, show=False, save=False):
        self.data_path = data_path
        self.model = model
        self.show = show
        self.save = save
        model = YOLO(task="detect", model=self.model)
