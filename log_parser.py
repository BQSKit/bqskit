import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LogData:

    def __init__(self, run_time: float, idle_time: float, comms_time: float, num_tasks: float, timeline: list[tuple[str, float]]):
        self.run_time = run_time
        self.idle_time = idle_time
        self.comms_time = comms_time
        self.num_tasks = num_tasks
        self.timeline = timeline

    def __str__(self) -> str:
        return f"Run Time: {self.run_time} Idle Time: {self.idle_time} Misc Time: {self.comms_time} Num Tasks: {self.num_tasks}"
    
    def plot_timeline(self, axes: plt.Axes, y_height:float, width):
        x = 0
        for item in self.timeline:
            if item[0] == "step":
                rect = patches.Rectangle((x, y_height), width=item[1], height=width, edgecolor="b", facecolor="b")
                axes.add_patch(rect)
            elif item[0] == "idle":
                rect = patches.Rectangle((x, y_height), width=item[1], height=width, edgecolor="r", facecolor="r")
                axes.add_patch(rect)
            
            x += item[1]
        return x


class Parser:

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self.sent_tasks = 0
        self.completed_tasks = 0
        self.run_time = 0
        self.idle_time = 0
        self.comms_time = 0
        self.misc_time = 0
        self.prev_run_time = None
        self.prev_idle_time = None
        self.prev_comms_time = None
        self.prev_misc_time = None
        self.timeline = []
    
    def get_log_data(self):
        return LogData(self.run_time, self.idle_time, self.misc_time, max(self.sent_tasks, self.completed_tasks), self.timeline)
    
    def split_log_line(line: str):
        if not line.startswith("Worker"):
            return [-1, "", "", 0.0]
        arr = line.split("|")
        arr = [x.strip() for x in arr]
        arr[0] = int(arr[0].removeprefix("Worker "))
        arr[-1] = float(arr[-1])
        return arr

    def parse_line(self, worker_id: int, task_type: str, task_name: str, time: float):
        if worker_id != self.worker_id:
            return
        
        if self.prev_misc_time is None:
            self.prev_misc_time = time
        
        if task_type == "finish step": # Actually performing a step
            self.run_time += time - self.prev_run_time
            self.prev_misc_time = time
            self.timeline.append(("step", time - self.prev_run_time))
        elif task_type == "start step": # Starting a step
            self.prev_run_time = time
            self.misc_time = time - self.prev_misc_time
            self.timeline.append(("misc", time - self.prev_misc_time))
        elif task_type == "start idle": 
            self.prev_idle_time = time
            self.misc_time = time - self.prev_misc_time
            self.timeline.append(("misc", time - self.prev_misc_time))
        elif task_type == "finish idle":
            self.idle_time += time - self.prev_idle_time
            self.timeline.append(("idle", time - self.prev_misc_time))
            self.prev_misc_time = time
        

def parse_worker(worker_id: int) -> LogData:
    global file_name
    parser = Parser(worker_id)
    with open(file_name, "r") as f_obj:
        for line in f_obj.readlines():
            parser.parse_line(*Parser.split_log_line(line))

    return parser.get_log_data()

if __name__ == '__main__':
    global file_name
    file_name = sys.argv[1]
    num_workers = int(sys.argv[2])

    with mp.Pool(processes=num_workers) as pool:
        log_data = pool.map(parse_worker, range(num_workers))


    fig, axes = plt.subplots(1,1)
    max_x = 20
    for i, data in enumerate(log_data):
        x = data.plot_timeline(axes, i*5 + 1, 4.5)
        if x > max_x:
            max_x = x

    axes.set_ybound(0, num_workers*5 + 6)
    axes.set_xbound(0, x + 1)

    fig.savefig("profiler_timeline.png")








