import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os import path


class LogData:

    def __init__(self, run_time: float, idle_time: float, num_tasks: float, timeline: list[tuple[str, float]]):
        self.run_time = run_time
        self.idle_time = idle_time
        self.num_tasks = num_tasks
        self.timeline = timeline

    def __str__(self) -> str:
        return f"Run Time: {self.run_time} Idle Time: {self.idle_time} Num Tasks: {self.num_tasks}"
    
    def plot_timeline(self, axes: plt.Axes, y_height:float, width: float):
        color_map = {
            "idle": "r",
            "instantiate": "g",
            "qsd": "c",
            "decompose": "m",
        }
        x = 20
        for item in self.timeline:
            task_name, start, duration = item
            color = color_map.get(task_name, "b")
            rect = patches.Rectangle((start, y_height), width=duration, height=width, edgecolor=color, facecolor=color)
            axes.add_patch(rect) 
            x = start + duration
            
        return x


class Parser:

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        # self.sent_tasks = 0
        # self.completed_tasks = 0
        self.prog_start_time = None
        # self.run_time = 0
        # self.idle_time = 0
        self.start_task_time = None
        self.timeline = []
    
    def get_log_data(self):
        return LogData(0, 0, 0, self.timeline)
        # return LogData(self.run_time, self.idle_time, max(self.sent_tasks, self.completed_tasks), self.timeline)
    
    def split_log_line(line: str):
        if not line.startswith("Worker"):
            return []
        log_lines = line.split("Worker ")
        all_arrs = []
        for log_line in log_lines:
            if len(log_line) == 0:
                continue
            arr = log_line.split("|")
            arr = [x.strip() for x in arr]
            arr[0] = int(arr[0])
            arr[-1] = float(arr[-1])
            all_arrs.append(arr)
        return all_arrs

    def parse_line(self, worker_id: int, task_type: str, task_name: str, time: float):
        if worker_id != self.worker_id:
            return
        
        if self.prog_start_time is None:
            self.prog_start_time = time
            assert(task_type.startswith("start"))

        if task_type == "finish step": # Actually performing a step
            assert task_name == self.cur_task
            # Add to timeline (task_name, start_time, duration_time)
            time_obj = (task_name, self.start_task_time, time - self.start_task_time - self.prog_start_time)
            # print(time_obj)
            self.timeline.append(time_obj)
        elif task_type == "start step": # Starting a step
            # Track start of task time
            self.start_task_time = time - self.prog_start_time
            self.cur_task = task_name
        elif task_type == "start idle":
            # Track start of idle
            self.start_task_time = time - self.prog_start_time
            self.cur_task = "idle"
        elif task_type == "stop idle" or task_type == "finish idle":
            assert self.cur_task == "idle"
            # Add to timeline ("idle", start_time, duration_time)
            time_obj = (task_name, self.start_task_time, time - self.start_task_time - self.prog_start_time)
            # print(time_obj)
            self.timeline.append(time_obj)
        

def parse_worker(worker_id: int) -> LogData:
    global file_name
    parser = Parser(worker_id)
    with open(file_name, "r") as f_obj:
        for line in f_obj.readlines():
            lines = Parser.split_log_line(line)
            for line in lines:
                parser.parse_line(*line)
    return parser.get_log_data()

if __name__ == '__main__':
    global file_name
    # file_name = sys.argv[1]
    circ = sys.argv[1]
    num_qubits = sys.argv[2]
    min_qubits = int(sys.argv[3])

    tree_scan_depths = [4, 8, 12]

    fig = plt.figure(figsize=(20, 20))
    axes: list[list[plt.Axes]] = fig.subplots(3,3)


    for ii, tree_scan_depth in enumerate(tree_scan_depths):
        for jj , num_workers in enumerate([4, 8, 64]):
            file_name = f"/pscratch/sd/j/jkalloor/profiler/bqskit/new_queue/{circ}/{num_qubits}/{min_qubits}/{tree_scan_depth}/{num_workers}/log.txt"
            print(file_name)
            if path.exists(file_name):
                log_data: list[LogData] = []
                # with mp.Pool(processes=num_workers) as pool:
                #     log_data = pool.map(parse_worker, range(num_workers))
                for worker in range(num_workers):
                    log_data.append(parse_worker(worker))

                max_x = 20
                for j, data in enumerate(log_data):
                    x = data.plot_timeline(axes[ii][jj], j+ 1, 0.8)
                    if x > max_x:
                        max_x = x

                axes[ii][jj].set_ybound(0, num_workers + 1)
                axes[ii][jj].set_xbound(0, x)
                axes[0][jj].set_title(f"Num Workers: {num_workers}")

        axes[ii][0].set_ylabel(f"Tree Depth: {tree_scan_depth}")


    fig.savefig(f"{circ}_{num_qubits}_{min_qubits}_single_queue.png")








