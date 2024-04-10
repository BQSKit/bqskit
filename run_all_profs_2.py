import time
import os


if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    file = "qsd_test"
    # tols = range(1, 7)
    methods = ["qft", "random"]
    num_qudits = [4]
    min_qudits = [2]
    tree_depths = [4, 8, 12]
    # tree_depths = [1]
    for method in methods:
        for num_qudit in num_qudits:
            for min_qudit in min_qudits:
                for tree_depth in tree_depths:
                    # for exp in range(tree_depth - 4, tree_depth + 1):
                    for num_worker in [4, 8, 64]:
                        # num_worker = 2 ** exp
                        # pathlib.Path(f"new_queue/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}").mkdir(parents=True, exist_ok=True)

                        execute = f"python {file}.py {method} {num_qudit} {min_qudit} {tree_depth} {num_worker} > new_queue/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}/log.txt"
                        print(execute)
                        os.system(execute)
                        time.sleep(0.05)