import time
import subprocess
import pathlib

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=00:55:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{method}/{block_size}/{num_worker}/slurm_run.log

module load python
conda activate /pscratch/sd/j/jkalloor/profiler_env
echo "{file}.py {method} {block_size} {num_worker}"
python {file}.py {method} {block_size} {num_worker} > {method}/{block_size}/{num_worker}/log.txt
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    file = "prof_tester"
    # file = "full_compile"
    circs = ["TFXY_t"]
    circs = ["Heisenberg_7"]
    # tols = range(1, 7)
    tols = [3,4,5,6]
    methods = ["scan", "leap", "qsearch"]
    block_sizes = [3,4,5]
    num_workers = [2,4,8,16,32,64]
    for method in methods:
        for block_size in block_sizes:
            for num_worker in num_workers:
                # pathlib.Path(f"{method}/{block_size}/{num_worker}").mkdir(parents=True, exist_ok=True)
                to_write = open(file_name, 'w')
                to_write.write(header.format(file=file, method=method, block_size=block_size, num_worker=num_worker))
                to_write.close()
                time.sleep(2*sleep_time)
                print(method, block_size, num_worker)
                output = subprocess.check_output(['sbatch' , file_name])
                print(output)
                time.sleep(sleep_time)