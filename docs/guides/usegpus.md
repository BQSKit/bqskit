# Using BQSkit on a GPU Cluster

This guide explains how to use BQSkit with GPUs by leveraging the `bqskit-qfactor-jax` package. This package provides GPU implementation support for the [QFactor](https://ieeexplore.ieee.org/abstract/document/10313638) and [QFactor-Sample](https://arxiv.org/abs/2405.12866) instantiation algorithms. For more detailed information and advanced configurations of the BQSkit runtime, refer to the [BQSKit distribution guide](https://bqskit.readthedocs.io/en/latest/guides/distributing.html).

We will guide you through the installation, setup, and execution process for BQSkit on a GPU cluster.

## bqskit-qfactor-jax Package Installation

First, you will need to install `bqskit-qfactor-jax`. Follow the instructions available on the [PyPI page](https://pypi.org/project/bqskit-qfactor-jax/).

## Setting Up the Environment

To run BQSkit with GPUs, you need to set up the BQSkit runtime properly. Each worker should be assigned to a specific GPU, and several workers can use the same GPU by utilizing [NVIDIA's MPS](https://docs.nvidia.com/deploy/mps/). You can set up the runtime on an interactive node or using SBATCH on several nodes. Below are the scripts to help you set up the runtime.

You may configure the number of GPUs to use on each node and also the number of workers on each GPU. If you use too many workers on the same GPU, you will get an out-of-memory exception. You may use the following table as a starting configuration and adjust the number of workers according to your specific circuit, unitary size, and GPU performance. You can use the `nvidia-smi` command to check the GPU usage during execution; it specifies the utilization of the memory and the execution units.

| Unitary Size   | Workers per GPU |
|----------------|------------------|
| 3,4            | 10               |
| 5              | 8                |
| 6              | 4                |
| 7              | 2                |
| 8 and more     | 1                |

Make sure that in your Python script you are creating the compiler object with the appropriate IP address. When running on the same node as the server, you can use `localhost` as the IP address.


### Interactive Node Setup Script
Use the following script to set up the environment on an interactive node. After the enviorment is up, you may open a seconed terminal and run your python script.

```bash
hostname=$(uname -n)
unique_id=bqskit_${RANDOM}
amount_of_gpus=<Number of GPUS to use in the node>
amount_of_workers_per_gpu=<Number of workers per GPU>
total_amount_of_workers=$(($amount_of_gpus * $amount_of_workers_per_gpu))
scratch_dir=$SCRATCH

wait_for_outgoing_thread_in_manager_log() {
    while [[ ! -f "$manager_log_file" ]]
    do
            sleep 0.5
    done

    while ! grep -q "Started outgoing thread." $manager_log_file; do
            sleep 1
    done
}

wait_for_server_to_connect(){
    while [[ ! -f "$server_log_file" ]]
    do
            sleep 0.5
    done

    while ! grep -q "Connected to manager" $server_log_file; do
            sleep 1
    done
}

mkdir -p $scratch_dir/bqskit_logs

manager_log_file=$scratch_dir/bqskit_logs/manager_${unique_id}.log
server_log_file=$scratch_dir/bqskit_logs/server_${unique_id}.log

echo "Will start bqskit runtime with id $unique_id gpus = $amount_of_gpus and workers per gpu = $amount_of_workers_per_gpu"

# Clean old server and manager logs, if exists
rm -f $manager_log_file
rm -f $server_log_file

echo "Starting MPS server"
nvidia-cuda-mps-control -d

echo "starting BQSKit managers"

bqskit-manager -x -n$total_amount_of_workers -vvv &> $manager_log_file &
manager_pid=$!
wait_for_outgoing_thread_in_manager_log

echo "starting BQSKit server on main node"
echo "Will run the command bqskit-server ${hostname} -vvv" > $server_log_file
bqskit-server $hostname -vvv &>> $server_log_file &
server_pid=$!

wait_for_server_to_connect

echo "Starting $total_amount_of_workers workers on $amount_of_gpus gpus"
for (( gpu_id=0; gpu_id<$amount_of_gpus; gpu_id++ ))
do
    echo "XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amount_of_workers_per_gpu"
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amount_of_workers_per_gpu > $scratch_dir/bqskit_logs/workers_${SLURM_JOB_ID}_${hostname}_${gpu_id}.log &
done

wait

echo "Stop MPS on $hostname"
echo quit | nvidia-cuda-mps-control

```

### Scripts to be Used in an SBATCH Across Several Nodes

Use the following SBATCH script to set up the job on a cluster:

```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t <time_to_run>
#SBATCH -n <number_of_nodes>
#SBATCH --gpus=<total number of GPUs, not nodes>
#SBATCH --output=<full_path_to_log_file>

date
uname -a

### load any modules needed and activate the conda enviorment
module load <module1>
module load <module2>
conda activate <conda-env-name>


echo "starting BQSKit managers on all nodes"
srun run_workers_and_managers.sh <number_of_gpus_per_node> <number_of_workers_per_gpu> &
managers_pid=$!

managers_started_file=$SCRATCH/managers_${SLURM_JOB_ID}_started
n=<number_of_nodes>


# Wait until  all the the  managers have started
while [[ ! -f "$managers_started_file" ]]
do
        sleep 0.5
done

while [ "$(cat "$managers_started_file" | wc -l)" -lt "$n" ]; do
    sleep 1
done

echo "starting BQSKit server on main node"
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ') &> $SCRATCH/bqskit_logs/server_${SLURM_JOB_ID}.log &
server_pid=$!

uname -a >> $SCRATCH/server_${SLURM_JOB_ID}_started

echo "will run python your command"

python <Your command>

date

echo "Killing the server"
kill -2 $server_pid

sleep 2
```


Save the following script as 'run_workers_and_managers.sh' in the same directory as your SBATCH script:
```bash
#!/bin/bash

node_id=$(uname -n)
amount_of_gpus=$1
amount_of_workers_per_gpu=$2
total_amount_of_workers=$(($amount_of_gpus * $amount_of_workers_per_gpu))
manager_log_file="$SCRATCH/bqskit_logs/manager_${SLURM_JOB_ID}_${node_id}.log"
server_started_file="$SCRATCH/server_${SLURM_JOB_ID}_started"
managers_started_file="$SCRATCH/managers_${SLURM_JOB_ID}_started"

touch $managers_started_file

wait_for_outgoing_thread_in_manager_log() {
    while ! grep -q "Started outgoing thread." $manager_log_file; do
        sleep 1
    done
    uname -a >> $managers_started_file
}

start_mps_servers() {
    echo "Starting MPS servers on node $node_id with CUDA $CUDA_VISIBLE_DEVICES"
    nvidia-cuda-mps-control -d
}

wait_for_bqskit_server() {
    i=0
    while [[ ! -f $server_started_file && $i -lt 10 ]]; do
        sleep 1
        i=$((i+1))
    done
}

start_workers() {
    echo "Starting $total_amount_of_workers workers on $amount_of_gpus gpus"
    for (( gpu_id=0; gpu_id<$amount_of_gpus; gpu_id++ )); do
        XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amount_of_workers_per_gpu &> $SCRATCH/bqskit_logs/workers_${SLURM_JOB_ID}_${node_id}_${gpu_id}.log &
    done
    wait
}

stop_mps_servers() {
    echo "Stop MPS servers on node $node_id"
    echo quit | nvidia-cuda-mps-control
}

if [ $amount_of_gpus -eq 0 ]; then
    echo "Will run manager on node $node_id with n args of $amount_of_workers_per_gpu"
    bqskit-manager -n $amount_of_workers_per_gpu -v &> $manager_log_file
    echo "Manager finished on node $node_id"
else
    echo "Will run manager on node $node_id"
    bqskit-manager -x -n$total_amount_of_workers -vvv &> $manager_log_file &
    wait_for_outgoing_thread_in_manager_log
    start_mps_servers
    wait_for_bqskit_server
    start_workers
    echo "Manager and workers finished on node $node_id" >> $manager_log_file
    stop_mps_servers
fi

```
