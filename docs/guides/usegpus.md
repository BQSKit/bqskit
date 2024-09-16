# Using BQSKit on a GPU Cluster

This guide explains how to use BQSKit with GPUs by leveraging the `bqskit-qfactor-jax` package. The `bqskit-qfactor-jax` package provides GPU implementation support for the [QFactor](https://ieeexplore.ieee.org/abstract/document/10313638) and [QFactor-Sample](https://arxiv.org/abs/2405.12866) instantiation algorithms. For more detailed information and advanced configurations of the BQSKit runtime, refer to the [BQSKit distribution guide](https://bqskit.readthedocs.io/en/latest/guides/distributing.html).

We will guide you through the installation, setup, and execution process for BQSKit on a GPU cluster.

## bqskit-qfactor-jax Package Installation

First, you will need to install `bqskit-qfactor-jax`. This can easily done by using pip
```sh
pip install bqskit-qfactor-jax
```

This command will install also all the dependencies including BQSKit and JAX with GPU support.

## Optimizing a Circuit Using QFactor-Sample and the Gate Deletion Flow
This section explains how to optimize a quantum circuit using QFactor-Sample and the gate deletion flow.

First we load the circuit to be optimized using the Circuit class.
```python
from bqskit import Circuit

# Load a circuit from QASM
in_circuit = Circuit.from_file("circuit_to_opt.qasm")
```

Then we create the instniator instance, and set the number of multistarts to 32.
```python
from qfactorjax.qfactor_sample_jax import QFactorSampleJax

num_multistarts = 32

qfactor_sample_gpu_instantiator = QFactorSampleJax()

instantiate_options = {
        'method': qfactor_sample_gpu_instantiator,
        'multistarts': num_multistarts,
    }

```

Next, generate the optimization flow.
```python
from bqskit.passes import *

# Prepare the compilation passes
passes = [
    # Convert U3s to VU
    ToVariablePass(),

    # Split the circuit into partitions
    QuickPartitioner(partition_size),

    # For each partition perform scanning gate removal using QFactor jax
    ForEachBlockPass([
        ScanningGateRemovalPass(
            instantiate_options=instantiate_options,
        ),
    ]),

    # Combine the partitions back into a circuit
    UnfoldPass(),

    # Convert back the VariablueUnitaires into U3s
    ToU3Pass(),
]
```


Finally, use a compiler instance to execute the passes, and then print the statistics. If your system has more than a single GPU, then you should initiate a detached server and connect to it. A destailed explanation on how to setup BQSKit runtime is given in the next sections of the this guide.
```python
from bqskit.compiler import Compiler

with Compiler(num_workers=1) as compiler:
    
    out_circuit = compiler.compile(in_circuit, passes)

    print(
            f'Circuit finished with gates: {out_circuit.gate_counts}, '
            f'while started with {in_circuit.gate_counts}',
        )
```

## QFactor-JAX and QFactor-Sample-JAX Use Examples


For other usage examples, please refer to the [examples directory](https://github.com/BQSKit/bqskit-qfactor-jax/tree/main/examples)  in the `bqskit-qfactor-jax` package. There, you will find two Toffoli instantiation examples using QFactor and QFactor-Sample, as well as two different synthesis flows that also utilize these algorithms.


## Setting Up a Multi-GPU Environment

To run BQSKit with multiple GPUs, you need to set up the BQSKit runtime properly. Each worker should be assigned to a specific GPU by leveragig NVIDIA's CUDA_VISIBLE_DEVICES enviorment variable. Several workers can use the same GPU by utilizing [NVIDIA's MPS](https://docs.nvidia.com/deploy/mps/). You can set up the runtime on a single server ( or interactive node on a cluster) or using SBATCH on several nodes. You can find scripts to help you set up the runtime in this [link](https://github.com/BQSKit/bqskit-qfactor-jax/tree/main/examples/bqskit_env_scripts).

You may configure the number of GPUs to use on each server and also the number of workers on each GPU. If you use too many workers on the same GPU, you will run out of memory and experince an out-of-memory exception. If you are using QFactor, you may use the following table as a starting configuration and adjust the number of workers according to your specific circuit, unitary size, and GPU performance. If you are using QFactor-Sample, start with a single worker and increase if the memory premits it. You can use the `nvidia-smi` command to check the GPU usage during execution; it specifies the utilization of the memory and the execution units.

| Unitary Size   | Workers per GPU |
|----------------|------------------|
| 3,4            | 10               |
| 5              | 8                |
| 6              | 4                |
| 7              | 2                |
| 8 and more     | 1                |

Make sure that in your Python script you are creating the compiler object with the appropriate IP address. When running on the same node as the server, you can use \`localhost\` as the IP address.

```python
with Compiler('localhost') as compiler:
    out_circuit = compiler.compile(in_circuit, passes)
```


### Single Server Multiple GPUs Setup
This section of the guide explains the main concepts in the [single_server_env.sh](https://github.com/BQSKit/bqskit-qfactor-jax/blob/main/examples/bqskit_env_scripts/single_server_env.sh) script template and how to use it. The script creates a GPU enabled BQSKit runtime and is easily configured for any system. 

After you configure the template (replacing every  <> with an appropriate value) run it, and then in a seperate shell execute your python scirpt that uses this runtime enviorment.

The enviorment script has the following parts:
1. Variable configuration - choosing the number of GPUs to use, and the number of workrs per GPU. Moreover, the scratch dir path is configured, later to be used for logging.
```bash
#!/bin/bash
hostname=$(uname -n)
unique_id=bqskit_${RANDOM}
amount_of_gpus=<Number of GPUS to use in the node>
amount_of_workers_per_gpu=<Number of workers per GPU>
total_amount_of_workers=$(($amount_of_gpus * $amount_of_workers_per_gpu))
scratch_dir=<temp_dir>
```
2. Log file monitoring functions to monitor the startup of BQSKit managers and server.
```bash
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
```
3. Creating the log directory, and deleting any old log files that conflicts with the current run logs.
```bash
mkdir -p $scratch_dir/bqskit_logs

manager_log_file=$scratch_dir/bqskit_logs/manager_${unique_id}.log
server_log_file=$scratch_dir/bqskit_logs/server_${unique_id}.log

echo "Will start bqskit runtime with id $unique_id gpus = $amount_of_gpus and workers per gpu = $amount_of_workers_per_gpu"

# Clean old server and manager logs, if exists
rm -f $manager_log_file
rm -f $server_log_file
```
4. Starting NVIDA MPS to allow an efficient execution of multiple works on a single GPU.
```bash
echo "Starting MPS server"
nvidia-cuda-mps-control -d
```
5. Starting the BQSKit manager, and indicating to wait for workers to connect to it. Waiting for the manager to start listening for a connection from a server. This is important as the server might timeout if the manager isn't ready for the connection.
```bash
echo "starting BQSKit managers"

bqskit-manager -x -n$total_amount_of_workers -vvv &> $manager_log_file &
manager_pid=$!
wait_for_outgoing_thread_in_manager_log
```
6. Starting the BQSKit server indicating that there is a single manager in the current server. Waiting untill the server connects to the manager before continuing to start the workers.
```bash
echo "starting BQSKit server"
bqskit-server $hostname -vvv &>> $server_log_file &
server_pid=$!

wait_for_server_to_connect
```
7. Starting the workrs, each seeing only a specific GPU.
```bash
echo "Starting $total_amount_of_workers workers on $amount_of_gpus gpus"
for (( gpu_id=0; gpu_id<$amount_of_gpus; gpu_id++ ))
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amount_of_workers_per_gpu > $scratch_dir/bqskit_logs/workers_${SLURM_JOB_ID}_${hostname}_${gpu_id}.log &
done
```
8. After all the processes have finished, stop the MPS server.
```bash
wait

echo "Stop MPS on $hostname"
echo quit | nvidia-cuda-mps-control
```


### Multis-Server Multi-GPU Enviorment Setup

This section of the guide explains the main concepts in the [init_multi_node_multi_gpu_slurm_run.sh](https://github.com/BQSKit/bqskit-qfactor-jax/blob/main/examples/bqskit_env_scripts/init_multi_node_multi_gpu_slurm_run.sh) [run_workers_and_managers.sh](https://github.com/BQSKit/bqskit-qfactor-jax/blob/main/examples/bqskit_env_scripts/run_workers_and_managers.sh) scripts and how to use them. After configuring the scripts (updating every <>), place both of them in the same directory and initate a an SBATCH command. These scripts assume a SLURM enviorment, but can be easily ported to other disterbutation systems.

```bash
sbatch init_multi_node_multi_gpu_slurm_run.sh
```

The rest of this section exaplains in detail both of the scripts.

#### init_multi_node_multi_gpu_slurm_run 
This is a SLURM batch script for running a multi-node BQSKit task across multiple GPUs. It manages job submission, environment setup, launching the BQSKit server and workers on different nodes, and the execution of the main application.

1. Job configuration and logging - this is a standard SLURM SBATCH header.
```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t <time_to_run>
#SBATCH -n <number_of_nodes>
#SBATCH --gpus=<total number of GPUs, not nodes>
#SBATCH --output=<full_path_to_log_file>

scratch_dir=<temp_dir>
```

2. Shell environment setup - Please consulte with your HPC system admin to choose the apropriate modules to load that will enable you to JAX on NVDIA's GPUs. You may use NERSC's Perlmutter [documentation](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#jax) as a reference.
```bash
### load any modules needed and activate the conda enviorment
module load <module1>
module load <module2>
conda activate <conda-env-name>
```

3. Starting the managers on all of the nodes using SLURM’s srun command, initiating the run_workers_and_managers.sh script across all nodes. The former handles starting managers and workers on each node.
```bash
echo "starting BQSKit managers on all nodes"
srun run_workers_and_managers.sh <number_of_gpus_per_node> <number_of_workers_per_gpu> &
managers_pid=$!

managers_started_file=$scratch_dir/managers_${SLURM_JOB_ID}_started
n=<number_of_nodes>
```

4. Waiting for all managers to start, by tracking the number of lines in the log file, one created by each manager.
```bash
while [[ ! -f "$managers_started_file" ]]
do
        sleep 0.5
done

while [ "$(cat "$managers_started_file" | wc -l)" -lt "$n" ]; do
    sleep 1
done
```

5. Starting the BQSKit server on the main node, and using SLURM's `SLURM_JOB_NODELIST` enviorment variable to indicate the BQSKit server the hostnames of the managers.
```bash
echo "starting BQSKit server on main node"
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ') &> $scratch_dir/bqskit_logs/server_${SLURM_JOB_ID}.log &
server_pid=$!

uname -a >> $scratch_dir/server_${SLURM_JOB_ID}_started
```

6. Executing the main application, that will connect to the BQSKit runtime
```bash
python <Your command>
```

7. After the run is over, closing the BQSKit server.
```bash
echo "Killing the server"
kill -2 $server_pid
```

#### run_workers_and_managers.sh
This script is executed by each node to start the workers and managers on that specific node. It interacts with `init_multi_node_multi_gpu_slurm_run.sh`, the SBATCH script. If GPUs are required, the workers are spawnd seperatly from the manager, allowing for better configuratio of each worker.

The script starts with argument parsing and some variable configuration
```bash
#!/bin/bash

node_id=$(uname -n)
amount_of_gpus=$1
amount_of_workers_per_gpu=$2
total_amount_of_workers=$(($amount_of_gpus * $amount_of_workers_per_gpu))

scratch_dir=<temp_dir>
manager_log_file="$scratch_dir/bqskit_logs/manager_${SLURM_JOB_ID}_${node_id}.log"
server_started_file="$scratch_dir/server_${SLURM_JOB_ID}_started"
managers_started_file="$scratch_dir/managers_${SLURM_JOB_ID}_started"

touch $managers_started_file
```

Then the script declares a few utility methods.

```bash
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
        XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id bqskit-worker $amount_of_workers_per_gpu &> $scratch_dir/bqskit_logs/workers_${SLURM_JOB_ID}_${node_id}_${gpu_id}.log &
    done
    wait
}

stop_mps_servers() {
    echo "Stop MPS servers on node $node_id"
    echo quit | nvidia-cuda-mps-control
}
```

Finaly, the script chekcs if GPUs are not needed, it spwans the manager with its default behaviour, else suing the "-x" argument, it indicates to the manager to wait for connecting workers.
```bash
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