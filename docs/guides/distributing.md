# Distributing BQSKit Across a Cluster

This guide describes how to launch a BQSKit Runtime Server in detached mode on
one or more computers, connect to it, and perform compilations on the server.
The detached mode allows for greater parallelization from cluster computing
all the way up to supercomputers.

## Detached Runtime Architecture

When distributing a runtime across many computers, we need to start it in
detached mode, where the server is "detached" from the client, i.e., started
and stopped independently. As opposed to the attached mode, which happens
transparently to the user whenever they instantiate a [`Compiler`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.compiler.Compiler.html#bqskit.compiler.Compiler) or call the [`compile()`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.compiler.compile.html) function.

The detached runtime architecture consists of four types of entities:

    1. **Clients** who submit compilation tasks
    2. A **Server** that acts as a central controller
    3. **Managers** that manage workers and act as a liaison between the
       server and workers
    4. **Workers** that perform the actual computation

## Starting a Detached Server

To start a detached server, we first must start the managers, who will spin
up workers, then wait to connect to a server. To do this, launch the following
executable:

```sh
bqskit-manager
```

See `bqskit-manager --help` for info on options. A manager should be started
on all shared memory (typically individual computers/nodes) systems first.
Then, we launch the server and provide the IP addresses of all the managers
to connect to:

```sh
bqskit-server <manager-ip-1> <manager-ip-2> ...
```

See `bqskit-server --help` for info on potential options.

### Optionally Configuring Worker Ranks

There is an optional, slightly altered way to start the managers that allows
more fine-grained control over worker ranks. This method is helpful if you
need to specify environment variables to different groups of workers on the
same system. For example, in a multi-GPU system, you may use environment
variables to control access to GPUs and want to split the workers to different
GPUs but still have one manager responsible for the lot.

To accomplish this, we now start the workers independently, then the
managers, and finally the server as usual:

```sh
ENV_VAR=SOMETHING bqskit-worker <num_workers_in_rank>
```

See `bqskit-worker --help` for more info. You can also use the `--cpus`
flag to pin workers to cores.

In this start-up method, we need to tell the manager to connect to the workers
rather than spawn them. This is done with `-x` flag.

```sh
bqskit-manager -x
```

### Spawning a Manager Hierarchy

Typically, in a small cluster, managers will directly manage workers and
report to the central server. However, this is not a necessity. Managers
can manage other managers and work under different managers. This hierarchical,
potentially-unbalanced architecture may be helpful in heterogenous clusters
or large supercomputers where more explicit control over communication is
desired.

While spawning the server, we spawn the lower-level managers first,
then work our way up until we complete the runtime with a server. The
lowest-level workers and managers can be spawned the same as above, but now
we spawn the next-level up managers by passing in the ip address of the
lower-level managers:

```sh
bqskit-manager -m <lower-level-manager-ip-1> <lower-level-manager-ip-2> ...
```

Lastly, the server is started the same way:

```sh
bqskit-server <most-senior-manager-ip-1> <most-senior-manager-ip-2> ...
```

### Shutting Down a Server

Interrupting the central server will properly shut down the entire runtime
architecture and clean up all resources. A client cannot shut down the
runtime in detached mode.

## Connecting to and Compiling with a Server

Once a bqskit server has been started in detached mode, we can connect to it
by simplying passing it's IP address into a `Compiler` constructor or as a
keyword argument in the `compile()` method:

```python
with Compiler(ip='server_ip') as compiler:
    compiler.compile(...)
```

or

```python
bqskit.compile(..., ip='server_ip')
```

## Sample SLURM Script Template

Below is a sample SLURM script template that can be filled in to start a
detached runtime and execute a client script all-in-one.

```
#!/bin/bash
#SBATCH --job-name={name}
#SBATCH -A {account}
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t {timelimit}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --output={out_log}
#SBATCH --error={err_log}

module load python &> /dev/null
conda activate {pythonenv} &> /dev/null

echo "Starting runtime managers..."
srun --output {manager_log}-%t bqskit-manager -n{num_workers} -v &

count_started_managers() {{
    for log in $(find $(dirname {manager_log}) -wholename "{manager_log}-*");
    do
        grep -q "Started outgoing thread." $log && printf ".";
    done | wc -m
}}

while [ $(count_started_managers | xargs echo -n) -lt {nodes} ]; do
    sleep 1
done

echo "Starting runtime server..."
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\\n' ' ') -v &> {server_log} &
server_pid=$!

sleep 1

echo "Running compilation workflow..."
python {python_script} localhost

echo "Shutting down server."
kill -2 $server_pid
```

## See Also

- [Runtime](https://bqskit.readthedocs.io/en/latest/source/runtime.html)
- [Compiler](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.compiler.Compiler.html#bqskit.compiler.Compiler)
