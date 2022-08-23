"""This module implements the Compiler class."""
from __future__ import annotations

import logging
import sys
import time
import uuid
from subprocess import PIPE
from subprocess import Popen
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from dask.distributed import Future
    from bqskit.compiler.task import CompilationTask
    from bqskit.ir.circuit import Circuit

from dask.distributed import Client
from threadpoolctl import threadpool_limits

from bqskit.compiler.executor import Executor

_logger = logging.getLogger(__name__)


class Compiler:
    """
    The BQSKit compiler class.

    A compiler is responsible for accepting and managing compilation tasks.
    The compiler class spins up a Dask execution environment, which
    compilation tasks can then access to parallelize their operations.
    The compiler is implemented as a context manager and it is recommended
    to use it as one. If the compiler is not used in a context manager, it
    is the responsibility of the user to call `close()`.

    Examples:
        1. Use in a context manager:
        >>> with Compiler() as compiler:
        ...     circuit = compiler.compile(task)

        2. Use compiler without context manager:
        >>> compiler = Compiler()
        >>> circuit = compiler.compile(task)
        >>> compiler.close()

        3. Connect to an already running dask cluster:
        >>> with Compiler('127.0.0.1:8786') as compiler:
        ...     circuit = compiler.compile(task)

        4. Connect to a dask cluster with a scheduler file:
        >>> with Compiler(scheduler_file='/path/to/file/') as compiler:
        ...     circuit = compiler.compile(task)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Construct a Compiler object.

        Notes:
            All arguments are passed directly to Dask. You can use
            these to connect to and configure a Dask cluster.
        """
        # Connect to or start a Dask cluster
        self.managed = len(args) == 0 and len(kwargs) == 0
        if self.managed:
            self.processes = start_dask_cluster()
            self.client = Client('127.0.0.1:8786')
        else:
            self.client = Client(*args, **kwargs)

        # Initialize workers for logging support
        def set_logging_level(level: int) -> None:
            import bqskit  # noqa
            logging.getLogger('bqskit').setLevel(level)

        self.client.run(set_logging_level, logging.getLogger('bqskit').level)

        # Limit threads for low level math calls on workers
        def limit_threads() -> None:
            threadpool_limits(limits=1)
        self.client.run(limit_threads)

        # Keep track of submitted compilation tasks
        self.tasks: dict[uuid.UUID, Future] = {}
        _logger.info('Started compiler process.')

    def __enter__(self) -> Compiler:
        """Enter a context for this compiler."""
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """Shutdown compiler."""
        self.close()

    def close(self) -> None:
        """Shutdown compiler."""
        self.client.close()
        self.tasks = {}
        if self.managed:
            for proc in self.processes:
                proc.terminate()  # type: ignore
                proc.wait()  # type: ignore
        _logger.info('Stopped compiler process.')

    def __del__(self) -> None:
        if self.managed and hasattr(self, 'processes'):
            for proc in self.processes:
                proc.terminate()  # type: ignore
                proc.wait()  # type: ignore

    def submit(self, task: CompilationTask) -> None:
        """Submit a CompilationTask to the Compiler."""
        executor = self.client.scatter(Executor(task))
        future = self.client.submit(Executor.run, executor, pure=False)
        self.tasks[task.task_id] = future
        _logger.info('Submitted task: %s' % task.task_id)

    def status(self, task: CompilationTask) -> str:
        """Retrieve the status of the specified CompilationTask."""
        return self.tasks[task.task_id].status

    def result(self, task: CompilationTask) -> Circuit:
        """Block until the CompilationTask is finished, return its result."""
        circ = self.tasks[task.task_id].result()[0]
        return circ

    def cancel(self, task: CompilationTask) -> None:
        """Remove a task from the compiler's workqueue."""
        self.client.cancel(self.tasks[task.task_id])
        _logger.info('Cancelled task: %s' % task.task_id)

    def compile(self, task: CompilationTask) -> Circuit:
        """Submit and execute the CompilationTask, block until its done."""
        _logger.info('Compiling task: %s' % task.task_id)
        self.submit(task)
        result = self.result(task)
        return result

    def analyze(self, task: CompilationTask, key: str) -> Any:
        """Gather the value associated with `key` in the task's data."""
        if task.task_id not in self.tasks:
            self.submit(task)
        return self.tasks[task.task_id].result()[1][key]


def start_dask_cluster() -> tuple[Popen[bytes], Popen[str]]:
    """Start a dask cluster and return the scheduler and worker processes."""
    scheduler_proc: Any = None
    worker_proc: Any = None

    # Start a dask scheduler in another process
    try:
        scheduler_proc = Popen(
            ['dask-scheduler', '--port', '8786'],
            bufsize=1,
            universal_newlines=True,
            stdout=sys.stdout,
            stderr=PIPE,
        )
        for line in iter(scheduler_proc.stderr.readline, ''):
            if 'Scheduler at:' in line:
                break
        scheduler_proc.stderr.close()
    except Exception as e:
        raise RuntimeError('Failed to start dask scheduler') from e

    # Start dask workers in another process
    try:
        worker_proc = Popen(
            ['dask-worker', '--nworkers', 'auto', '127.0.0.1:8786'],
            bufsize=1,
            universal_newlines=True,
            stdout=sys.stdout,
            stderr=PIPE,
        )
        for line in iter(worker_proc.stderr.readline, ''):
            if 'Starting established connection' in line:
                break
        worker_proc.stderr.close()
    except Exception as e:
        scheduler_proc.terminate()
        scheduler_proc.wait()
        raise RuntimeError('Failed to start dask worker') from e

    time.sleep(1)
    return worker_proc, scheduler_proc
