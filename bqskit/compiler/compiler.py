"""This module implements the Compiler class."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from dask.distributed import Client
from dask.distributed import Future
from dask.distributed import SpecCluster

from bqskit.compiler.executor import Executor
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit

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
        >>> with Compiler() as compiler:
        ...     circuit = compiler.compile(task)

        >>> compiler = Compiler()
        >>> circuit = compiler.compile(task)
        >>> compiler.close()
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Construct a Compiler object.

        Notes:
            All arguments are passed directly to Dask. You can use
            these to connect to and configure a Dask cluster.
        """
        if any(isinstance(arg, SpecCluster) for arg in args):
            dask_options = {}
        else:
            dask_options = {
                'silence_logs': logging.getLogger('bqskit').level,
            }
            if 'address' not in kwargs and 'scheduler_file' not in kwargs:
                dask_options['threads_per_worker'] = 1

        dask_options.update(kwargs)

        self.client = Client(*args, **dask_options)
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
        try:
            self.client.close()
            self.tasks = {}
            _logger.info('Stopped compiler process.')
        except (AttributeError, TypeError):
            pass

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
