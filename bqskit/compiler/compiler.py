"""
This module implements the Compiler class.

The Compiler class is a handle on a backend compiler. The backend compiler is
responsible for executing compilation tasks. When creating a Compiler class, by
default, a new process is started where compilation tasks can be submitted to be
run.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from dask.distributed import Client
from dask.distributed import Future

from bqskit.compiler.executor import Executor
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class Compiler:
    """The Compiler class."""

    def __init__(self) -> None:
        """
        Compiler Constructor.

        Examples:
            >>> compiler = Compiler()
            >>> task = CompilationTask(...)
            >>> compiler.submit(task)
            >>> print(compiler.status(task))
            TaskStatus.RUNNING
        """
        self.client = Client()
        self.tasks: dict[uuid.UUID, Future] = {}
        _logger.info('Started compiler process.')

    def __enter__(self) -> Compiler:
        """Enter a context for this compiler."""
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """Shutdowns compiler and closes connection."""
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Shutdowns compiler and closes connection."""
        try:
            self.client.close()
        except AttributeError:
            pass
        # _logger.info('Stopped compiler process.')

    def submit(self, task: CompilationTask) -> None:
        """Submit a CompilationTask to the Compiler."""
        executor = Executor(task)
        future = self.client.submit(Executor.run, executor, pure=False)
        self.tasks[task.task_id] = future
        _logger.info('Submitted task: %s' % task.task_id)

    def status(self, task: CompilationTask) -> str:
        """Retrieve the status of the specified CompilationTask."""
        return self.tasks[task.task_id].status

    def result(self, task: CompilationTask) -> Circuit:
        """Block until the CompilationTask is finished, return its result."""
        circ = self.tasks[task.task_id].result()
        return circ

    def cancel(self, task: CompilationTask) -> None:
        """Remove a task from the compiler's workqueue."""
        self.client.cancel(self.tasks[task.task_id])
        _logger.info('Cancelled task: %s' % task.task_id)

    def compile(self, task: CompilationTask) -> Circuit:
        """Execute the CompilationTask."""
        _logger.info('Compiling task: %s' % task.task_id)
        self.submit(task)
        result = self.result(task)
        return result
