"""
This module implements the Compiler class.

The Compiler class is a handle on a backend compiler. The backend compiler is
responsible for executing compilation tasks. When creating a Compiler class, by
default, a new process is started where compilation tasks can be submitted to be
run.
"""
from __future__ import annotations

import logging
from multiprocessing import Pipe
from multiprocessing import Process
from typing import Any

from bqskit.compiler.task import CompilationTask
from bqskit.compiler.task import TaskResult
from bqskit.compiler.task import TaskStatus
from bqskit.compiler.workqueue import WorkQueue
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class Compiler:
    """The Compiler class."""

    def __init__(self) -> None:
        """
        Compiler Constructor. Starts a new backend compiler on the local machine
        with an empty WorkQueue, establishes a connection, and then starts
        running.

        Examples:
            >>> compiler = Compiler()
            >>> task = CompilationTask(...)
            >>> compiler.submit(task)
            >>> print(compiler.status(task))
            TaskStatus.RUNNING
        """
        self.conn, backend_conn = Pipe()
        self.process = Process(target=WorkQueue.run, args=(backend_conn,))
        self.process.start()
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
        self.conn.send('CLOSE')
        self.process.join()
        self.conn.close()
        _logger.info('Stopped compiler process.')

    def submit(self, task: CompilationTask) -> None:
        """Submit a CompilationTask to the Compiler."""
        self.conn.send('SUBMIT')
        self.conn.send(task)
        okay_msg = self.conn.recv()  # Block until response
        if (okay_msg != 'OKAY'):
            raise Exception('Failed to submit job.')
        _logger.info('Submitted task: %s' % task.task_id)

    def status(self, task: CompilationTask) -> TaskStatus:
        """Retrieve the status of the specified CompilationTask."""
        self.conn.send('STATUS')
        self.conn.send(task.task_id)
        return self.conn.recv()  # Block until response

    def result(self, task: CompilationTask) -> TaskResult:
        """Block until the CompilationTask is finished, return its result."""
        self.conn.send('RESULT')
        self.conn.send(task.task_id)
        return self.conn.recv()  # Block until response

    def remove(self, task: CompilationTask) -> None:
        """Remove a task from the compiler's workqueue."""
        self.conn.send('REMOVE')
        self.conn.send(task.task_id)
        self.conn.recv()  # Block until response
        _logger.info('Removed task: %s' % task.task_id)

    def compile(self, task: CompilationTask) -> Circuit:
        """Execute the CompilationTask."""
        _logger.info('Compiling task: %s' % task.task_id)
        self.submit(task)
        result = self.result(task)
        return result.get_circuit()

    # def get_supported_passes(...): TODO
