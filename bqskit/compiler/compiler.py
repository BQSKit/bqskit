"""
This module implements the Compiler class.

The Compiler class is a handle on a backend compiler.
The backend compiler is responsible for executing compilation tasks.
When creating a Compiler class, by default, a new process
is started where compilation tasks can be submitted to be run.
"""
import logging
from multiprocessing import Pipe
from multiprocessing import Process

from bqskit.compiler.task import CompilationTask
from bqskit.compiler.task import TaskStatus
from bqskit.compiler.workqueue import WorkQueue
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger('bqskit')


class Compiler:
    """The Compiler class."""

    def __init__(self) -> None:
        """
        Compiler Constructor.
        Starts a new backend compiler on the local machine with an empty
        WorkQueue, establishes a connection, and then starts running.

        Examples:
            >>> compiler = Compiler()
            >>> task = CompilationTask(...)
            >>> compiler.submit(task)
            >>> print(compiler.status(task))
            RUNNING
        """

        self.conn, backend_conn = Pipe()
        self.process = Process(target=WorkQueue.run, args=(backend_conn,))
        self.process.start()
        _logger.info('Started compiler process.')

    def __del__(self) -> None:
        """Shutdowns compiler and closes connection."""
        self.conn.send('CLOSE')
        self.process.join()
        self.conn.close()
        _logger.info('Stopped compiler process.')

    def submit(self, task: CompilationTask) -> None:
        """Submit a CompilationTask to the Compiler."""
        self.conn.send('SUBMIT')
        self.conn.send(task)
        self.conn.recv()  # Block until response
        _logger.info('Submitted task: %s' % task.task_id)

    def status(self, task: CompilationTask) -> TaskStatus:
        """Retrieve the status of the specified CompilationTask."""
        self.conn.send('STATUS')
        self.conn.send(task.task_id)
        return self.conn.recv()  # Block until response

    def result(self, task: CompilationTask) -> Circuit:
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
        self.conn.send('SUBMIT')
        self.conn.send(task)
        self.conn.recv()  # Block until response
        self.conn.send('RESULT')
        self.conn.send(task.task_id)
        return self.conn.recv()  # Block until response
