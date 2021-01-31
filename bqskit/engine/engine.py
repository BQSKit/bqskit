"""
This module implements the Engine class.

The engine is responsible for executing compilation tasks,
and the Engine class is the bqskit frontend's handle on the backend's
engine. When creating a Backend class, by default, a new process
is started where compilation tasks can be submitted to be run.
"""

from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing.connection import Connection

from bqskit import CompilationTask
from bqskit.engine import WorkQueue


class Engine:
    """The Engine class."""

    def __init__(self) -> None:
        """
        Engine Constructor.
        Starts a new backend engine on the local machine with an empty
        WorkQueue, establishes a connection, and then starts running.

        Examples:
            >>> backend = Backend()
            >>> task = CompilationTask(...)
            >>> backend.submit(task)
            >>> print(backend.status(task))
            RUNNING
        """

        self.conn, backend_conn = Pipe()
        self.process = Process(target=WorkQueue.run, args=(backend_conn,))
        self.process.start()

    def stop(self) -> None:
        """Shutdowns engine and closes connection."""
        self.conn.send('CLOSE')
        self.conn.close()
        self.process.join()

    def submit(self, task: CompilationTask) -> None:
        """Submit a CompilationTask to the Backend."""
        self.conn.send('SUBMIT')
        self.conn.send(task)
        self.conn.recv()  # Block until response

    def status(self, task: CompilationTask) -> str:  # TODO: Status Enum
        """Retrieve the status of the specified CompilationTask."""
        self.conn.send('STATUS')
        self.conn.send(task.task_id)
        return self.conn.recv()  # Block until response

    def result(self, task: CompilationTask) -> str:  # TODO: Results Class
        """Block until the CompilationTask is finished, return its result."""
        self.conn.send('RESULT')
        self.conn.send(task.task_id)
        return self.conn.recv()  # Block until response

    def remove(self, task: CompilationTask) -> None:
        """Remove a task from the engine's workqueue."""
        self.conn.send('REMOVE')
        self.conn.send(task.task_id)
        self.conn.recv()  # Block until response
