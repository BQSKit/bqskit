"""
This module implements the Backend class.

The Backend class is a handle on a bqskit backend server.
The backend server is responsible for executing compilation tasks.
When creating a Backend class, by default, a new process is started
where compilation tasks can be submitted to be run.
"""
from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing.connection import Connection

from bqskit.task import CompilationTask
from bqskit.workqueue import WorkQueue


class Backend:
    """The Backend class."""

    def __init__(self) -> None:
        """
        Backend Constructor.
        Starts a new backend server on the local machine with an empty
        WorkQueue, establishes a connection, and then starts working.

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

    def close(self) -> None:
        """Shutdowns backend and closes connection."""
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
        self.conn.send(task)
        return self.conn.recv()  # Block until response

    def result(self, task: CompilationTask) -> str:  # TODO: Results Class
        """Block until the CompilationTask is finished, return its result."""
        self.conn.send('RESULT')
        self.conn.send(task)
        return self.conn.recv()  # Block until response

    # TODO: def remove ( self, task: CompilationTask ) -> None:
