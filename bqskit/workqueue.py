"""
This module implements the WorkQueue class.

The WorkQueue Class starts a new work thread that executes and tracks
the tasks enqueued in it.
"""

import time
import uuid
from queue import Queue
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import Dict

from bqskit.task import CompilationTask

class WorkQueue:
    """The WorkQueue class."""

    def __init__ ( self ) -> None:
        """
        WorkQueue Constructor.
        Creates an empty queue and starts a worker thread. CompilationTasks
        can be enqueued as work in the queue. The worker thread gets the
        tasks from the queue in FIFO order and executes them.

        Examples:
            >>> wq = WorkQueue()
            >>> task = CompilationTask(...)
            >>> wq.enqueue(task)
            >>> print(wq.status(task))
            RUNNING
        """

        self.work_thread = Thread(target = self.do_work)
        self.work_queue = Queue()
        self.is_running = True
        self.work_thread.start()
        self.tasks: Dict[uuid.UUID, CompilationTask] = {}
        # TODO: self.tasks should be Dict[uuid.UUID, Executor]

    def do_work ( self ) -> None:
        """Worker thread loop: gets work from queue and executes it"""
        while self.is_running:
            if self.work_queue.empty():
                time.sleep(1)
                continue
            task = self.work_queue.get()
            self.tasks[ task.task_id ] = { "status": "RUNNING" }
            self.process_task( task )
            self.tasks[ task.task_id ] = { "status": "DONE", "result": "YAY!" }

    def process_task ( self, task : CompilationTask ) -> None:
        """Executes a CompilationTask"""
        print( "Starting processing task: %s" % task.task_id )
        time.sleep(1)
        print( "Finished processing task: %s" % task.task_id )

    def stop ( self ) -> None:
        """Stops the worker thread from starting another task."""
        self.is_running = False

    def enqueue ( self, task: CompilationTask ) -> None:
        """Enqueues a CompilationTask."""
        self.tasks[ task.task_id ] = { "status": "WAITING" }
        self.work_queue.put( task )

    def status ( self, task: CompilationTask ) -> str: # TODO: Status Enum
        """Retrieve the status of the specified CompilationTask."""
        if task.task_id in self.tasks:
            return self.tasks[ task.task_id ]["status"]
        return "ERROR"

    def result ( self, task: CompilationTask ) -> str: # TODO: Result Class
        """Block until the CompilationTask is finished, return its result."""
        if task.task_id in self.tasks:
            status = self.tasks[ task.task_id ]["status"]
            while status != "DONE" and status != "ERROR":
                time.sleep(1)
                status = self.tasks[ task.task_id ]["status"]
            return self.tasks[ task.task_id ]["result"]

    @staticmethod
    def run( conn: Connection ) -> None:
        """
        Static Process Function.

        When a new Backend process is started, it enters here.
        Currently, it creates a WorkQueue with a worker thread
        and communicates with the BQSKit Frontend via conn.

        Args:
            conn (Connection): The connection object used to communicate
                to the frontend. Reads commands from the connection,
                processes them, and responds in a loop.
        """

        wq = WorkQueue()

        while True:

            msg = conn.recv()

            if msg == "CLOSE":
                wq.stop()
                conn.close()
                break

            elif msg == "SUBMIT":
                task = conn.recv()
                wq.enqueue( task )
                conn.send( "OKAY" )

            elif msg == "STATUS":
                task = conn.recv()
                conn.send( wq.status( task ) )

            elif msg == "RESULT":
                task = conn.recv()
                conn.send( wq.result( task ) )
