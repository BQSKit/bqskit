"""
This module implements the WorkQueue class.

The WorkQueue Class starts a new work thread that executes and tracks the tasks
enqueued in it.

"""
from __future__ import annotations

import logging
import time
import uuid
from multiprocessing.connection import Connection
from queue import Queue
from threading import Thread
from typing import Any

from bqskit.compiler.executor import Executor
from bqskit.compiler.task import CompilationTask
from bqskit.compiler.task import TaskResult
from bqskit.compiler.task import TaskStatus


_logger = logging.getLogger(__name__)


class WorkQueue:
    """The WorkQueue class."""

    def __init__(self) -> None:
        """
        WorkQueue Constructor. Creates an empty queue and starts a worker
        thread. CompilationTasks can be enqueued as work in the queue. The
        worker thread gets the tasks from the queue in FIFO order and executes
        them.

        Examples:
            >>> wq = WorkQueue()
            >>> task = CompilationTask(...)
            >>> wq.enqueue(task)
            >>> print(wq.status(task))
            TaskStatus.RUNNING

        """

        self.work_thread = Thread(target=self.do_work)
        self.work_queue: Queue[CompilationTask] = Queue()
        self.is_running = True
        self.work_thread.start()
        self.tasks: dict[uuid.UUID, dict[str, Any]] = {}

    def do_work(self) -> None:
        """Worker thread loop: gets work from queue and executes it."""
        _logger.info('Worker thread started.')
        while self.is_running:

            # Wait for work
            if self.work_queue.empty():
                time.sleep(1)
                continue
            task = self.work_queue.get()
            self.tasks[task.task_id] = {'status': TaskStatus.RUNNING}
            _logger.info('Started executing task: %s' % task.task_id)

            # Execute task
            try:
                executor = Executor(task)
                executor.run()
                self.tasks[task.task_id] = {
                    'status': TaskStatus.DONE,
                    'result': executor.get_result(),
                }
                _logger.info('Finished executing task: %s' % task.task_id)

            # Handle Errors
            except Exception as e:
                _logger.debug('Error executing task: %s' % task.task_id)
                _logger.debug(e)
                self.tasks[task.task_id] = {
                    'status': TaskStatus.ERROR,
                    'message': str(e),
                }

    def stop(self) -> None:
        """Stops the worker thread from starting another task."""
        self.is_running = False

    def enqueue(self, task: CompilationTask) -> None:
        """Enqueues a CompilationTask."""
        self.tasks[task.task_id] = {'status': TaskStatus.WAITING}
        self.work_queue.put(task)

    def status(self, task_id: uuid.UUID) -> TaskStatus:
        """Retrieve the status of the specified CompilationTask."""
        if task_id in self.tasks:
            return self.tasks[task_id]['status']
        return TaskStatus.ERROR

    def result(self, task_id: uuid.UUID) -> TaskResult:
        """Block until the CompilationTask is finished, return its result."""
        if task_id not in self.tasks:
            return TaskResult.does_not_exist()

        # wait for it to finish
        status = self.tasks[task_id]['status']
        while status != TaskStatus.DONE and status != TaskStatus.ERROR:
            time.sleep(1)
            status = self.tasks[task_id]['status']

        # Handle errors
        if 'result' not in self.tasks[task_id]:
            return TaskResult(self.tasks[task_id]['message'], TaskStatus.ERROR)

        return TaskResult.from_circuit(self.tasks[task_id]['result'])

    def remove(self, task_id: uuid.UUID) -> None:
        if task_id in self.tasks:
            _logger.info('Removing task: %s' % task_id)
            # TODO: Lock
            # If waiting, error, or done, remove easy
            # If running, remove carefully
            # unlock

    @staticmethod
    def run(conn: Connection) -> None:
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

            if msg == 'CLOSE':
                wq.stop()
                conn.close()
                break

            elif msg == 'SUBMIT':
                task = conn.recv()
                if not isinstance(task, CompilationTask):
                    pass  # TODO: Handle Error
                wq.enqueue(task)
                conn.send('OKAY')

            elif msg == 'STATUS':
                task_id = conn.recv()
                if not isinstance(task_id, uuid.UUID):
                    pass  # TODO: Handle Error
                conn.send(wq.status(task_id))

            elif msg == 'RESULT':
                task_id = conn.recv()
                if not isinstance(task_id, uuid.UUID):
                    pass  # TODO: Handle Error
                conn.send(wq.result(task_id))

            elif msg == 'REMOVE':
                task_id = conn.recv()
                if not isinstance(task_id, uuid.UUID):
                    pass  # TODO: Handle Error
                wq.remove(task_id)
                conn.send('OKAY')
