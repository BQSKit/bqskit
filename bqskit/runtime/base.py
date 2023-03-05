"""This module implements the NodeBase abstract class."""
from __future__ import annotations

import abc
import functools
import logging
import os
import random
import selectors
import signal
import sys
import time
import traceback
from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from multiprocessing.connection import Listener
from queue import Queue
from threading import Thread
from types import FrameType
from typing import Any
from typing import cast
from typing import Sequence

from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.direction import MessageDirection
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask
from bqskit.runtime.worker import start_worker


class RuntimeEmployee:
    """Data structure for a boss's view of an employee."""

    def __init__(
        self,
        conn: Connection,
        total_workers: int,
        process: Process | None = None,
        num_tasks: int = 0,
    ) -> None:
        """Construct an employee with all resources idle."""
        self.conn: Connection = conn
        self.total_workers = total_workers
        self.process = process
        self.num_tasks = num_tasks
        self.num_idle_workers = total_workers

    def shutdown(self) -> None:
        """Shutdown the employee."""
        try:
            self.conn.send((RuntimeMessage.SHUTDOWN, None))
        except Exception:
            pass

        if self.process is not None:
            self.process.join()

        self.process = None
        self.conn.close()

    @property
    def has_idle_resources(self) -> bool:
        return self.num_idle_workers > 0


def send_outgoing(node: ServerBase) -> None:
    """Outgoing thread forwards messages as they are created."""
    while True:
        outgoing = node.outgoing.get()

        if not node.running:
            # NodeBase's handle_shutdown will put a dummy value in the
            # queue to wake the thread up so it can exit safely.
            # Hence the node.running check now rather than in the
            # while condition.
            break

        outgoing[0].send((outgoing[1], outgoing[2]))
        node.logger.debug(f'Sent message {outgoing[1].name}.')
        node.logger.log(1, f'{outgoing[2]}\n')


def sigint_handler(signum: int, _: FrameType | None, node: ServerBase) -> None:
    """Interrupt the node."""
    if not node.running:
        return

    node.running = False
    node.terminate_hotline.send(b'\0')
    node.logger.info('Server interrupted.')


class ServerBase:
    """Base class for all non-worker process nodes in the BQSKit Runtime."""

    def __init__(self) -> None:
        """Initialize a runtime node component."""

        self.lower_id_bound = 0
        self.upper_id_bound = int(2 ** 30)
        """
        The node starts with an ID range from 0 -> 2^30. ID ranges are then
        assigned to managers by evenly splitting this range.

        Managers then recursively split their range when connecting the sub-
        managers. Finally, workers are assigned specific ids from within this
        range.
        """

        self.running = False
        """True while the node is running."""

        self.sel = selectors.DefaultSelector()
        """Used to efficiently idle and wake when communication is ready."""

        self.terminate_hotline, p = Pipe()
        self.sel.register(p, selectors.EVENT_READ, MessageDirection.SIGNAL)
        """Terminate hotline is used to unblock select while running."""

        self.logger = logging.getLogger('bqskit-runtime')
        """Logger used to print operational log messages."""

        self.employees: list[RuntimeEmployee] = []
        """Tracks this node's employees, which are managers or workers."""

        self.conn_to_employee_dict: dict[Connection, RuntimeEmployee] = {}
        """Used to find the employee associated with a message."""

        # Safely and immediately exit on interrupt signals
        handle = functools.partial(sigint_handler, node=self)
        signal.signal(signal.SIGINT, handle)

        # Start outgoing thread
        self.outgoing: Queue[tuple[Connection, RuntimeMessage, Any]] = Queue()
        self.outgoing_thread = Thread(target=send_outgoing, args=(self,))
        self.outgoing_thread.start()
        self.logger.info('Started outgoing thread.')

    def connect_to_managers(self, ipports: Sequence[tuple[str, int]]) -> None:
        """Connect to all managers given by endpoints in `ipports`."""
        d = len(ipports)
        self.step_size = (self.upper_id_bound - self.lower_id_bound) // d

        # Establish connections with all managers
        manager_conns = []
        for i, (ip, port) in enumerate(ipports):
            lb = self.lower_id_bound + (i * self.step_size)
            ub = min(
                self.lower_id_bound + ((i + 1) * self.step_size),
                self.upper_id_bound,
            )
            manager_conns.append(self.connect_to_manager(ip, port, lb, ub))
            self.logger.info(f'Connected to manager {i} at {ip}:{port}.')
            self.logger.debug(f'Gave bounds {lb=} and {ub=} to manager {i}.')

        # Wait for started messages from all managers and register them
        self.total_workers = 0
        for i, conn in enumerate(manager_conns):
            msg, num_workers = conn.recv()
            assert msg == RuntimeMessage.STARTED
            self.employees.append(RuntimeEmployee(conn, num_workers))
            self.conn_to_employee_dict[conn] = self.employees[-1]
            self.sel.register(
                conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            self.logger.info(f'Registered manager {i} with {num_workers = }.')
            self.total_workers += num_workers

        self.logger.info(f'Node has {self.total_workers} total workers.')

    def connect_to_manager(
        self,
        ip: str,
        port: int,
        lb: int,
        ub: int,
    ) -> Connection:
        """
        Connect to a manager at the endpoint given by `ip` and `port`.

        Args:
            ip (str): The IP address where the manager is expected to be
                listening.

            port (int): The port number on which the manager is expected
                to be listening.

            lb (int): The ID lower bound to send to the manager.

            ub (int): The ID upper bound to send to the manager.
        """
        max_retries = 5
        wait_time = .25

        for _ in range(max_retries):
            try:
                conn = Client((ip, port))
            except ConnectionRefusedError:
                time.sleep(wait_time)
                wait_time *= 2
            else:
                conn.send((RuntimeMessage.CONNECT, (lb, ub)))
                return conn

        raise RuntimeError(f'Manager connection refused at {ip}:{port}')

    def spawn_workers(self, num_workers: int = -1) -> None:
        """
        Spawn worker processes.

        Args:
            num_workers (int): The number of workers to spawn. If -1,
                then spawn as many workers as CPUs on the system.
                (Default: -1).
        """
        if num_workers == -1:
            oscount = os.cpu_count()
            num_workers = oscount if oscount else 1

        if self.lower_id_bound + num_workers >= self.upper_id_bound:
            raise RuntimeError('Insufficient id range for workers.')

        for i in range(num_workers):
            p, q = Pipe()
            args = (self.lower_id_bound + i, q)
            proc = Process(target=start_worker, args=args)
            self.employees.append(RuntimeEmployee(p, 1, proc))
            self.conn_to_employee_dict[p] = self.employees[-1]
            proc.start()

        for i, employee in enumerate(self.employees):
            assert employee.conn.recv() == ((RuntimeMessage.STARTED, None))
            self.sel.register(
                employee.conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            self.logger.info(f'Registered worker {i}.')

        self.step_size = 1
        self.total_workers = num_workers
        self.logger.info(f'Node has spawned {num_workers} workers.')

    def listen_once(self, port: int) -> Connection:
        """Listen on `port` for a connection and return on first one."""
        listener = Listener(('0.0.0.0', port))
        conn = listener.accept()
        listener.close()
        return conn

    def run(self) -> None:
        """Main loop."""
        self.logger.info(f'{self.__class__.__name__} running...')
        self.running = True

        try:
            while self.running:
                # Wait for messages
                events = self.sel.select()  # Say that 5 times fast

                for key, _ in events:
                    # Unpack message, payload, and direction.
                    conn = cast(Connection, key.fileobj)
                    direction = cast(MessageDirection, key.data)

                    # If interrupted by signal, shutdown and exit
                    if direction == MessageDirection.SIGNAL:
                        self.logger.debug('Received interrupt signal.')
                        self.handle_shutdown()
                        return

                    # Unpack and Log message
                    try:
                        msg, payload = conn.recv()
                    except EOFError:
                        self.handle_disconnect(conn)
                        continue
                    log = f'Received message {msg.name} from {direction.name}.'
                    self.logger.debug(log)
                    self.logger.log(1, f'{payload}\n')

                    # Handle message
                    self.handle_message(msg, direction, conn, payload)

        except Exception:
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            self.logger.error(error_str)
            self.handle_system_error(error_str)
            self.handle_shutdown()

    @abc.abstractmethod
    def handle_message(
        self,
        msg: RuntimeMessage,
        direction: MessageDirection,
        conn: Connection,
        payload: Any,
    ) -> None:
        """
        Process the message coming from `direction`.

        Args:
            msg (RuntimeMessage): The message type to handle.

            direction (MessageDirection): The direction the message came from.

            conn (Connection): The connection object where this came from.

            payload (Any): The message data.
        """

    @abc.abstractmethod
    def handle_system_error(self, error_str: str) -> None:
        """
        Handle an error in runtime code as opposed to client code.

        This is called when an error arises in runtime code not in a
        RuntimeTask's coroutine code.
        """

    def handle_shutdown(self) -> None:
        """Shutdown the node and release resources."""
        # Stop running
        self.logger.info('Shutting down node.')
        self.running = False

        # Instruct employees to shutdown
        for employee in self.employees:
            employee.shutdown()
        self.employees.clear()
        self.logger.debug('Shutdown employees.')

        # Close selector
        self.sel.close()
        self.logger.debug('Cleared selector.')

        # Close outgoing thread
        if self.outgoing_thread.is_alive():
            self.outgoing.put(b'\0')  # type: ignore
            self.outgoing_thread.join()
            self.logger.debug('Joined outgoing thread.')
            assert not self.outgoing_thread.is_alive()

    def handle_disconnect(self, conn: Connection) -> None:
        self.sel.unregister(conn)
        conn.close()

        # If one of my employees crashed/shutdown/disconnected, I shutdown
        if conn in self.conn_to_employee_dict:
            self.handle_shutdown()

    def __del__(self) -> None:
        """Ensure resources are cleaned up."""
        self.handle_shutdown()

    def assign_tasks(self, tasks: Sequence[RuntimeTask]) -> list[list[RuntimeTask]]:
        """
        Go through the tasks and assign each one to an employee.

        Strategy:
            - Assign first to idle workers evenly and randomly
            - Employees with more idle workers get prioritized
            - Assign remaining tasks to employees with the fewest tasks
        """
        # assignments will contain the assigned task list for each employee
        assignments: list[list[RuntimeTask]] = [[] for _ in self.employees]

        # Every employee's id repeated as many time as it has idle workers:
        idle_id_repeated_list: list[int] = sum(
            (
                [i] * e.num_idle_workers
                for i, e in enumerate(self.employees)
            ), [],
        )

        # Shuffle to reduce chance of inefficiency described in handle_waiting
        random.shuffle(idle_id_repeated_list)

        # Assign tasks to idle workers in random order
        for idle_employee_id, task in zip(idle_id_repeated_list, tasks):
            assignments[idle_employee_id].append(task)

        # Check if there are more tasks to be assigned
        num_remaining_tasks = len(tasks) - len(idle_id_repeated_list)

        if num_remaining_tasks <= 0:
            return assignments

        remaining_tasks = list(tasks[-num_remaining_tasks:])

        # Sort the employees by how many tasks they have
        ntasks = sorted([
            (e.num_tasks, random.random(), i)  # Random value for tie breaker
            for i, e in enumerate(self.employees)
        ])

        while len(remaining_tasks) > 0:
            num_tasks, r, employee_id = ntasks[0]
            assignments[employee_id].append(remaining_tasks.pop())
            ntasks[0] = (num_tasks + 1, r, employee_id)

            # Maintain sorted order by swapping up the updated count
            idx = 0
            while idx + 1 < len(ntasks) and ntasks[idx] > ntasks[idx + 1]:
                ntasks[idx + 1], ntasks[idx] = ntasks[idx], ntasks[idx + 1]
                idx += 1

        return assignments

    def schedule_tasks(self, tasks: Sequence[RuntimeTask]) -> None:
        """Schedule tasks between this node's employees."""
        assignments = self.assign_tasks(tasks)

        for e, assignment in zip(self.employees, assignments):
            num_tasks = len(assignment)

            if num_tasks == 0:
                continue

            self.outgoing.put((e.conn, RuntimeMessage.SUBMIT_BATCH, assignment))

            e.num_tasks += num_tasks
            e.num_idle_workers -= min(num_tasks, e.num_idle_workers)

    def send_result_down(self, result: RuntimeResult) -> None:
        """Send the `result` to the appropriate employee."""
        dest_worker_id = result.return_address.worker_id

        if not self.is_my_worker(dest_worker_id):
            raise RuntimeError('Cannot send result to unmanaged worker.')

        employee = self.get_employee_responsible_for(dest_worker_id)
        self.outgoing.put((employee.conn, RuntimeMessage.RESULT, result))

    def is_my_worker(self, worker_id: int) -> bool:
        """Return true if `worker_id` is one of my workers (recursively)."""
        employee_id = (worker_id - self.lower_id_bound) // self.step_size
        return 0 <= employee_id < len(self.employees)

    def get_employee_responsible_for(self, worker_id: int) -> RuntimeEmployee:
        """Return the employee that manages `worker_id`."""
        employee_id = (worker_id - self.lower_id_bound) // self.step_size
        return self.employees[employee_id]

    def broadcast_cancel(self, addr: RuntimeAddress) -> None:
        """Broadcast a cancel message to my employees."""
        for employee in self.employees:
            self.outgoing.put((employee.conn, RuntimeMessage.CANCEL, addr))

    def handle_waiting(self, conn: Connection, new_idle_count: int) -> None:
        """
        Record that an employee is idle with nothing to do.

        There is a race condition here that is allowed. If an employee
        sends a waiting message at the same time that this sends it a
        task, it will still be marked waiting even though it is running
        a task. We allow this for two reasons. First, the consequences are
        minimal: this situation can only lead to one extra task assigned
        to the worker that could otherwise go to a truly idle worker.
        Second, it is unlikely in the common BQSKit workflows, which have
        wide and shallow task graphs and each leaf task can require seconds
        of runtime.
        """
        self.conn_to_employee_dict[conn].num_idle_workers = new_idle_count
