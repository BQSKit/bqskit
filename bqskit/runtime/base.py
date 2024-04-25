"""This module implements the NodeBase abstract class."""
from __future__ import annotations

import abc
import functools
import logging
import os
import random
import selectors
import signal
import socket
import sys
import time
import traceback
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

from bqskit.runtime import default_manager_port
from bqskit.runtime import default_worker_port
from bqskit.runtime import set_blas_thread_counts
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.direction import MessageDirection
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask
from bqskit.runtime.worker import start_worker


_logger = logging.getLogger(__name__)


class RuntimeEmployee:
    """Data structure for a boss's view of an employee."""

    def __init__(
        self,
        id: int,
        conn: Connection,
        total_workers: int,
        process: Process | None = None,
        is_manager: bool = False,
    ) -> None:
        """Construct an employee with all resources idle."""

        self.id = id
        """
        The ID of the employee.

        If this is a worker, then their unique worker id. If this is a manager,
        then their local id.
        """

        self.conn: Connection = conn
        self.total_workers = total_workers
        self.process = process
        self.num_tasks = 0
        self.num_idle_workers = total_workers
        self.is_manager = is_manager

        self.submit_cache: list[tuple[RuntimeAddress, int]] = []
        """
        Tracks recently submitted tasks by id and count.

        This is used to adjust the idle worker count when the employee sends a
        waiting message.
        """

    def initiate_shutdown(self) -> None:
        """Instruct employee to shutdown."""
        try:
            self.conn.send((RuntimeMessage.SHUTDOWN, None))
        except Exception:
            pass

    def complete_shutdown(self) -> None:
        """Ensure employee is shutdown and clean up resources."""
        if self.process is not None:
            self.process.join()

        self.process = None
        self.conn.close()

    def shutdown(self) -> None:
        """Initiate and complete shutdown."""
        self.initiate_shutdown()
        self.complete_shutdown()

    @property
    def recipient_string(self) -> str:
        """Return a string representation of the employee."""
        return f'{"Manager" if self.is_manager else "Worker"} {self.id}'

    @property
    def has_idle_resources(self) -> bool:
        return self.num_idle_workers > 0

    def get_num_of_tasks_sent_since(
        self,
        read_receipt: RuntimeAddress | None,
    ) -> int:
        """Return the number of tasks sent since the read receipt."""
        if read_receipt is None:
            return sum(count for _, count in self.submit_cache)

        for i, (addr, _) in enumerate(self.submit_cache):
            if addr == read_receipt:
                self.submit_cache = self.submit_cache[i:]
                return sum(count for _, count in self.submit_cache[1:])

        raise RuntimeError('Read receipt not found in submit cache.')


class MultiLevelQueue:
    """A multi-level queue for delaying submitted tasks."""

    def __init__(self) -> None:
        """Initialize the multi-level queue."""
        self.queue: dict[int, list[RuntimeTask]] = {}
        self.levels: int | None = None

    def delay(self, tasks: Sequence[RuntimeTask]) -> None:
        """Update the multi-level queue with tasks."""
        for task in tasks:
            task_depth = len(task.breadcrumbs)

            if task_depth not in self.queue:
                if self.levels is None or task_depth > self.levels:
                    self.levels = task_depth
                self.queue[task_depth] = []

            self.queue[task_depth].append(task)

    def empty(self) -> bool:
        """Return True if the multi-level queue is empty."""
        return self.levels is None

    def pop(self) -> RuntimeTask:
        """Pop the next task from the multi-level queue."""
        if self.empty():
            raise RuntimeError('Cannot pop from an empty multi-level queue.')

        task = self.queue[self.levels].pop()  # type: ignore  # checked above

        while self.levels is not None:
            if self.levels in self.queue:
                if len(self.queue[self.levels]) != 0:
                    break
                self.queue.pop(self.levels)
            self.levels -= 1
            if self.levels < 0:
                self.levels = None

        return task


def sigint_handler(signum: int, _: FrameType | None, node: ServerBase) -> None:
    """Interrupt the node."""
    if not node.running:
        return

    node.running = False
    node.terminate_hotline.send(b'\0')
    _logger.info('Server interrupted.')


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

        self.running = True
        """True while the node is running."""

        self.sel = selectors.DefaultSelector()
        """Used to efficiently idle and wake when communication is ready."""

        p, self.terminate_hotline = socket.socketpair()
        self.sel.register(p, selectors.EVENT_READ, MessageDirection.SIGNAL)
        """Terminate hotline is used to unblock select while running."""

        self.employees: list[RuntimeEmployee] = []
        """Tracks this node's employees, which are managers or workers."""

        self.conn_to_employee_dict: dict[Connection, RuntimeEmployee] = {}
        """Used to find the employee associated with a message."""

        self.multi_level_queue = MultiLevelQueue()
        """Used to delay tasks until they can be scheduled."""

        # Servers do not need blas threads
        set_blas_thread_counts(1)

        # Safely and immediately exit on interrupt signals
        handle = functools.partial(sigint_handler, node=self)
        signal.signal(signal.SIGINT, handle)

        # Start outgoing thread
        self.outgoing: Queue[tuple[Connection, RuntimeMessage, Any]] = Queue()
        self.outgoing_thread = Thread(target=self.send_outgoing, daemon=True)
        self.outgoing_thread.start()
        _logger.info('Started outgoing thread.')

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
            _logger.info(f'Connected to manager {i} at {ip}:{port}.')
            _logger.debug(f'Gave bounds {lb=} and {ub=} to manager {i}.')

        # Wait for started messages from all managers and register them
        self.total_workers = 0
        for i, conn in enumerate(manager_conns):
            msg, num_workers = conn.recv()
            assert msg == RuntimeMessage.STARTED
            self.employees.append(
                RuntimeEmployee(
                    i,
                    conn,
                    num_workers,
                    is_manager=True,
                ),
            )
            self.conn_to_employee_dict[conn] = self.employees[-1]
            self.sel.register(
                conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            _logger.info(f'Registered manager {i} with {num_workers=}.')
            self.total_workers += num_workers
        self.num_idle_workers = self.total_workers

        _logger.info(f'Node has {self.total_workers} total workers.')

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

    def spawn_workers(
        self,
        num_workers: int = -1,
        port: int = default_worker_port,
        logging_level: int = logging.WARNING,
        num_blas_threads: int = 1,
        profile: bool = False
    ) -> None:
        """
        Spawn worker processes.

        Args:
            num_workers (int): The number of workers to spawn. If -1,
                then spawn as many workers as CPUs on the system.
                (Default: -1).

            port (int): The port this server will listen for workers on.
                Default can be found in the
                :obj:`~bqskit.runtime.default_worker_port` global variable.

            logging_level (int): The logging level for the workers.

            num_blas_threads (int): The number of threads to use in BLAS
                libraries. (Default: 1).
        """
        if num_workers == -1:
            oscount = os.cpu_count()
            num_workers = oscount if oscount else 1

        if self.lower_id_bound + num_workers >= self.upper_id_bound:
            raise RuntimeError('Insufficient id range for workers.')

        # Create and start all worker processes
        procs = {}
        for i in range(num_workers):
            w_id = self.lower_id_bound + i
            procs[w_id] = Process(
                target=start_worker,
                args=(w_id, port, profile),
                kwargs={
                    'logging_level': logging_level,
                    'num_blas_threads': num_blas_threads,
                },
            )
            procs[w_id].daemon = True
            procs[w_id].start()
            _logger.debug(f'Stated worker process {i}.')

        # Listen for the worker connections
        family = 'AF_INET' if sys.platform == 'win32' else None
        listener = Listener(('localhost', port), family, backlog=num_workers)
        conns = [listener.accept() for _ in range(num_workers)]
        listener.close()

        # Organize all workers into the employees data structure
        temp_reorder = {}
        for i, conn in enumerate(conns):
            msg, w_id = conn.recv()
            assert msg == RuntimeMessage.STARTED
            employee = RuntimeEmployee(w_id, conn, 1, procs[w_id])
            temp_reorder[w_id - self.lower_id_bound] = employee
            self.conn_to_employee_dict[conn] = employee

        # The employess list needs to be sorted according to the IDs
        for i in range(num_workers):
            self.employees.append(temp_reorder[i])

        # Register employee communication
        for employee in self.employees:
            self.sel.register(
                employee.conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            _logger.debug(f'Registered worker {employee.id}.')

        self.step_size = 1
        self.total_workers = num_workers
        self.num_idle_workers = num_workers
        _logger.info(f'Node has spawned {num_workers} workers.')

    def connect_to_workers(
        self,
        num_workers: int = -1,
        port: int = default_worker_port,
    ) -> None:
        """
        Connect to worker processes.

        Args:
            num_workers (int): The number of workers to expect. If -1,
                then expect as many workers as CPUs on the system.
                (Default: -1).

            port (int): The port this server will listen for workers on.
                Default can be found in the
                :obj:`~bqskit.runtime.default_worker_port` global variable.
        """
        if num_workers == -1:
            oscount = os.cpu_count()
            num_workers = oscount if oscount else 1

        _logger.info(f'Expecting {num_workers} worker connections.')

        if self.lower_id_bound + num_workers >= self.upper_id_bound:
            raise RuntimeError('Insufficient id range for workers.')

        # Listen for the worker connections
        family = 'AF_INET' if sys.platform == 'win32' else None
        listener = Listener(('localhost', port), family, backlog=num_workers)
        conns = [listener.accept() for _ in range(num_workers)]
        listener.close()

        for i, conn in enumerate(conns):
            w_id = self.lower_id_bound + i
            self.outgoing.put((conn, RuntimeMessage.STARTED, w_id))
            employee = RuntimeEmployee(w_id, conn, 1)
            self.employees.append(employee)
            self.conn_to_employee_dict[conn] = employee

        # Register employee communication
        for employee in self.employees:
            w_id = employee.id
            assert employee.conn.recv() == (RuntimeMessage.STARTED, w_id)
            self.sel.register(
                employee.conn,
                selectors.EVENT_READ,
                MessageDirection.BELOW,
            )
            _logger.info(f'Registered worker {w_id}.')

        self.step_size = 1
        self.total_workers = num_workers
        self.num_idle_workers = num_workers
        _logger.info(f'Node has connected to {num_workers} workers.')

    def listen_once(self, ip: str, port: int) -> Connection:
        """Listen on `ip`:`port` for a connection and return on first one."""
        family = 'AF_INET' if sys.platform == 'win32' else None
        listener = Listener((ip, port), family)
        conn = listener.accept()
        listener.close()
        return conn

    def send_outgoing(self) -> None:
        """Outgoing thread forwards messages as they are created."""
        while True:
            outgoing = self.outgoing.get()

            if not self.running:
                # NodeBase's handle_shutdown will put a dummy value in the
                # queue to wake the thread up so it can exit safely.
                # Hence the node.running check now rather than in the
                # while condition.
                break

            if outgoing[0].closed:
                continue

            outgoing[0].send((outgoing[1], outgoing[2]))
            if _logger.isEnabledFor(logging.DEBUG):
                to = self.get_to_string(outgoing[0])
                _logger.debug(f'Sent message {outgoing[1].name} to {to}.')

            if outgoing[1] == RuntimeMessage.SUBMIT_BATCH:
                _logger.log(1, f'[{outgoing[2][0]}] * {len(outgoing[2])}\n')
            else:
                _logger.log(1, f'{outgoing[2]}\n')

            self.outgoing.task_done()

    def run(self) -> None:
        """Main loop."""
        _logger.info(f'{self.__class__.__name__} running...')

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
                        _logger.debug('Received interrupt signal.')
                        self.handle_shutdown()
                        return

                    # Unpack and Log message
                    try:
                        msg, payload = conn.recv()
                    except (EOFError, ConnectionResetError):
                        self.handle_disconnect(conn)
                        continue
                    log = f'Received message {msg.name} from {direction.name}.'
                    _logger.debug(log)
                    if msg == RuntimeMessage.SUBMIT_BATCH:
                        _logger.log(1, f'[{payload[0]}] * {len(payload)}\n')
                    else:
                        _logger.log(1, f'{payload}\n')

                    # Handle message
                    self.handle_message(msg, direction, conn, payload)

        except Exception:
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            _logger.error(error_str)
            self.handle_system_error(error_str)

        finally:
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

    @abc.abstractmethod
    def get_to_string(self, conn: Connection) -> str:
        """Return a string representation of the connection."""

    def handle_shutdown(self) -> None:
        """Shutdown the node and release resources."""
        # Stop running
        _logger.info('Shutting down node.')
        self.running = False

        self.all_profiles = {}

        # Instruct employees to shutdown
        for employee in self.employees:
            employee.initiate_shutdown()

        for employee in self.employees:
            employee.complete_shutdown()

        self.employees.clear()
        _logger.debug('Shutdown employees.')

        # Close selector
        self.sel.close()
        _logger.debug('Cleared selector.')

        # Close outgoing thread
        if self.outgoing_thread.is_alive():
            self.outgoing.put(b'\0')  # type: ignore
            self.outgoing_thread.join()
            _logger.debug('Joined outgoing thread.')
            assert not self.outgoing_thread.is_alive()

    def handle_disconnect(self, conn: Connection) -> None:
        """Remove `conn` from the server."""
        self.sel.unregister(conn)
        conn.close()

        # If one of my employees crashed/shutdown/disconnected, I shutdown
        if conn in self.conn_to_employee_dict:
            self.handle_shutdown()

    def assign_tasks(
        self,
        tasks: Sequence[RuntimeTask],
    ) -> list[list[RuntimeTask]]:
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
            (
                e.num_tasks + len(assignments[i]),  # Consider idle assignments
                random.random(),  # Random value for tie breaker
                i,
            )
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
        if len(tasks) == 0:
            return
        assignments = zip(self.employees, self.assign_tasks(tasks))
        sorted_assignments = sorted(
            assignments,
            key=lambda x: x[0].num_idle_workers,
            reverse=True,
        )  # Employees with the most idle workers get assignments first
        for e, assignment in sorted_assignments:
            self.send_tasks_to_employee(e, assignment)

        self.num_idle_workers = sum(e.num_idle_workers for e in self.employees)

    def schedule_for_workers(self, tasks: Sequence[RuntimeTask]) -> None:
        """Schedule tasks for workers, return the amount assigned."""
        if len(tasks) == 0:
            return

        num_assigned_tasks = 0
        for e in self.employees:
            # TODO: randomize? prioritize workers idle but with less tasks?
            if num_assigned_tasks >= len(tasks):
                break

            if e.num_idle_workers > 0:
                task = tasks[num_assigned_tasks]
                self.send_tasks_to_employee(e, [task])
                num_assigned_tasks += 1

        self.num_idle_workers = sum(e.num_idle_workers for e in self.employees)
        self.multi_level_queue.delay(tasks[num_assigned_tasks:])

    def send_tasks_to_employee(
        self,
        e: RuntimeEmployee,
        tasks: Sequence[RuntimeTask],
    ) -> None:
        """Send the `task` to the employee responsible for `worker_id`."""
        num_tasks = len(tasks)

        if num_tasks == 0:
            return

        e.num_tasks += num_tasks
        e.num_idle_workers -= min(num_tasks, e.num_idle_workers)
        e.submit_cache.append((tasks[0].unique_id, num_tasks))
        if num_tasks == 1:
            self.outgoing.put((e.conn, RuntimeMessage.SUBMIT, tasks[0]))
        else:
            self.outgoing.put((e.conn, RuntimeMessage.SUBMIT_BATCH, tasks))

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

    def broadcast(self, msg: RuntimeMessage, payload: Any) -> None:
        """Broadcast a cancel message to my employees."""
        for employee in self.employees:
            self.outgoing.put((employee.conn, msg, payload))

    def handle_importpath(self, paths: list[str]) -> None:
        """Update the system path with the given paths."""
        for path in paths:
            if path not in sys.path:
                sys.path.append(path)
        self.broadcast(RuntimeMessage.IMPORTPATH, paths)

    def handle_waiting(
        self,
        conn: Connection,
        new_idle_count: int,
        read_receipt: RuntimeAddress | None,
    ) -> None:
        """
        Record that an employee is idle with nothing to do.

        There is a race condition that is corrected here. If an employee sends a
        waiting message at the same time that its boss sends it a task, the
        boss's idle count will eventually be incorrect. To fix this, every
        waiting message sent by an employee is accompanied by a read receipt of
        the latest batch of tasks it has processed. The boss can then adjust the
        idle count by the number of tasks sent since the read receipt.
        """
        employee = self.conn_to_employee_dict[conn]
        unaccounted_tasks = employee.get_num_of_tasks_sent_since(read_receipt)
        adjusted_idle_count = max(new_idle_count - unaccounted_tasks, 0)

        old_count = employee.num_idle_workers
        employee.num_idle_workers = adjusted_idle_count
        self.num_idle_workers += (adjusted_idle_count - old_count)
        assert 0 <= self.num_idle_workers <= self.total_workers

    def handle_direct_worker_waiting(
        self,
        conn: Connection,
        new_idle_count: int,
        read_receipt: RuntimeAddress | None,
    ) -> None:
        """
        Record that a worker is idle with nothing to do.

        Schedule tasks from the multi-level queue to the worker.
        """
        ServerBase.handle_waiting(self, conn, new_idle_count, read_receipt)

        if self.multi_level_queue.empty():
            return

        employee = self.conn_to_employee_dict[conn]
        if employee.num_idle_workers > 0:
            task = self.multi_level_queue.pop()
            self.send_tasks_to_employee(employee, [task])
            self.num_idle_workers -= 1


def parse_ipports(ipports_str: Sequence[str]) -> list[tuple[str, int]]:
    """Parse command line ip and port inputs."""
    ipports = []
    for ipport_group in ipports_str:
        for ipport in ipport_group.split(','):
            if ipport.strip() == '':
                continue
            comps = ipport.strip().split(':')

            if len(comps) == 1:
                ip, port = comps[0], str(default_manager_port)
                # Expect only managers to be listening on these ips
                # so default port is manager's default port.

            elif len(comps) == 2:
                ip, port = comps

            else:
                raise ValueError(f'Invalid manager address: {ipport}.')

            if not (0 <= int(port) < 65536):
                raise ValueError(f'Invalid port number: {ipport}')

            ipports.append((ip, int(port)))
    return ipports


def import_tests_package() -> None:
    """
    Import tests package recursively during detached architecture testing.

    This should only be run by the CI test suite from the root bqskit folder.

    credit: https://www.youtube.com/watch?v=t43zBsVcva0
    """
    sys.path.append(os.path.join(os.getcwd()))
    import tests
    import pkgutil
    for mod in pkgutil.walk_packages(tests.__path__, f'{tests.__name__}.'):
        __import__(mod.name, fromlist=['_trash'])
