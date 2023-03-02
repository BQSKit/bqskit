"""This module implements the DetachedServer runtime."""
from __future__ import annotations

import argparse
import functools
import logging
import selectors
import signal
import socket
import sys
import time
import traceback
import uuid
from logging import LogRecord
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from multiprocessing.connection import Listener
from threading import Thread
from types import FrameType
from typing import Any
from typing import cast
from typing import List

from bqskit.compiler.status import CompilationStatus
from bqskit.compiler.task import CompilationTask
from bqskit.runtime import default_manager_port
from bqskit.runtime import default_server_port
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


def listen(server: DetachedServer, listener: Listener) -> None:
    """Listening thread listens for client connections."""
    while server.running:
        client = listener.accept()
        server.logger.debug('Connected to client.')
        if server.running:
            server.clients[client] = set()
            server.sel.register(client, selectors.EVENT_READ, 'from_client')
            server.logger.info('Registered client.')


def send_outgoing(server: DetachedServer) -> None:
    """Outgoing thread forwards messages as they are created."""
    while server.running:
        while len(server.outgoing) > 0:
            outgoing = server.outgoing.pop()
            outgoing[0].send((outgoing[1], outgoing[2]))

            server.logger.debug(f'Sent message {outgoing[1]}.')
            server.logger.log(1, f'{outgoing[2]}\n')
        time.sleep(0)


class DetachedServer:
    """
    BQSKit Runtime Server in detached mode.

    In detached mode, the runtime is started separately from the client. Clients
    can connect and disconnect while not shutting down the server. This
    architecture is designed for the distributed setting, where managers manage
    workers in shared memory and communicate with the server over a network.
    """

    def __init__(
        self,
        ipports: list[tuple[str, int]],
        port: int = default_server_port,
    ) -> None:
        """
        Create a server and connect to the managers at `ipports`.

        Args:
            ipports (list[tuple[str, int]]): The ip and port pairs were
                managers are expected to be waiting.

            port (int): The port this server will listen for clients on.
                Default can be found in the
                :obj:`~bqskit.runtime.default_server_port` global variable.
        """

        self.clients: dict[Connection, set[uuid.UUID]] = {}
        """Tracks all connected clients and all the tasks they have created."""

        self.tasks: dict[uuid.UUID, tuple[int, Connection]] = {}
        """Tracks all active CompilationTasks submitted to the cluster."""

        self.mailbox_to_task_dict: dict[int, uuid.UUID] = {}
        """Used to convert internal RuntimeTasks to client CompilationTasks."""

        self.mailboxes: dict[int, Any] = {}
        """
        A server mailbox is where CompilationTask results are stored.

        This is a mapping from mailbox IDs to mailboxes.
        """

        self.mailbox_counter = 0
        """Counter to ensure all mailboxes have unique IDs."""

        self.managers: list[Connection] = []
        """Maintains one duplex connection with all connected managers."""

        self.manager_resources: list[int] = []
        """Tracks the total number of workers each manager manages."""

        self.sel = selectors.DefaultSelector()
        """Used to efficiently idle and wake when communication is ready."""

        self.running = True
        """True while the server is running."""

        self.logger = logging.getLogger('bqskit-runtime')
        """Logger used to print operational log messages."""

        self.lower_id_bound = 0
        self.upper_id_bound = int(2 ** 30)
        """
        The server starts with an ID range from 0 -> 2^30 ID ranges are then
        assigned to managers by evenly splitting this range.

        Managers then recursively split their range when connecting the sub-
        managers. Finally, workers are assigned specific ids from within this
        range.
        """

        # Connect to managers
        self._connect_to_managers(ipports)

        # Task tracking data structures
        self.total_resources = sum(self.manager_resources)
        self.total_idle_resources = self.total_resources
        self.manager_idle_resources: list[int] = self.manager_resources[:]
        self.logger.info(f'Server has {self.total_resources} workers.')
        self.logger.debug(f'{self.manager_resources = }')

        # Start client listener
        self.port = port
        self.listener = Listener(('0.0.0.0', self.port))
        self.listening_thread = Thread(
            target=listen, args=(self, self.listener),
        )
        self.listening_thread.start()
        self.logger.info(f'Started client listener on port {self.port}.')

        # Start outgoing thread
        self.outgoing: list[tuple[Connection, RuntimeMessage, Any]] = []
        self.outgoing_thread = Thread(target=send_outgoing, args=(self,))
        self.outgoing_thread.start()
        self.logger.info('Started outgoing thread.')

    def _connect_to_managers(self, ipports: list[tuple[str, int]]) -> None:
        """Connect to all managers given by endpoints in `ipports`."""
        d = len(ipports)
        self.step_size = (self.upper_id_bound - self.lower_id_bound) // d
        for i, (ip, port) in enumerate(ipports):
            lb = self.lower_id_bound + (i * self.step_size)
            ub = min(
                self.lower_id_bound + ((i + 1) * self.step_size),
                self.upper_id_bound,
            )
            self._connect_to_manager(ip, port, lb, ub)
            self.logger.info(f'Connected to manager {i} at {ip}:{port}.')
            self.logger.debug(f'Gave bounds {lb=} and {ub=} to manager {i}.')

        for mconn in self.managers:
            msg, payload = mconn.recv()
            assert msg == RuntimeMessage.STARTED
            self.manager_resources.append(payload)
            self.sel.register(mconn, selectors.EVENT_READ, 'from_below')
            self.logger.info(f'Registered manager {i}.')

    def _connect_to_manager(self, ip: str, port: int, lb: int, ub: int) -> None:
        """Connect to a manager at the endpoint given by `ip` and `port`."""
        max_retries = 5
        wait_time = .25
        for _ in range(max_retries):
            try:
                conn = Client((ip, port))
            except ConnectionRefusedError:
                time.sleep(wait_time)
                wait_time *= 2
            else:
                self.managers.append(conn)
                conn.send((RuntimeMessage.CONNECT, (lb, ub)))
                return
        raise RuntimeError(f'Manager connection refused at {ip}:{port}')

    def __del__(self) -> None:
        """Shutdown the server and clean up spawned processes."""
        self._handle_shutdown()

    def _run(self) -> None:
        """Main server loop."""
        self.logger.info('Server running...')
        while self.running:
            try:
                events = self.sel.select(1)  # Say that 5 times fast
                for key, _ in events:
                    conn = cast(Connection, key.fileobj)
                    msg, payload = conn.recv()
                    self.logger.debug(f'Received message {msg}.')
                    self.logger.log(1, f'{payload}\n')

                    if key.data == 'from_client':

                        if msg == RuntimeMessage.CONNECT:
                            pass

                        elif msg == RuntimeMessage.DISCONNECT:
                            self._handle_disconnect(conn)

                        elif msg == RuntimeMessage.SUBMIT:
                            task = cast(CompilationTask, payload)
                            self._handle_new_comp_task(conn, task)

                        elif msg == RuntimeMessage.REQUEST:
                            request = cast(uuid.UUID, payload)
                            self._handle_request(conn, request)

                        elif msg == RuntimeMessage.STATUS:
                            request = cast(uuid.UUID, payload)
                            self._handle_status(conn, request)

                        elif msg == RuntimeMessage.CANCEL:
                            request = cast(uuid.UUID, payload)
                            self._handle_cancel_comp_task(request)

                    elif key.data == 'from_below':

                        if msg == RuntimeMessage.SUBMIT:
                            rtask = cast(RuntimeTask, payload)
                            self._handle_new_task(rtask)

                        elif msg == RuntimeMessage.SUBMIT_BATCH:
                            tasks = cast(List[RuntimeTask], payload)
                            self._handle_new_tasks(tasks)

                        elif msg == RuntimeMessage.RESULT:
                            result = cast(RuntimeResult, payload)
                            self._handle_result(result)

                        elif msg == RuntimeMessage.ERROR:
                            self._handle_error(payload)
                            return

                        elif msg == RuntimeMessage.LOG:
                            self._handle_log(payload)

                        elif msg == RuntimeMessage.CANCEL:
                            self._handle_cancel(payload)

                        else:
                            raise RuntimeError(f'Unknown message type: {msg}')

            except Exception:
                exc_info = sys.exc_info()
                error_str = ''.join(traceback.format_exception(*exc_info))
                self.logger.error(error_str)
                for client in self.clients.keys():
                    client.send((RuntimeMessage.ERROR, error_str))
                self._handle_shutdown()

    def _handle_shutdown(self) -> None:
        """Shutdown the runtime."""
        if not self.running:
            return

        # Stop running
        self.logger.info('Shutting down server.')
        self.running = False

        # Instruct managers to shutdown
        if self.managers is not None:
            for mconn in self.managers:
                try:
                    mconn.send((RuntimeMessage.SHUTDOWN, None))
                    mconn.close()
                except Exception:
                    pass
            self.managers.clear()
            self.managers = None  # type: ignore
            self.logger.debug('Cleared managers.')

        # Close client connections
        if self.clients is not None:
            for client in self.clients.keys():
                try:
                    client.close()
                except Exception:
                    pass
            self.clients.clear()
            self.clients = None  # type: ignore
            self.logger.debug('Cleared clients.')

        # Close selector
        if self.sel is not None:
            self.sel.close()
            self.sel = None  # type: ignore
            self.logger.debug('Cleared selector.')

        # Close listener
        if self.listening_thread is not None:
            # Listener will be blocked on the accept call
            # Create a dummy connection to unblock it
            # Workaround credit: https://stackoverflow.com/questions/16734534
            # Related bug: https://github.com/python/cpython/issues/76425
            dummy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            dummy_socket.connect(('localhost', self.port))
            dummy_socket.close()
            self.listener.close()
            self.listener = None  # type: ignore
            self.listening_thread.join()
            self.listening_thread = None  # type: ignore
            self.logger.debug('Joined listening thread.')

        # Close outgoing thread
        if self.outgoing_thread is not None:
            self.outgoing_thread.join()
            self.outgoing_thread = None  # type: ignore
            self.logger.debug('Joined outgoing thread.')

    def _handle_disconnect(self, conn: Connection) -> None:
        """Disconnect a client connection from the runtime."""
        self.sel.unregister(conn)
        conn.close()
        tasks = self.clients.pop(conn)
        for task_id in tasks:
            self._handle_cancel_comp_task(task_id)
        self.logger.info('Unregistered client.')

    def _handle_new_comp_task(
        self,
        conn: Connection,
        task: CompilationTask,
    ) -> None:
        """Convert a :class:`CompilationTask` into an internal one."""
        mailbox_id = self._get_new_mailbox_id()
        self.tasks[task.task_id] = (mailbox_id, conn)
        self.mailbox_to_task_dict[mailbox_id] = task.task_id
        self.mailboxes[mailbox_id] = [None, False]
        self.logger.info(f'New CompilationTask: {task.task_id}.')

        self.clients[conn].add(task.task_id)

        internal_task = RuntimeTask(
            (CompilationTask.run, (task,), {}),
            RuntimeAddress(-1, mailbox_id, 0),
            mailbox_id,
            tuple(),
            task.logging_level,
            task.max_logging_depth,
        )

        self._handle_new_task(internal_task)

    def _handle_new_task(self, task: RuntimeTask) -> None:
        """Schedule a task on a manager."""
        # Select manager with least tasks
        min_tasks = max(self.manager_idle_resources)
        best_id = self.manager_idle_resources.index(min_tasks)
        self.manager_idle_resources[best_id] -= 1

        # Assign work
        manager = self.managers[best_id]
        self.outgoing.append((manager, RuntimeMessage.SUBMIT, task))

    def _handle_new_tasks(self, tasks: list[RuntimeTask]) -> None:
        """Schedule many tasks between the managers."""
        assignments: list[list[RuntimeTask]] = [[] for _ in self.managers]
        for task in tasks:
            # Select manager
            min_tasks = max(self.manager_idle_resources)
            best_id = self.manager_idle_resources.index(min_tasks)
            self.manager_idle_resources[best_id] -= 1
            assignments[best_id].append(task)

        # Assign work
        for i, assignment in enumerate(assignments):
            if len(assignment) == 0:
                continue

            elif len(assignment) == 1:
                m = (self.managers[i], RuntimeMessage.SUBMIT, assignment[0])
                self.outgoing.append(m)

            else:
                n = (self.managers[i], RuntimeMessage.SUBMIT_BATCH, assignment)
                self.outgoing.append(n)

    def _handle_request(self, conn: Connection, request: uuid.UUID) -> None:
        """Record the requested task, and ship it as soon as it's ready."""
        if request not in self.clients[conn] or request not in self.tasks:
            self.outgoing.append((conn, RuntimeMessage.ERROR, 'Unknown task.'))
            self._handle_disconnect(conn)  # Bad client
            return

        mailbox_id = self.tasks[request][0]
        box = self.mailboxes[mailbox_id]

        # If the result has not arrived yet
        if box[0] is None:
            # Let the server know the client is waiting on this result
            box[1] = True

        # Otherwise the result is already ready
        else:
            self.logger.info(f'Responding to request for task {request}.')
            self.outgoing.append((conn, RuntimeMessage.RESULT, box[0]))
            self.mailboxes.pop(mailbox_id)
            self.clients[conn].remove(request)
            # t_id is not removed from self.tasks or
            # self.mailbox_to_task_dict incase there are left
            # over log messages arriving.

    def _handle_result(self, result: RuntimeResult) -> None:
        """Either store the result here or ship it to the destination worker."""
        w_id = result.completed_by
        m_id = (w_id - self.lower_id_bound) // self.step_size
        self.manager_idle_resources[m_id] += 1
        dest_w_id = result.return_address.worker_id

        # If it isn't for me
        if dest_w_id != -1:
            w_id = result.return_address.worker_id - self.lower_id_bound
            m_id = w_id // self.step_size
            assert m_id >= 0 and m_id < len(self.managers)
            m = (self.managers[m_id], RuntimeMessage.RESULT, result)
            self.outgoing.append(m)

        # If it is for me
        else:
            mailbox_id = result.return_address.mailbox_index
            if mailbox_id not in self.mailboxes:
                return  # Silently discard results from cancelled tasks

            box = self.mailboxes[mailbox_id]
            box[0] = result.result
            t_id = self.mailbox_to_task_dict[mailbox_id]
            self.logger.info(f'Finished: {t_id}.')
            if box[1]:
                self.logger.info(f'Responding to request for task {t_id}.')
                m = (self.tasks[t_id][1], RuntimeMessage.RESULT, box[0])
                self.outgoing.append(m)
                self.clients[self.tasks[t_id][1]].remove(t_id)
                self.mailboxes.pop(mailbox_id)
                # t_id is not removed from self.tasks or
                # self.mailbox_to_task_dict incase there are left
                # over log messages arriving.

    def _handle_status(self, conn: Connection, request: uuid.UUID) -> None:
        """Inform the client if the task is finished or not."""
        if request not in self.clients[conn] or request not in self.tasks:
            conn.send((RuntimeMessage.STATUS, CompilationStatus.UNKNOWN))

        mailbox_id = self.tasks[request][0]
        box = self.mailboxes[mailbox_id]

        if box[0] is None:
            conn.send((RuntimeMessage.STATUS, CompilationStatus.RUNNING))

        else:
            conn.send((RuntimeMessage.STATUS, CompilationStatus.DONE))

    def _handle_cancel_comp_task(self, request: uuid.UUID) -> None:
        """Cancel a compilation task in the system."""
        self.logger.info(f'Cancelling: {request}.')

        # Remove task from server data
        mailbox_id, client_conn = self.tasks[request]  # .pop(request)
        # self.mailbox_to_task_dict.pop(mailbox_id)
        self.mailboxes.pop(mailbox_id)
        self.clients[client_conn].remove(request)

        # Forward internal cancel messages
        addr = RuntimeAddress(-1, mailbox_id, 0)
        self._handle_cancel(addr)

        # Acknowledge the cancel request
        if not client_conn.closed:
            client_conn.send((RuntimeMessage.CANCEL, None))

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Cancel a runtime task in the system."""
        for mconn in self.managers:
            self.outgoing.append((mconn, RuntimeMessage.CANCEL, addr))

    def _handle_error(self, error_payload: tuple[int, str]) -> None:
        """Forward an error to the appropriate client and disconnect it."""
        if not isinstance(error_payload, tuple):
            self._handle_shutdown()
            raise RuntimeError(error_payload)

        tid = error_payload[0]
        conn = self.tasks[self.mailbox_to_task_dict[tid]][1]
        self.outgoing.append((conn, RuntimeMessage.ERROR, error_payload[1]))
        self._handle_disconnect(conn)

    def _handle_log(self, log_payload: tuple[int, LogRecord]) -> None:
        """Forward logs to appropriate client."""
        tid = log_payload[0]
        conn = self.tasks[self.mailbox_to_task_dict[tid]][1]
        self.outgoing.append((conn, RuntimeMessage.LOG, log_payload[1]))

    def _get_new_mailbox_id(self) -> int:
        """Unique mailbox id counter."""
        new_id = self.mailbox_counter
        self.mailbox_counter += 1
        return new_id


def parse_ipports(ipports_str: list[str]) -> list[tuple[str, int]]:
    """Parse command line ip and port inputs."""
    ipports = []
    for ipport_group in ipports_str:
        for ipport in ipport_group.split(','):
            if ipport.strip() == '':
                continue
            comps = ipport.strip().split(':')

            if len(comps) == 1:
                ip, port = comps[0], str(default_manager_port)

            elif len(comps) == 2:
                ip, port = comps

            else:
                raise ValueError(f'Invalid manager address: {ipport}.')

            if not (0 <= int(port) < 65536):
                raise ValueError(f'Invalid port number: {ipport}')

            ipports.append((ip, int(port)))
    return ipports


def start_server() -> None:
    """Entry point for a detached runtime server process."""
    parser = argparse.ArgumentParser(
        prog='BQSKit Server',
        description='Launch a BQSKit runtime server process.',
    )
    parser.add_argument(
        'managers',
        nargs='+',
        help='The ip and port pairs were managers are expected to be waiting.',
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=default_server_port,
        help='The port this server will listen for clients on.',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Enable logging of increasing verbosity, either -v, -vv, or -vvv.',
    )
    args = parser.parse_args()

    # If ips and ports were provided parse them
    ipports = parse_ipports(args.managers)

    # Set up logging
    _logger = logging.getLogger('bqskit-runtime')
    _logger.setLevel([30, 20, 10, 1][min(args.verbose, 3)])
    _logger.addHandler(logging.StreamHandler())

    # Create the server
    server = DetachedServer(ipports, args.port)

    # Force shutdown on interrupt signals
    handle = functools.partial(sigint_handler, server=server)
    signal.signal(signal.SIGINT, handle)

    # Start the server
    server._run()


def sigint_handler(
    signum: int,
    frame: FrameType | None,
    server: DetachedServer,
) -> None:
    server.logger.debug('Server interrupted.')
    server._handle_shutdown()
    exit(-1)
