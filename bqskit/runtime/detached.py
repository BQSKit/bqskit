"""This module implements the DetachedServer runtime."""
from __future__ import annotations

import argparse
import logging
import selectors
import socket
import time
import uuid
from dataclasses import dataclass
from logging import LogRecord
from multiprocessing.connection import Connection
from multiprocessing.connection import Listener
from threading import Thread
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from bqskit.runtime import default_server_port
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.base import import_tests_package
from bqskit.runtime.base import parse_ipports
from bqskit.runtime.base import ServerBase
from bqskit.runtime.direction import MessageDirection
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


_logger = logging.getLogger(__name__)


@dataclass
class ServerMailbox:
    """A mailbox on a server is a final destination for a compilation task."""

    result: Any = None
    """Where the mailbox will store the result once it arrives."""

    client_waiting: bool = False
    """If true, the server knows a client is blocked waiting on this result."""

    @property
    def ready(self) -> bool:
        """Return true if the mailbox result is ready and waiting delivery."""
        return self.result is not None


class DetachedServer(ServerBase):
    """
    BQSKit Runtime Server in detached mode.

    In detached mode, the runtime is started separately from the client. Clients
    can connect and disconnect while not shutting down the server. This
    architecture is designed for the distributed setting, where managers manage
    workers in shared memory and communicate with the server over a network.
    """

    def __init__(
        self,
        ipports: Sequence[tuple[str, int]],
        port: int = default_server_port,
    ) -> None:
        """
        Create a server and connect to the managers at `ipports`.

        Args:
            ipports (list[tuple[str, int]]): The ip and port pairs were
                managers are expected to be listening for server connections.

            port (int): The port this server will listen for clients on.
                Default can be found in the
                :obj:`~bqskit.runtime.default_server_port` global variable.
        """
        super().__init__()

        self.clients: dict[Connection, set[uuid.UUID]] = {}
        """Tracks all connected clients and all the tasks they have created."""

        self.tasks: dict[uuid.UUID, tuple[int, Connection]] = {}
        """Tracks all active CompilationTasks submitted to the cluster."""

        self.mailbox_to_task_dict: dict[int, uuid.UUID] = {}
        """Used to convert internal RuntimeTasks to client CompilationTasks."""

        self.mailboxes: dict[int, ServerMailbox] = {}
        """Mapping from mailbox ids to mailboxes."""

        self.mailbox_counter = 0
        """Counter to ensure all mailboxes have unique IDs."""

        # Connect to managers
        self.connect_to_managers(ipports)

        # Start client listener
        self.port = port
        self.listen_thread = Thread(target=self.listen, args=(port,))
        self.listen_thread.daemon = True
        self.listen_thread.start()
        _logger.info(f'Started client listener on port {self.port}.')

    def listen(self, port: int) -> None:
        """Listening thread listens for client connections."""
        listener = Listener(('0.0.0.0', port))
        while self.running:
            client = listener.accept()

            if self.running:
                # We check again that the server is running before registering
                # the client because dummy data is sent to unblock
                # listener.accept() during server shutdown
                self.clients[client] = set()
                self.sel.register(
                    client,
                    selectors.EVENT_READ,
                    MessageDirection.CLIENT,
                )
                _logger.debug('Connected and registered new client.')

        listener.close()

    def handle_message(
        self,
        msg: RuntimeMessage,
        direction: MessageDirection,
        conn: Connection,
        payload: Any,
    ) -> None:
        """Process the message coming from `direction`."""
        if direction == MessageDirection.CLIENT:

            if msg == RuntimeMessage.CONNECT:
                paths = cast(List[str], payload)
                self.handle_connect(conn, paths)

            elif msg == RuntimeMessage.DISCONNECT:
                self.handle_disconnect(conn)

            elif msg == RuntimeMessage.SUBMIT:
                self.handle_new_comp_task(conn, payload)

            elif msg == RuntimeMessage.REQUEST:
                request = cast(uuid.UUID, payload)
                self.handle_request(conn, request)

            elif msg == RuntimeMessage.STATUS:
                request = cast(uuid.UUID, payload)
                self.handle_status(conn, request)

            elif msg == RuntimeMessage.CANCEL:
                request = cast(uuid.UUID, payload)
                self.handle_cancel_comp_task(request)

            else:
                raise RuntimeError(f'Unexpected message type: {msg.name}')

        elif direction == MessageDirection.BELOW:

            if msg == RuntimeMessage.SUBMIT:
                rtask = cast(RuntimeTask, payload)
                self.schedule_tasks([rtask])

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                rtasks = cast(List[RuntimeTask], payload)
                self.schedule_tasks(rtasks)

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self.handle_result(result)

            elif msg == RuntimeMessage.ERROR:
                self.handle_error(payload)
                return

            elif msg == RuntimeMessage.LOG:
                self.handle_log(payload)

            elif msg == RuntimeMessage.CANCEL:
                self.broadcast(msg, payload)

            elif msg == RuntimeMessage.SHUTDOWN:
                self.handle_shutdown()

            elif msg == RuntimeMessage.WAITING:
                p = cast(Tuple[int, Optional[RuntimeAddress]], payload)
                num_idle, read_receipt = p
                self.handle_waiting(conn, num_idle, read_receipt)

            elif msg == RuntimeMessage.UPDATE:
                task_diff = cast(int, payload)
                self.conn_to_employee_dict[conn].num_tasks += task_diff

            else:
                raise RuntimeError(f'Unexpected message type: {msg.name}')

        else:
            raise RuntimeError(f'Unexpected message from {direction.name}.')

    def handle_connect(self, conn: Connection, paths: list[str]) -> None:
        """Handle a client connection request."""
        self.handle_importpath(paths)
        self.outgoing.put((conn, RuntimeMessage.READY, None))

    def handle_system_error(self, error_str: str) -> None:
        """
        Handle an error in runtime code as opposed to client code.

        This is called when an error arises in runtime code not in a
        RuntimeTask's coroutine code.
        """
        for client in self.clients.keys():
            client.send((RuntimeMessage.ERROR, error_str))

        # Sleep to ensure clients receive error message before shutdown
        time.sleep(1)

    def handle_shutdown(self) -> None:
        """Shutdown the runtime."""
        super().handle_shutdown()

        # Close client connections
        for client in self.clients.keys():
            try:
                client.close()
            except Exception:
                pass
        self.clients.clear()
        _logger.debug('Cleared clients.')

        # Close listener (hasattr checked for attachedserver shutdown)
        if hasattr(self, 'listen_thread') and self.listen_thread.is_alive():
            # Listener will be blocked on the accept call
            # Create a dummy connection to unblock it
            # Workaround credit: https://stackoverflow.com/questions/16734534
            # Related bug: https://github.com/python/cpython/issues/76425
            dummy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            dummy_socket.connect(('localhost', self.port))
            dummy_socket.close()
            self.listen_thread.join()
            _logger.debug('Joined listening thread.')

    def handle_disconnect(self, conn: Connection) -> None:
        """Disconnect a client connection from the runtime."""
        super().handle_disconnect(conn)
        tasks = self.clients.pop(conn)
        for task_id in tasks:
            self.handle_cancel_comp_task(task_id)
        _logger.info('Unregistered client.')

    def handle_new_comp_task(
        self,
        conn: Connection,
        task: Any,  # Explicitly not CompilationTask to avoid early import
    ) -> None:
        """Convert a :class:`CompilationTask` into an internal one."""
        from bqskit.compiler.task import CompilationTask
        mailbox_id = self._get_new_mailbox_id()
        self.tasks[task.task_id] = (mailbox_id, conn)
        self.mailbox_to_task_dict[mailbox_id] = task.task_id
        self.mailboxes[mailbox_id] = ServerMailbox()
        _logger.info(f'New CompilationTask: {task.task_id}.')

        self.clients[conn].add(task.task_id)

        internal_task = RuntimeTask(
            (CompilationTask.run, (task,), {}),
            RuntimeAddress(-1, mailbox_id, 0),
            mailbox_id,
            tuple(),
            task.logging_level,
            task.max_logging_depth,
        )

        self.schedule_tasks([internal_task])

    def handle_request(self, conn: Connection, request: uuid.UUID) -> None:
        """Record the requested task, and ship it as soon as it's ready."""
        if request not in self.clients[conn] or request not in self.tasks:
            self.outgoing.put((conn, RuntimeMessage.ERROR, 'Unknown task.'))
            self.handle_disconnect(conn)  # Bad client
            return

        # Get the mailbox associated with this task.
        mailbox_id = self.tasks[request][0]
        box = self.mailboxes[mailbox_id]

        if box.ready:
            # If the result has already arrived, ship it to the client.
            _logger.info(f'Responding to request for task {request}.')
            self.outgoing.put((conn, RuntimeMessage.RESULT, box.result))
            self.mailboxes.pop(mailbox_id)
            self.clients[conn].remove(request)
            # t_id is not removed from self.tasks or
            # self.mailbox_to_task_dict incase there are left
            # over log messages arriving.

        else:
            # Otherwise, note that the client is waiting on this result.
            # This is so the result will be shipped the instant it arrives.
            box.client_waiting = True

    def handle_status(self, conn: Connection, request: uuid.UUID) -> None:
        """Inform the client if the task is finished or not."""
        from bqskit.compiler.status import CompilationStatus
        if request not in self.clients[conn] or request not in self.tasks:
            # This task is unknown to the system
            m = (conn, RuntimeMessage.STATUS, CompilationStatus.UNKNOWN)
            self.outgoing.put(m)

        # Get the mailbox associated with this task.
        mailbox_id = self.tasks[request][0]
        box = self.mailboxes[mailbox_id]

        # Send status
        s = CompilationStatus.DONE if box.ready else CompilationStatus.RUNNING
        self.outgoing.put((conn, RuntimeMessage.STATUS, s))

    def handle_cancel_comp_task(self, request: uuid.UUID) -> None:
        """Cancel a compilation task in the system."""
        _logger.info(f'Cancelling: {request}.')

        # Remove task from server data
        mailbox_id, client_conn = self.tasks[request]
        self.mailboxes.pop(mailbox_id)
        if client_conn in self.clients:
            self.clients[client_conn].remove(request)

        # Forward internal cancel messages
        addr = RuntimeAddress(-1, mailbox_id, 0)
        self.broadcast(RuntimeMessage.CANCEL, addr)

        # Acknowledge the client's cancel request
        if not client_conn.closed:
            # Check if it closed first since the client may have disconnected
            self.outgoing.put((client_conn, RuntimeMessage.CANCEL, None))

    def handle_result(self, result: RuntimeResult) -> None:
        """Either store the result here or ship it to the destination worker."""
        # Record a task has been completed
        self.get_employee_responsible_for(result.completed_by).num_tasks -= 1

        # Check if the result is for a client
        if result.return_address.worker_id == -1:
            mailbox_id = result.return_address.mailbox_index
            if mailbox_id not in self.mailboxes:
                return  # Silently discard results from cancelled tasks

            box = self.mailboxes[mailbox_id]
            box.result = result.result
            t_id = self.mailbox_to_task_dict[mailbox_id]
            _logger.info(f'Finished: {t_id}.')

            if box.client_waiting:
                _logger.info(f'Responding to request for task {t_id}.')
                m = (self.tasks[t_id][1], RuntimeMessage.RESULT, box.result)
                self.outgoing.put(m)
                self.clients[self.tasks[t_id][1]].remove(t_id)
                self.mailboxes.pop(mailbox_id)
                # t_id is not removed from self.tasks or
                # self.mailbox_to_task_dict incase there are left
                # over log messages arriving.

        else:
            self.send_result_down(result)

    def handle_error(self, error_payload: tuple[int, str]) -> None:
        """Forward an error to the appropriate client and disconnect it."""
        if not isinstance(error_payload, tuple):
            # Internal errors may bubble up without a task_id
            assert isinstance(error_payload, str)
            self.handle_system_error(error_payload)
            self.handle_shutdown()
            raise RuntimeError(error_payload)

        tid = error_payload[0]
        conn = self.tasks[self.mailbox_to_task_dict[tid]][1]
        self.outgoing.put((conn, RuntimeMessage.ERROR, error_payload[1]))
        # TODO: Broadcast cancel to all tasks with compilation task id tid
        # But avoid double broadcasting it. If the client crashes due to
        # this error, which it may not, then we will quickly process
        # a handle_disconnect and call the cancel anyways. We should
        # still cancel here incase the client catches the error and
        # resubmits a job.

    def handle_log(self, log_payload: tuple[int, LogRecord]) -> None:
        """Forward logs to appropriate client."""
        tid = log_payload[0]
        conn = self.tasks[self.mailbox_to_task_dict[tid]][1]
        self.outgoing.put((conn, RuntimeMessage.LOG, log_payload[1]))

    def _get_new_mailbox_id(self) -> int:
        """Unique mailbox id counter."""
        new_id = self.mailbox_counter
        self.mailbox_counter += 1
        return new_id


def start_server() -> None:
    """Entry point for a detached runtime server process."""
    parser = argparse.ArgumentParser(
        prog='bqskit-server',
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
    parser.add_argument(
        '--import-tests', '-i',
        action='store_true',
        help='Import the bqskit tests package; used during testing.',
    )
    args = parser.parse_args()
    ipports = parse_ipports(args.managers)

    # Set up logging
    log_level = [30, 20, 10, 1][min(args.verbose, 3)]
    logging.getLogger().setLevel(log_level)
    _handler = logging.StreamHandler()
    _handler.setLevel(0)
    _fmt_header = '%(asctime)s.%(msecs)03d - %(levelname)-8s |'
    _fmt_message = ' %(module)s: %(message)s'
    _fmt = _fmt_header + _fmt_message
    _formatter = logging.Formatter(_fmt, '%H:%M:%S')
    _handler.setFormatter(_formatter)
    logging.getLogger().addHandler(_handler)

    # Import tests package recursively
    if args.import_tests:
        import_tests_package()

    # Create the server
    server = DetachedServer(ipports, args.port)

    # Start the server
    server.run()
