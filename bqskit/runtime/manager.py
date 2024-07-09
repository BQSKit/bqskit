"""This module implements the Manager class."""
from __future__ import annotations

import argparse
import logging
import selectors
import time
from multiprocessing.connection import Connection
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from bqskit.runtime import default_manager_port
from bqskit.runtime import default_worker_port
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.base import import_tests_package
from bqskit.runtime.base import parse_ipports
from bqskit.runtime.base import ServerBase
from bqskit.runtime.direction import MessageDirection
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


_logger = logging.getLogger(__name__)


class Manager(ServerBase):
    """
    BQSKit Runtime Manager.

    A Manager is a middle node in the process hierarchy and is responsible for
    managing workers or other managers. The manager is part of the detached
    architecture. Here managers are started individually as separate processes,
    which in turn start their own workers. Then, if necessary, more managers can
    be started to manage the level-1 managers, and so on, until finally, a
    detached server is started and clients can connect.
    """

    def __init__(
        self,
        port: int = default_manager_port,
        num_workers: int = -1,
        ipports: list[tuple[str, int]] | None = None,
        worker_port: int = default_worker_port,
        only_connect: bool = False,
        log_level: int = logging.WARNING,
        num_blas_threads: int = 1,
    ) -> None:
        """
        Create a manager instance in one of two ways:

        1) Leave all options default and start a manager which spawns and
            manages as many worker processes as there are os threads.
            You can also specify the number of workers to spawn via
            the `num_workers` parameter. In this mode, the manager
            is a level-1 manager and manages workers.

        2) Specify ip and port pairs, then assume at each endpoint there
            is a listening manager and attempt to establish a connection.
            In this mode, the manager will not spawn any workers and just
            manage the specified managers.

        In either case, if any problems arise during startup, no recovery
        is attempted and the manager terminates.

        Args:
            port (int): The port this manager listens for server connections.
                Default can be found in the
                :obj:`~bqskit.runtime.default_manager_port` global variable.

            num_workers (int): The number of workers to spawn. If -1,
                then spawn as many workers as CPUs on the system.
                (Default: -1). Ignored if `ipports` is not None.

            ipports (list[tuple[str, int]] | None): If not None, then all
                the addresses and ports of running managers to connect to.

            worker_port (int): The port this server will listen for workers
                on. Default can be found in the
                :obj:`~bqskit.runtime.default_worker_port` global variable.

            only_connect (bool): If true, do not spawn workers, only connect
                to already spawned workers.

            log_level (int): The logging level for the manager and workers.
                If `only_connect` is True, doesn't set worker's log level.
                In that case, set the worker's log level when spawning them.
                (Default: logging.WARNING).

            num_blas_threads (int): The number of threads to use in BLAS
                libraries. If `only_connect` is True this is ignored. In
                that case, set the thread count when spawning workers.
                (Default: 1).
        """
        super().__init__()

        # Connect upstream
        self.upstream = self.listen_once('0.0.0.0', port)

        # Handshake with upstream
        msg, payload = self.upstream.recv()
        assert msg == RuntimeMessage.CONNECT
        self.lower_id_bound = payload[0]
        self.upper_id_bound = payload[1]
        self.sel.register(
            self.upstream,
            selectors.EVENT_READ,
            MessageDirection.ABOVE,
        )

        # Case 1: spawn and/or manage workers
        if ipports is None:
            if only_connect:
                self.connect_to_workers(num_workers, worker_port)
            else:
                self.spawn_workers(
                    num_workers,
                    worker_port,
                    log_level,
                    num_blas_threads,
                )

        # Case 2: Connect to detached managers at ipports
        else:
            self.connect_to_managers(ipports)

        # Track info on sent messages to reduce redundant messages:
        self.last_num_idle_sent_up = self.total_workers

        # Track info on received messages to report read receipts:
        self.most_recent_read_submit: RuntimeAddress | None = None

        # Inform upstream we are starting
        msg = (self.upstream, RuntimeMessage.STARTED, self.total_workers)
        self.outgoing.put(msg)
        _logger.info('Sent start message upstream.')

    def handle_message(
        self,
        msg: RuntimeMessage,
        direction: MessageDirection,
        conn: Connection,
        payload: Any,
    ) -> None:
        """Process the message coming from `direction`."""
        if direction == MessageDirection.ABOVE:

            if msg == RuntimeMessage.SUBMIT:
                rtask = cast(RuntimeTask, payload)
                self.most_recent_read_submit = rtask.unique_id
                self.schedule_tasks([rtask])
                # self.update_upstream_idle_workers()

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                rtasks = cast(List[RuntimeTask], payload)
                self.most_recent_read_submit = rtasks[0].unique_id
                self.schedule_tasks(rtasks)
                # self.update_upstream_idle_workers()

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self.send_result_down(result)

            elif msg == RuntimeMessage.CANCEL:
                self.broadcast(RuntimeMessage.CANCEL, payload)

            elif msg == RuntimeMessage.SHUTDOWN:
                self.handle_shutdown()

            elif msg == RuntimeMessage.IMPORTPATH:
                paths = cast(List[str], payload)
                self.handle_importpath(paths)

            elif msg == RuntimeMessage.COMMUNICATE:
                self.broadcast(RuntimeMessage.COMMUNICATE, payload)

            else:
                raise RuntimeError(f'Unexpected message type: {msg.name}')

        elif direction == MessageDirection.BELOW:

            if msg == RuntimeMessage.SUBMIT:
                rtask = cast(RuntimeTask, payload)
                self.send_up_or_schedule_tasks([rtask])

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                rtasks = cast(List[RuntimeTask], payload)
                self.send_up_or_schedule_tasks(rtasks)

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self.handle_result_from_below(result)

            elif msg == RuntimeMessage.WAITING:
                p = cast(Tuple[int, Optional[RuntimeAddress]], payload)
                num_idle, read_receipt = p
                self.handle_waiting(conn, num_idle, read_receipt)
                self.update_upstream_idle_workers()

            elif msg == RuntimeMessage.UPDATE:
                task_diff = cast(int, payload)
                self.handle_update(conn, task_diff)

            else:
                # Forward all other messages up
                self.outgoing.put((self.upstream, msg, payload))

        else:
            raise RuntimeError(f'Unexpected message from {direction.name}.')

    def handle_system_error(self, error_str: str) -> None:
        """
        Handle an error in runtime code as opposed to client code.

        This is called when an error arises in runtime code not in a
        RuntimeTask's coroutine code.
        """
        try:
            self.upstream.send((RuntimeMessage.ERROR, error_str))

            # Sleep to ensure server receives error message before shutdown
            time.sleep(1)

        except Exception:
            # If server has crashed then just exit
            pass

    def get_to_string(self, conn: Connection) -> str:
        """Return a string representation of the connection."""
        if conn == self.upstream:
            return 'BOSS'

        return self.conn_to_employee_dict[conn].recipient_string

    def handle_shutdown(self) -> None:
        """Shutdown the manager and clean up spawned processes."""
        super().handle_shutdown()

        # Forward shutdown message upwards if possible
        try:
            self.upstream.send((RuntimeMessage.SHUTDOWN, None))
            self.upstream.close()

        except Exception:
            # If server has already shutdown or crashed, just exit
            pass

    def send_up_or_schedule_tasks(self, tasks: Sequence[RuntimeTask]) -> None:
        """Either send the tasks upstream or schedule them downstream."""
        num_idle = self.num_idle_workers

        if num_idle != 0:
            self.outgoing.put((self.upstream, RuntimeMessage.UPDATE, num_idle))
            self.schedule_tasks(tasks[:num_idle])
            self.update_upstream_idle_workers()

        if len(tasks) > num_idle:
            self.outgoing.put((
                self.upstream,
                RuntimeMessage.SUBMIT_BATCH,
                tasks[num_idle:],
            ))

    def handle_result_from_below(self, result: RuntimeResult) -> None:
        """Forward the result to its destination and track the completion."""
        # Record a task has been completed
        self.get_employee_responsible_for(result.completed_by).num_tasks -= 1

        # Forward result to final destination
        if self.is_my_worker(result.return_address.worker_id):
            self.send_result_down(result)
            self.outgoing.put((self.upstream, RuntimeMessage.UPDATE, -1))

        else:
            # If its destination worker is not an employee of mine,
            # then my boss will know where to send this result.
            self.outgoing.put((self.upstream, RuntimeMessage.RESULT, result))

    def update_upstream_idle_workers(self) -> None:
        """Update the total number of idle workers upstream."""
        if self.num_idle_workers != self.last_num_idle_sent_up:
            self.last_num_idle_sent_up = self.num_idle_workers
            payload = (self.num_idle_workers, self.most_recent_read_submit)
            m = (self.upstream, RuntimeMessage.WAITING, payload)
            self.outgoing.put(m)

    def handle_update(self, conn: Connection, task_diff: int) -> None:
        """Handle a task count update from a lower level manager or worker."""
        self.conn_to_employee_dict[conn].num_tasks += task_diff
        self.outgoing.put((self.upstream, RuntimeMessage.UPDATE, task_diff))


def start_manager() -> None:
    """Entry point for runtime manager processes."""
    parser = argparse.ArgumentParser(
        prog='bqskit-manager',
        description='Launch a BQSKit runtime manager process.',
    )
    parser.add_argument(
        '-n', '--num-workers',
        default=-1,
        type=int,
        help='The number of workers to spawn. If negative, will spawn'
        ' one worker for each available CPU. Defaults to -1.',
    )
    parser.add_argument(
        '-m', '--managers',
        nargs='+',
        help='The ip and port pairs were managers are expected to be waiting.',
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=default_manager_port,
        help='The port this manager will listen for servers on.',
    )
    parser.add_argument(
        '-w', '--worker-port',
        type=int,
        default=default_worker_port,
        help='The port this manager will listen for workers on.',
    )
    parser.add_argument(
        '-x', '--only-connect',
        action='store_true',
        help='Do not spawn workers, only connect to them.',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Enable logging of increasing verbosity, either -v, -vv, or -vvv.',
    )
    parser.add_argument(
        '-i', '--import-tests',
        action='store_true',
        help='Import the bqskit tests package; used during testing.',
    )
    args = parser.parse_args()

    # If ips and ports were provided parse them
    ipports = None if args.managers is None else parse_ipports(args.managers)

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

    if args.num_workers < -1:
        num_workers = -1
    else:
        num_workers = args.num_workers

    # Create the manager
    manager = Manager(
        args.port,
        num_workers,
        ipports,
        args.worker_port,
        args.only_connect,
    )

    # Start the manager
    manager.run()
