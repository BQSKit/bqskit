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
from typing import Sequence

from bqskit.runtime import default_manager_port
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.base import parse_ipports
from bqskit.runtime.base import ServerBase
from bqskit.runtime.direction import MessageDirection
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


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
        """
        super().__init__()

        # Connect upstream
        self.upstream = self.listen_once(port)

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

        # Case 1: spawn and manage workers
        if ipports is None:
            self.spawn_workers(num_workers)

        # Case 2: Connect to managers at ipports
        else:
            self.connect_to_managers(ipports)

        # Track info on sent messages to reduce redundant messages:
        self.last_num_idle_sent_up = self.total_workers

        # Inform upstream we are starting
        msg = (self.upstream, RuntimeMessage.STARTED, self.total_workers)
        self.outgoing.put(msg)
        self.logger.info('Sent start message upstream.')

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
                self.schedule_tasks([rtask])
                self.update_upstream_idle_workers()

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                rtasks = cast(List[RuntimeTask], payload)
                self.schedule_tasks(rtasks)
                self.update_upstream_idle_workers()

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self.send_result_down(result)

            elif msg == RuntimeMessage.CANCEL:
                addr = cast(RuntimeAddress, payload)
                self.broadcast_cancel(addr)

            elif msg == RuntimeMessage.SHUTDOWN:
                self.handle_shutdown()

            else:
                raise RuntimeError(f'Unexpected message type: {msg.name}')

        elif direction == MessageDirection.BELOW:

            if msg == RuntimeMessage.SUBMIT:
                rtask = cast(RuntimeTask, payload)
                self.send_up_or_schedule_tasks([rtask])
                self.update_upstream_idle_workers()

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                rtasks = cast(List[RuntimeTask], payload)
                self.send_up_or_schedule_tasks(rtasks)
                self.update_upstream_idle_workers()

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self.handle_result_from_below(result)

            elif msg == RuntimeMessage.WAITING:
                num_idle = cast(int, payload)
                self.handle_waiting(conn, num_idle)
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
            m = (self.upstream, RuntimeMessage.WAITING, self.num_idle_workers)
            self.outgoing.put(m)

    def handle_update(self, conn: Connection, task_diff: int) -> None:
        """Handle a task count update from a lower level manager or worker."""
        self.conn_to_employee_dict[conn].num_tasks += task_diff
        self.outgoing.put((self.upstream, RuntimeMessage.UPDATE, task_diff))


def start_manager() -> None:
    """Entry point for runtime manager processes."""
    parser = argparse.ArgumentParser(
        prog='BQSKit Manager',
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
        '--verbose', '-v',
        action='count',
        default=0,
        help='Enable logging of increasing verbosity, either -v, -vv, or -vvv.',
    )
    args = parser.parse_args()

    # If ips and ports were provided parse them
    ipports = None if args.managers is None else parse_ipports(args.managers)

    # Set up logging
    _logger = logging.getLogger('bqskit-runtime')
    _logger.setLevel([30, 20, 10, 1][min(args.verbose, 3)])
    _logger.addHandler(logging.StreamHandler())

    # Create the manager
    manager = Manager(args.port, args.num_workers, ipports)

    # Start the manager
    manager.run()
