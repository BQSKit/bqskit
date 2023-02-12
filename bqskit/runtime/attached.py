"""This module implements the AttachedServer runtime."""
from __future__ import annotations

import functools
import os
import selectors
import signal
import uuid
from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing.connection import Connection
from multiprocessing.connection import Listener
from types import FrameType
from typing import Any

from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.detached import DetachedServer
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.worker import start_worker
os.environ['OMP_NUM_THREADS'] = '1'


class AttachedServer(DetachedServer):
    def __init__(self, num_workers: int = -1) -> None:
        """Create a server with `num_workers` workers."""
        self.tasks: dict[uuid.UUID, tuple[int, Connection]] = {}
        self.mailbox_to_task_dict: dict[int, uuid.UUID] = {}
        self.mailboxes: dict[int, Any] = {}
        self.mailbox_counter = 0
        self.managers: list[Connection] = []
        self.manager_resources: list[int] = []
        self.sel = selectors.DefaultSelector()
        self.client_counter = 0
        self.running = False
        self.lower_id_bound = 0
        self.upper_id_bound = int(2e9)
        self.step_size = 1
        self.worker_procs: list[Process] = []

        # Start workers
        self._spawn_workers(num_workers)

        # Task tracking data structure
        self.total_resources = sum(self.manager_resources)
        self.total_idle_resources = self.total_resources
        self.manager_idle_resources: list[int] = [
            r for r in self.manager_resources
        ]

        # Connect to client
        self.clients: dict[Connection, set[uuid.UUID]] = {}
        self._listen_once()

    def _spawn_workers(self, num_workers: int = -1) -> None:
        if num_workers == -1:
            oscount = os.cpu_count()
            num_workers = oscount if oscount else 1

        for i in range(num_workers):
            p, q = Pipe()
            self.managers.append(p)
            self.worker_procs.append(Process(target=start_worker, args=(i, q)))
            self.worker_procs[-1].start()
            self.manager_resources.append(1)

        for wconn in self.managers:
            assert wconn.recv() == ((RuntimeMessage.STARTED, None))
            self.sel.register(wconn, selectors.EVENT_READ, 'from_below')

    def _listen_once(self) -> None:
        listener = Listener(('localhost', 7472))
        client = listener.accept()
        listener.close()

        self.clients[client] = set()
        self.sel.register(client, selectors.EVENT_READ, 'from_client')

    def __del__(self) -> None:
        """Shutdown the server and clean up spawned processes."""
        # Instruct workers to shutdown
        for wconn, wproc in zip(self.managers, self.worker_procs):
            try:
                wconn.send((RuntimeMessage.SHUTDOWN, None))
            except Exception:
                pass

            if wproc.pid is not None:
                os.kill(wproc.pid, signal.SIGUSR1)

        # Clean up processes
        for wproc in self.worker_procs:
            if wproc.exitcode is None and wproc.pid is not None:
                os.kill(wproc.pid, signal.SIGKILL)
            wproc.join()

    def _handle_disconnect(self, conn: Connection) -> None:
        super()._handle_disconnect(conn)
        self.running = False

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Cancel a runtime task in the system."""
        for wconn, wproc in zip(self.managers, self.worker_procs):
            if wproc.pid is not None:
                wconn.send((RuntimeMessage.CANCEL, addr))
                os.kill(wproc.pid, signal.SIGUSR1)


def start_attached_server(*args: Any, **kwargs: Any) -> None:
    """Start a runtime server in attached mode."""
    # When the server is started using fork instead of spawn
    # global variables are shared. This can leak erroneous logging
    # configurations into the workers. We clear the information here:
    import logging
    for _, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.PlaceHolder):
            continue
        logger.handlers.clear()

    # Ignore interrupts on workers (handler is inherited by subprocesses)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Initialize the server
    server = AttachedServer(*args, **kwargs)

    # Force shutdown on interrupt signals
    handle = functools.partial(sigint_handler, server=server)
    signal.signal(signal.SIGINT, handle)

    # Run the server
    server._run()


def sigint_handler(
    signum: int,
    frame: FrameType,
    server: AttachedServer,
) -> None:
    # Clean up workers
    for wproc in server.worker_procs:
        if wproc.exitcode is None and wproc.pid is not None:
            os.kill(wproc.pid, signal.SIGKILL)
        wproc.join()
    exit(-1)
