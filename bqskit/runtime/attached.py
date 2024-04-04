"""This module implements the AttachedServer runtime."""
from __future__ import annotations

import logging
import selectors
import uuid
from multiprocessing.connection import Connection
from typing import Any

from bqskit.runtime import default_server_port
from bqskit.runtime import default_worker_port
from bqskit.runtime.base import ServerBase
from bqskit.runtime.detached import DetachedServer
from bqskit.runtime.detached import ServerMailbox
from bqskit.runtime.direction import MessageDirection


_logger = logging.getLogger(__name__)


class AttachedServer(DetachedServer):
    """
    BQSKit Runtime Server in attached mode.

    In attached mode, the runtime is started by the client. The client owns the
    server and there is only one client on the server. Additionally, the client
    is responsible for shutting down the server. There are no managers in the
    attached architecture, only workers directly managed by a single server.
    This architecture is designed for single-machine shared-memory settings.
    BQSKit will, by default, create, manage, and shutdown an AttachedServer when
    a :class:`~bqskit.compiler.compiler.Compiler` object is created.
    """

    def __init__(
        self,
        num_workers: int = -1,
        port: int = default_server_port,
        worker_port: int = default_worker_port,
        log_level: int = logging.WARNING,
    ) -> None:
        """
        Create a server with `num_workers` workers.

        Args:
            num_workers (int): The number of workers to spawn. If -1,
                then spawn as many workers as CPUs on the system.
                (Default: -1).

            port (int): The port this server will listen for clients on.
                Default can be found in the
                :obj:`~bqskit.runtime.default_server_port` global variable.

            worker_port (int): The port this server will listen for workers
                on. Default can be found in the
                :obj:`~bqskit.runtime.default_worker_port` global variable.
        """
        # Initialize runtime logging
        logging.getLogger().setLevel(log_level)
        _handler = logging.StreamHandler()
        _handler.setLevel(0)
        _fmt_header = '%(asctime)s.%(msecs)03d - %(levelname)-8s |'
        _fmt_message = ' %(module)s: %(message)s'
        _fmt = _fmt_header + _fmt_message
        _formatter = logging.Formatter(_fmt, '%H:%M:%S')
        _handler.setFormatter(_formatter)
        logging.getLogger().addHandler(_handler)

        ServerBase.__init__(self)

        # See DetachedServer for more info on the following fields:
        self.clients: dict[Connection, set[uuid.UUID]] = {}
        self.tasks: dict[uuid.UUID, tuple[int, Connection]] = {}
        self.mailbox_to_task_dict: dict[int, uuid.UUID] = {}
        self.mailboxes: dict[int, ServerMailbox] = {}
        self.mailbox_counter = 0

        # Connect to client
        client_conn = self.listen_once('localhost', port)
        self.clients[client_conn] = set()
        self.sel.register(
            client_conn,
            selectors.EVENT_READ,
            MessageDirection.CLIENT,
        )
        _logger.info('Connected to client.')

        # Start workers
        self.spawn_workers(num_workers, worker_port, log_level)

    def handle_disconnect(self, conn: Connection) -> None:
        """A client disconnect in attached mode is equal to a shutdown."""
        self.handle_shutdown()


def start_attached_server(num_workers: int, **kwargs: Any) -> None:
    """Start a runtime server in attached mode."""
    # Initialize the server
    server = AttachedServer(num_workers, **kwargs)

    # Run the server
    server.run()
