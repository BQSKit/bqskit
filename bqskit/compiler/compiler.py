"""This module implements the Compiler class."""
from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import os
import signal
import time
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from types import FrameType
from typing import TYPE_CHECKING

from bqskit.compiler.status import CompilationStatus
from bqskit.runtime.attached import start_attached_server
from bqskit.runtime.message import RuntimeMessage

if TYPE_CHECKING:
    from typing import Any
    from bqskit.compiler.task import CompilationTask
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class Compiler:
    """
    A compiler is responsible for accepting and managing compilation tasks.

    The compiler class either spins up a parallel runtime or connects to
    a distributed one, which compilation tasks can then access to
    parallelize their operations. The compiler is implemented as a
    context manager and it is recommended to use it as one. If the
    compiler is not used in a context manager, it is the responsibility
    of the user to call `close()`.

    Examples:
        1. Use in a context manager:
        >>> with Compiler() as compiler:
        ...     circuit = compiler.compile(task)

        2. Use compiler without context manager:
        >>> compiler = Compiler()
        >>> circuit = compiler.compile(task)
        >>> compiler.close()

        3. Connect to an already running detached runtime:
        >>> with Compiler('localhost', 8786) as compiler:
        ...     circuit = compiler.compile(task)

        4. Start and attach to a runtime with 4 worker processes:
        >>> with Compiler(num_workers=4) as compiler:
        ...     circuit = compiler.compile(task)
    """

    def __init__(
        self,
        ip: None | str = None,
        port: None | int = None,
        num_workers: int = -1,
    ) -> None:
        """Construct a Compiler object."""
        self.p: mp.Process | None = None
        self.conn: Connection | None = None
        if port is None:
            port = 7472

        if ip is None:
            ip = 'localhost'
            self._start_server(num_workers)

        self._connect_to_server(ip, port)

    def _start_server(self, num_workers: int) -> None:
        self.p = mp.Process(target=start_attached_server, args=(num_workers,))
        _logger.debug('Starting runtime server process.')
        self.p.start()

    def _connect_to_server(self, ip: str, port: int) -> None:
        max_retries = 5
        wait_time = .25
        for _ in range(max_retries):
            try:
                conn = Client((ip, port))
            except ConnectionRefusedError:
                time.sleep(wait_time)
                wait_time *= 2
            else:
                self.conn = conn
                self.old_signal = signal.signal(
                    signal.SIGINT, functools.partial(
                        sigint_handler, compiler=self,
                    ),
                )
                if self.conn is None:
                    raise RuntimeError('Connection unexpectedly none.')
                self.conn.send((RuntimeMessage.CONNECT, None))
                _logger.debug('Successfully connected to runtime server.')
                return
        raise RuntimeError('Client connection refused')

    def __enter__(self) -> Compiler:
        """Enter a context for this compiler."""
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """Shutdown compiler."""
        self.close()

    def close(self) -> None:
        """Shutdown the compiler."""
        # Disconnect from server
        if self.conn is not None:
            try:
                self.conn.send((RuntimeMessage.DISCONNECT, None))
                self.conn.close()
            except Exception as e:
                _logger.debug(
                    'Unsuccessfully disconnected from runtime server.',
                )
                _logger.debug(e)
            else:
                _logger.debug('Disconnected from runtime server.')
            finally:
                self.conn = None

        # Shutdown server if attached
        if self.p is not None and self.p.pid is not None:
            try:
                os.kill(self.p.pid, signal.SIGINT)
                _logger.debug('Interrupted attached runtime server.')

                self.p.join(1)
                if self.p.exitcode is None:
                    os.kill(self.p.pid, signal.SIGKILL)
                    _logger.debug('Killed attached runtime server.')

            except Exception:
                _logger.debug(
                    'Error while shuting down attached runtime server.',
                )
            else:
                _logger.debug('Successfully shutdown attached runtime server.')
            finally:
                self.p.join()
                _logger.debug('Attached runtime server is down.')
                self.p = None

        # Reset interrupt signal handler
        signal.signal(signal.SIGINT, self.old_signal)

    def __del__(self) -> None:
        self.close()
        _logger.debug('Compiler successfully shutdown.')

    def _send(
        self,
        msg: RuntimeMessage,
        payload: Any,
    ) -> tuple[RuntimeMessage, Any]:
        if self.conn is None:
            raise RuntimeError('Connection unexpectedly none.')

        try:
            self.conn.send((msg, payload))

            return self._recv_handle_log_error()

        except Exception as e:
            self.conn = None
            raise RuntimeError('Server connection unexpectedly closed.') from e

    def submit(self, task: CompilationTask) -> None:
        """Submit a CompilationTask to the Compiler."""
        if task.logging_level is None:
            task.logging_level = logging.getLogger('bqskit').getEffectiveLevel()

        if self.conn is None:
            raise RuntimeError('Connection unexpectedly none.')

        try:
            # print(f"Compiler submitting task {task.task_id}.")
            self.conn.send((RuntimeMessage.SUBMIT, task))

        except Exception as e:
            self.conn = None
            raise RuntimeError('Server connection unexpectedly closed.') from e

    def status(self, task: CompilationTask) -> CompilationStatus:
        """Retrieve the status of the specified CompilationTask."""
        msg, payload = self._send(RuntimeMessage.STATUS, task.task_id)
        if msg != RuntimeMessage.STATUS:
            raise RuntimeError(f'Unexpected message type: {msg}.')
        return payload

    def result(
        self,
        task: CompilationTask,
    ) -> Circuit | tuple[Circuit, dict[str, Any]]:
        """Block until the CompilationTask is finished, return its result."""
        msg, payload = self._send(RuntimeMessage.REQUEST, task.task_id)
        if msg != RuntimeMessage.RESULT:
            raise RuntimeError(f'Unexpected message type: {msg}.')
        return payload

    def cancel(self, task: CompilationTask) -> bool:
        """Remove a task from the compiler's workqueue."""
        msg, _ = self._send(RuntimeMessage.CANCEL, task.task_id)
        if msg != RuntimeMessage.CANCEL:
            raise RuntimeError(f'Unexpected message type: {msg}.')
        return True

    def compile(
        self,
        task: CompilationTask,
    ) -> Circuit:
        """Submit and execute the CompilationTask, block until its done."""
        _logger.info('Compiling task: %s' % task.task_id)
        self.submit(task)
        result = self.result(task)
        return result  # type: ignore

    def analyze(
        self,
        task: CompilationTask,
    ) -> tuple[Circuit, dict[str, Any]]:
        """Execute a task, return output circuit and final pass data."""
        task.request_data = True
        return self.compile(task)  # type: ignore

    def _recv_handle_log_error(self) -> tuple[RuntimeMessage, Any]:
        """Return next msg, transparently emit log records and raise errors."""
        while self.conn is not None:
            msg, payload = self.conn.recv()
            if msg == RuntimeMessage.LOG:
                logging.getLogger(payload.name).handle(payload)

            elif msg == RuntimeMessage.ERROR:
                raise RuntimeError(payload)

            else:
                return (msg, payload)

        raise RuntimeError('Connection unexpectedly none.')


def sigint_handler(signum: int, frame: FrameType, compiler: Compiler) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _logger.critical('Compiler interrupted.')
    compiler.close()
    exit(-1)
