"""This module implements the Compiler class."""
from __future__ import annotations
import functools

import logging
import os
import signal
import time
from typing import TYPE_CHECKING
import multiprocessing as mp
from multiprocessing.connection import Client
import uuid
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

    def __init__(self, ip: None | str = None, port: None | int = None, num_workers = -1) -> None:
        """Construct a Compiler object."""
        self.p = None
        self.conn = None
        if ip is None:
            ip = "localhost"
            port = 7472
            self.start_server(num_workers)

        self.connect_to_server(ip, port)
    
    def start_server(self, num_workers: int) -> None:
        self.p = mp.Process(target=start_attached_server, args=(num_workers,))
        _logger.debug('Starting runtime server process.')
        self.p.start()

    def connect_to_server(self, ip, port) -> None:
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
                self.old_signal = signal.signal(signal.SIGINT, functools.partial(sigint_handler, compiler=self))
                self.conn.send((RuntimeMessage.CONNECT, None))
                _logger.debug('Successfully connected to runtime server.')
                return
        raise RuntimeError("Client connection refused")

    def __enter__(self) -> Compiler:
        """Enter a context for this compiler."""
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        """Shutdown compiler."""
        self.close()

    def close(self) -> None:
        """Shutdown compiler."""
        if self.conn is not None:
            try:
                self.conn.send((RuntimeMessage.DISCONNECT, None))
                self.conn.close()
            except Exception as e:
                _logger.debug('Unsuccessfully disconnected from runtime server.')
                _logger.debug(e)
            else:
                _logger.debug('Disconnected from runtime server.')
            finally:
                self.conn = None
        if self.p is not None:
            self.p.join(5)
            if self.p.exitcode is None:
                os.kill(self.p.pid, signal.SIGKILL)
                self.p.join()
            self.p = None
            
    def __del__(self) -> None:
        self.close()
        _logger.debug('Compiler successfully shutdown.')

    def submit(self, task: CompilationTask) -> None:
        """Submit a CompilationTask to the Compiler."""
        task.logging_level = logging.getLogger('bqskit').getEffectiveLevel()
        try:
            # print(f"Compiler submitting task {task.task_id}.")
            self.conn.send((RuntimeMessage.SUBMIT, task))

        except Exception as e:
            self.conn = None
            raise RuntimeError("Server connection unexpectedly closed.") from e

    def status(self, task: CompilationTask) -> bool:
        """
        Retrieve the status of the specified CompilationTask.

        A true value means the task is ready to be retrieved.
        """
        try:
            self.conn.send((RuntimeMessage.STATUS, task.task_id))

            msg, payload = self._recv_handle_log_error()

            if msg == RuntimeMessage.STATUS:
                return payload

            else:
                raise RuntimeError(f"Unexpected message type: {msg}.")

        except Exception as e:
            self.conn = None
            raise RuntimeError("Server connection unexpectedly closed.") from e

    def result(self, task: CompilationTask) -> Circuit | tuple[Circuit, dict[str, Any]]:
        """Block until the CompilationTask is finished, return its result."""
        try:
            # print(f"Compiler requesting task {task.task_id}.")
            self.conn.send((RuntimeMessage.REQUEST, task.task_id))

            msg, payload = self._recv_handle_log_error()
            if msg == RuntimeMessage.RESULT:
                return payload  # type: ignore
            
            else:
                raise RuntimeError(f"Unexpected message type: {msg}.")
                
        except Exception as e:
            self.conn = None
            raise RuntimeError("Server connection unexpectedly closed.") from e

    def cancel(self, task: CompilationTask) -> bool:
        """Remove a task from the compiler's workqueue."""
        try:
            self.conn.send((RuntimeMessage.CANCEL, task.task_id))

            msg, _ = self._recv_handle_log_error()
            if msg == RuntimeMessage.CANCEL:
                return True
            
            else:
                raise RuntimeError(f"Unexpected message type: {msg}.")
                
        except Exception as e:
            self.conn = None
            raise RuntimeError("Server connection unexpectedly closed.") from e

    def compile(self, task: CompilationTask) -> Circuit | tuple[Circuit, dict[str, Any]]:
        """Submit and execute the CompilationTask, block until its done."""
        _logger.info('Compiling task: %s' % task.task_id)
        self.submit(task)
        result = self.result(task)
        return result

    def _recv_handle_log_error(self) -> tuple[RuntimeMessage, Any]:
        """Return next msg, transparently emit log records and raise errors"""
        while True:
            msg, payload = self.conn.recv()
            if msg == RuntimeMessage.LOG:
                logging.getLogger(payload.name).handle(payload)
            
            elif msg == RuntimeMessage.ERROR:
                raise RuntimeError(payload)
            
            else:
                return (msg, payload)

def sigint_handler(signum, frame, compiler):
    _logger.critical("Compiler interrupted.")
    compiler.close()
    exit(-1)
