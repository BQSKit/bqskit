"""This module implements the DetachedServer runtime."""
from __future__ import annotations

import selectors
import sys
import time
import traceback
import uuid
from logging import LogRecord
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from multiprocessing.connection import Listener
from threading import Thread
from typing import Any
from typing import cast
from typing import List

from bqskit.compiler.status import CompilationStatus
from bqskit.compiler.task import CompilationTask
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


def listen(server: DetachedServer) -> None:
    """Listening thread listens for client connections."""
    listener = Listener(('localhost', 7472))
    while server.running:
        client = listener.accept()
        server.clients[client] = set()
        server.sel.register(client, selectors.EVENT_READ, 'from_client')


def send_outgoing(server: DetachedServer) -> None:
    """Outgoing thread forwards messages as they are created."""
    while server.running:
        if len(server.outgoing) > 0:
            outgoing = server.outgoing.pop()
            outgoing[0].send((outgoing[1], outgoing[2]))


class DetachedServer:
    """
    BQSKit Runtime Server in detached mode.

    In detached mode, the runtime is started separately than the client. Clients
    can connect and disconnect rather than shutdown a detached server. This
    architecture is designed for the distributed setting, where managers manage
    workers in shared memory and communicate with the server over a network.
    """

    def __init__(self, ipports: list[tuple[str, int]]) -> None:
        """Create a server and connect to the managers at `ipports`."""
        self.tasks: dict[uuid.UUID, tuple[int, Connection]] = {}
        self.mailbox_to_task_dict: dict[int, uuid.UUID] = {}
        self.mailboxes: dict[int, Any] = {}
        self.mailbox_counter = 0
        self.managers: list[Connection] = []
        self.manager_resources: list[int] = []
        self.sel = selectors.DefaultSelector()
        self.client_counter = 0
        self.running = True

        # Connect to managers
        self.lower_id_bound = 0
        self.upper_id_bound = int(2e9)
        d = len(ipports)
        self.step_size = (self.upper_id_bound - self.lower_id_bound) // d
        for i, (ip, port) in enumerate(ipports):
            lb = self.lower_id_bound + (i * self.step_size)
            ub = min(
                self.lower_id_bound + ((i + 1) * self.step_size),
                self.upper_id_bound,
            )
            self._connect_to_manager(ip, port, lb, ub)

        # Task tracking data structure
        self.total_resources = sum(self.manager_resources)
        self.total_idle_resources = self.total_resources
        self.manager_idle_resources: list[int] = [
            r for r in self.manager_resources
        ]

        # Start listener
        self.clients: dict[Connection, set[uuid.UUID]] = {}
        self.listening_thread = Thread(target=listen, args=(self,))
        self.listening_thread.start()

        # Start outgoing thread
        self.outgoing: list[tuple[Connection, RuntimeMessage, Any]] = []
        self.outgoing_thread = Thread(target=send_outgoing, args=(self,))
        self.outgoing_thread.start()

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
                msg, payload = conn.recv()
                assert msg == RuntimeMessage.STARTED
                self.manager_resources.append(payload)
                self.sel.register(conn, selectors.EVENT_READ, 'from_below')
                return
        raise RuntimeError('Manager connection refused')

    def __del__(self) -> None:
        """Shutdown the server and clean up spawned processes."""
        self._handle_shutdown()

    def _run(self) -> None:
        """Main server loop."""
        while self.running:
            try:
                events = self.sel.select(5)  # Say that 5 times fast
                for key, _ in events:
                    conn = cast(Connection, key.fileobj)
                    msg, payload = conn.recv()

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

            except Exception:
                exc_info = sys.exc_info()
                error_str = ''.join(traceback.format_exception(*exc_info))
                for client in self.clients.keys():
                    client.send((RuntimeMessage.ERROR, error_str))
                self._handle_shutdown()

    def _handle_shutdown(self) -> None:
        """Shutdown the runtime."""
        # Stop running
        self.running = False

        # Instruct managers to shutdown
        for mconn in self.managers:
            try:
                mconn.send((RuntimeMessage.SHUTDOWN, None))
                mconn.close()
            except Exception:
                pass
        self.managers.clear()

        # Close client connections
        for client in self.clients.keys():
            client.close()
        self.clients.clear()

        # Join threads
        self.listening_thread.join()
        self.outgoing_thread.join()

    def _handle_disconnect(self, conn: Connection) -> None:
        """Disconnect a client connection from the runtime."""
        self.sel.unregister(conn)
        conn.close()
        tasks = self.clients.pop(conn)
        for task_id in tasks:
            self._handle_cancel_comp_task(task_id)

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

        self.clients[conn].add(task.task_id)

        internal_task = RuntimeTask(
            (CompilationTask.run, (task,), {}),
            RuntimeAddress(-1, mailbox_id, 0),
            mailbox_id,
            tuple(),
            task.logging_level if task.logging_level else 30,
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
        if box[0] is None:
            box[1] = True
        else:
            self.outgoing.append((conn, RuntimeMessage.RESULT, box[0]))
            self.tasks.pop(request)
            self.mailboxes.pop(mailbox_id)
            self.mailbox_to_task_dict.pop(mailbox_id)
            self.clients[conn].remove(request)

    def _handle_result(self, result: RuntimeResult) -> None:
        """Either store the result here or ship it to the destination worker."""
        self.manager_idle_resources[result.completed_by] += 1
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
            mailbox_id = result.return_address.mailbox_slot
            box = self.mailboxes[mailbox_id]
            box[0] = result.result
            if box[1]:
                t_id = self.mailbox_to_task_dict[mailbox_id]
                m = (self.tasks[t_id][1], RuntimeMessage.RESULT, box[0])
                self.outgoing.append(m)
                self.tasks.pop(t_id)
                self.mailboxes.pop(mailbox_id)
                self.mailbox_to_task_dict.pop(mailbox_id)

    def _handle_status(self, conn: Connection, request: uuid.UUID) -> None:
        """Inform the client if the task is finished or not."""
        if request not in self.clients[conn] or request not in self.tasks:
            conn.send((RuntimeMessage.STATUS, CompilationStatus.UNKNOWN))

        mailbox_id = self.tasks[request][0]
        box = self.mailboxes[mailbox_id]

        if box[0] is None:
            conn.send((RuntimeMessage.STATUS, CompilationStatus.STARTED))

        else:
            conn.send((RuntimeMessage.STATUS, CompilationStatus.DONE))

    def _handle_cancel_comp_task(self, request: uuid.UUID) -> None:
        """Cancel a compilation task in the system."""
        addr = RuntimeAddress(-1, self.tasks[request][0], 0)
        self._handle_cancel(addr)
        client_conn = self.tasks[request][1]
        if not client_conn.closed:
            client_conn.send((RuntimeMessage.CANCEL, None))
        self.tasks.pop(request)

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Cancel a runtime task in the system."""
        for mconn in self.managers:
            self.outgoing.append((mconn, RuntimeMessage.CANCEL, addr))

    def _handle_error(self, error_payload: tuple[int, str]) -> None:
        """Forward an error to the appropriate client and disconnect it."""
        conn = self.tasks[self.mailbox_to_task_dict[error_payload[0]]][1]
        conn.send((RuntimeMessage.ERROR, error_payload[1]))
        self._handle_disconnect(conn)

    def _handle_log(self, log_payload: tuple[int, LogRecord]) -> None:
        """Forward logs to appropriate client."""
        tid = log_payload[0]

        if tid == -1:  # Log message generated outside of an executing task.
            print(log_payload[1].getMessage())  # Dump it in detached mode
            return

        conn = self.tasks[self.mailbox_to_task_dict[tid]][1]
        self.outgoing.append((conn, RuntimeMessage.LOG, log_payload[1]))

    def _get_new_mailbox_id(self) -> int:
        """Unique mailbox id counter."""
        new_id = self.mailbox_counter
        self.mailbox_counter += 1
        return new_id


def start_detached_server(*args: Any, **kwargs: Any) -> None:
    """Start a runtime server in detached mode."""
    DetachedServer(*args, **kwargs)._run()
