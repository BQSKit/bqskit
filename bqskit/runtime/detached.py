"""This module implements the DetachedServer runtime."""
from __future__ import annotations

import os
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
os.environ['OMP_NUM_THREADS'] = '1'


def listen(server: DetachedServer) -> None:
    listener = Listener(('localhost', 7472))
    while True:
        client = listener.accept()
        server.clients[client] = set()
        server.sel.register(client, selectors.EVENT_READ, 'from_client')


class DetachedServer:
    """
    BQSKit Runtime Server in detached mode.

    In detached mode, the runtime is started separately than the client. Clients
    can connect and disconnect but not shutdown a detached server. This
    architecture is designed for the distributed setting, where managers run in
    shared memory on nodes and communicate with the server over the network.
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
        self.running = False

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

    def _connect_to_manager(self, ip: str, port: int, lb: int, ub: int) -> None:
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

    def _handle_shutdown(self) -> None:
        # Stop running
        self.running = False

        # Close client connections
        for client in self.clients.keys():
            client.close()
        self.clients.clear()

        # Instruct managers to shutdown
        for mconn in self.managers:
            try:
                mconn.send((RuntimeMessage.SHUTDOWN, None))
                mconn.close()
            except Exception:
                pass

        self.managers.clear()

    def _run(self) -> None:
        """Main server loop."""
        self.running = True
        try:
            while self.running:
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
                            self._recieve_new_comp_task(conn, task)

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
                            self._recieve_new_task(rtask)

                        elif msg == RuntimeMessage.SUBMIT_BATCH:
                            tasks = cast(List[RuntimeTask], payload)
                            self._recieve_new_tasks(tasks)

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

    def _handle_disconnect(self, conn: Connection) -> None:
        conn.close()
        self.sel.unregister(conn)
        tasks = self.clients.pop(conn)
        for task_id in tasks:
            self._handle_cancel(RuntimeAddress(-1, self.tasks[task_id][0], 0))

    def _recieve_new_comp_task(
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

        self._recieve_new_task(internal_task)

    def _recieve_new_task(self, task: RuntimeTask) -> None:
        """Schedule a task on a manager."""
        # select worker
        min_tasks = max(self.manager_idle_resources)
        best_id = self.manager_idle_resources.index(min_tasks)
        self.manager_idle_resources[best_id] -= 1

        # assign work
        manager = self.managers[best_id]
        manager.send((RuntimeMessage.SUBMIT, task))

    def _recieve_new_tasks(self, tasks: list[RuntimeTask]) -> None:
        """Schedule many tasks between the managers."""
        assignments: list[list[RuntimeTask]] = [[] for _ in self.managers]
        for task in tasks:
            # select manager
            min_tasks = max(self.manager_idle_resources)
            best_id = self.manager_idle_resources.index(min_tasks)
            self.manager_idle_resources[best_id] -= 1
            assignments[best_id].append(task)

        # assign work
        for i, assignment in enumerate(assignments):
            if len(assignment) == 0:
                continue

            elif len(assignment) == 1:
                msg = (RuntimeMessage.SUBMIT, assignment[0])
                self.managers[i].send(msg)

            else:
                msgb = (RuntimeMessage.SUBMIT_BATCH, assignment)
                self.managers[i].send(msgb)

    def _handle_request(self, conn: Connection, request: uuid.UUID) -> None:
        """Record the requested task, and ship it as soon as it's ready."""
        if request not in self.clients[conn] or request not in self.tasks:
            conn.send((RuntimeMessage.ERROR, 'Unknown task.'))
            self._handle_disconnect(conn)  # Bad client
            return

        mailbox_id = self.tasks[request][0]
        box = self.mailboxes[mailbox_id]
        if box[0] is None:
            box[1] = True
        else:
            conn.send((RuntimeMessage.RESULT, box[0]))
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
            self.managers[m_id].send((RuntimeMessage.RESULT, result))
            return

        # If it is for me
        mailbox_id = result.return_address.mailbox_slot
        box = self.mailboxes[mailbox_id]
        box[0] = result.result
        if box[1]:
            t_id = self.mailbox_to_task_dict[mailbox_id]
            self.tasks[t_id][1].send((RuntimeMessage.RESULT, box[0]))
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
        """Cancel a compilation task or a runtime task in the system."""
        addr = RuntimeAddress(-1, self.tasks[request][0], 0)
        self._handle_cancel(addr)
        self.tasks[request][1].send((RuntimeMessage.CANCEL, None))

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Cancel a runtime task in the system."""
        for mconn in self.managers:
            mconn.send((RuntimeMessage.CANCEL, addr))

    def _handle_error(self, error_payload: tuple[int, str]) -> None:
        conn = self.tasks[self.mailbox_to_task_dict[error_payload[0]]][1]
        conn.send((RuntimeMessage.ERROR, error_payload[1]))
        self._handle_disconnect(conn)

    def _handle_log(self, log_payload: tuple[int, LogRecord]) -> None:
        tid = log_payload[0]

        if tid == -1:
            print(log_payload[1].getMessage())  # Dump it in detached mode
            return

        conn = self.tasks[self.mailbox_to_task_dict[tid]][1]
        conn.send((RuntimeMessage.LOG, log_payload[1]))

    def _get_new_mailbox_id(self) -> int:
        """Unique mailbox id counter."""
        new_id = self.mailbox_counter
        self.mailbox_counter += 1
        return new_id


def start_detached_server(*args: Any, **kwargs: Any) -> None:
    """Start a runtime server in detached mode."""
    DetachedServer(*args, **kwargs)._run()
