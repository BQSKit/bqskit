"""This module implements the Manager class."""
from __future__ import annotations

import argparse
import os
import signal
import sys
from threading import Thread
import time
import traceback
from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from multiprocessing.connection import Listener
from multiprocessing.connection import wait
from typing import Any
from typing import cast
from typing import List

from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask
from bqskit.runtime.worker import start_worker
from bqskit.runtime.detached import send_outgoing
from bqskit.runtime.detached import parse_ipports


class Manager:
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
        port: int = 7473,
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
        """
        # Connect upstream
        self._listen_once(port)

        self.workers: list[tuple[Connection, Process | None]] = []
        self.worker_resources: list[int] = []

        if ipports is None:  # Case 1: spawn and manage workers
            self._spawn_workers(num_workers)

        else:  # Case 2: Connect to managers at ipports
            self._connect_to_managers()

        # Task tracking data structure
        self.total_resources = sum(self.worker_resources)
        self.total_idle_resources = 0
        self.worker_idle_resources: list[int] = [
            r for r in self.worker_resources
        ]

        # Start outgoing thread
        self.running = True
        self.outgoing: list[tuple[Connection, RuntimeMessage, Any]] = []
        self.outgoing_thread = Thread(target=send_outgoing, args=(self,))
        self.outgoing_thread.start()

        # Ready and inform upstream
        msg = (self.upstream, RuntimeMessage.STARTED, self.total_resources)
        self.outgoing.append(msg)
        print("Sent upstream message")

    def _listen_once(self, port: int) -> None:
        listener = Listener(('0.0.0.0', port))
        self.upstream = listener.accept()
        listener.close()

        # Handshake
        msg, payload = self.upstream.recv()
        assert msg == RuntimeMessage.CONNECT
        self.lower_id_bound = payload[0]
        self.upper_id_bound = payload[1]

    def _spawn_workers(self, num_workers: int = -1) -> None:
        if num_workers == -1:
            oscount = os.cpu_count()
            num_workers = oscount if oscount else 1

        for i in range(num_workers):
            if self.lower_id_bound + i == self.upper_id_bound:
                raise RuntimeError('Insufficient id range for workers.')

            p, q = Pipe()
            args = (self.lower_id_bound + i, q)
            proc = Process(target=start_worker, args=args)
            self.workers.append((p, proc))
            self.workers[-1][1].start()  # type: ignore
            self.worker_resources.append(1)

        for wconn, _ in self.workers:
            assert wconn.recv() == ((RuntimeMessage.STARTED, None))
            print("Spawned worker.")
            
        self.step_size = 1

    def _connect_to_managers(self, ipports: list[tuple[str, int]]) -> None:
        d = len(ipports)
        self.step_size = (self.upper_id_bound - self.lower_id_bound) // d
        for i, (ip, port) in enumerate(ipports):
            lb = self.lower_id_bound + (i * self.step_size)
            ub = min(
                self.lower_id_bound + ((i + 1) * self.step_size),
                self.upper_id_bound,
            )
            self._connect_to_manager(ip, port, lb, ub)
        
        for wconn, _ in self.workers:
            msg, payload = wconn.recv()
            assert msg == RuntimeMessage.STARTED
            self.worker_resources.append(payload)

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
                self.workers.append((conn, None))
                conn.send((RuntimeMessage.CONNECT, (lb, ub)))
                return
        raise RuntimeError('Worker connection refused')

    def __del__(self) -> None:
        """Shutdown the manager and clean up spawned processes."""
        self._handle_shutdown()


    def _run(self) -> None:
        """Main server loop."""
        connections = [self.upstream] + [wconn for wconn, _ in self.workers]

        while self.running:
            for fd in wait(connections):
                conn = cast(Connection, fd)
                msg, payload = conn.recv()
                print(msg)

                if conn == self.upstream:

                    if msg == RuntimeMessage.SUBMIT:
                        task = cast(RuntimeTask, payload)
                        self._assign_new_task(task)

                    elif msg == RuntimeMessage.SUBMIT_BATCH:
                        tasks = cast(List[RuntimeTask], payload)
                        self._assign_new_tasks(tasks)

                    elif msg == RuntimeMessage.RESULT:
                        result = cast(RuntimeResult, payload)
                        self._handle_result_coming_down(result)

                    elif msg == RuntimeMessage.CANCEL:
                        addr = cast(RuntimeAddress, payload)
                        self._handle_cancel(addr)
                    
                    elif msg == RuntimeMessage.SHUTDOWN:
                        return

                else:

                    if msg == RuntimeMessage.SUBMIT:
                        task = cast(RuntimeTask, payload)
                        self._recieve_new_task(task)

                    elif msg == RuntimeMessage.SUBMIT_BATCH:
                        tasks = cast(List[RuntimeTask], payload)
                        self._recieve_new_tasks(tasks)

                    elif msg == RuntimeMessage.RESULT:
                        result = cast(RuntimeResult, payload)
                        self._handle_result_going_up(result)

                    else:
                        # Forward all other messages up
                        self.outgoing.append((self.upstream, msg, payload))

    def _handle_shutdown(self) -> None:
        """Shutdown the manager and clean up spawned processes."""
        print("Shut down server")
        # Stop running
        self.running = False

        # Instruct workers to shutdown
        for wconn, _ in self.workers:
            try:
                wconn.send((RuntimeMessage.SHUTDOWN, None))
                wconn.close()
            except Exception:
                pass
        
        for _, p in self.workers:
            if p is not None:
                p.join()
                print("joined worker")
        
        self.workers.clear()

        # Join threads
        self.outgoing_thread.join()

    def _recieve_new_task(self, task: RuntimeTask) -> None:
        """Either send the task upstream or schedule it downstream."""
        if self.total_idle_resources > 0:
            self._assign_new_task(task)
        else:
            self.outgoing.append((self.upstream, RuntimeMessage.SUBMIT, task))

    def _assign_new_task(self, task: RuntimeTask) -> None:
        """Schedule a task on a worker."""
        # select worker
        min_tasks = max(self.worker_idle_resources)
        best_id = self.worker_idle_resources.index(min_tasks)
        self.worker_idle_resources[best_id] -= 1

        # assign work
        worker = self.workers[best_id]
        self.outgoing.append((worker[0], RuntimeMessage.SUBMIT, task))

    def _recieve_new_tasks(self, tasks: list[RuntimeTask]) -> None:
        """Either send the tasks upstream or schedule them downstream."""
        if len(tasks) < self.total_idle_resources:
            self._assign_new_tasks(tasks)
        else:
            msg = (self.upstream, RuntimeMessage.SUBMIT_BATCH, tasks)
            self.outgoing.append(msg)

    def _assign_new_tasks(self, tasks: list[RuntimeTask]) -> None:
        """Schedule many tasks between the workers."""
        assignments: list[list[RuntimeTask]] = [[] for _ in self.workers]
        for task in tasks:
            # select worker
            min_tasks = max(self.worker_idle_resources)
            best_id = self.worker_idle_resources.index(min_tasks)
            self.worker_idle_resources[best_id] -= 1
            assignments[best_id].append(task)

        # assign work
        for i, assignment in enumerate(assignments):
            if len(assignment) == 0:
                continue

            elif len(assignment) == 1:
                m = (self.workers[i][0], RuntimeMessage.SUBMIT, assignment[0])
                self.outgoing.append(m)

            else:
                n = (self.workers[i][0], RuntimeMessage.SUBMIT_BATCH, assignment)
                self.outgoing.append(n)

    def _handle_result_coming_down(self, result: RuntimeResult) -> None:
        wid = (result.return_address.worker_id - self.lower_id_bound) // self.step_size
        assert 0 <= wid < (self.upper_id_bound - self.lower_id_bound)
        msg = (self.workers[wid][0], RuntimeMessage.RESULT, result)
        self.outgoing.append(msg)

    def _handle_result_going_up(self, result: RuntimeResult) -> None:
        """Either store the result here or ship it to the destination worker."""
        w_id = result.completed_by
        m_id = (w_id - self.lower_id_bound) // self.step_size
        self.worker_idle_resources[m_id] += 1
        dest_wid = result.return_address.worker_id

        # If it isn't for me, send it up, other send it down
        if dest_wid < self.lower_id_bound or dest_wid >= self.upper_id_bound:
            msg = (self.upstream, RuntimeMessage.RESULT, result)
            self.outgoing.append(msg)

        else:
            self._handle_result_coming_down(result)

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Cancel a compilation task or a runtime task in the system."""
        for wconn, _ in self.workers:
            self.outgoing.append((wconn, RuntimeMessage.CANCEL, addr))


def start_manager() -> None:
    """Entry point for runtime manager processes."""
    parser = argparse.ArgumentParser(
        prog = 'BQSKit Manager',
        description = 'Launch a BQSKit runtime manager process.',
    )
    parser.add_argument('-n', '--num-workers', default=-1, type=int)
    parser.add_argument('-m', '--managers', nargs='+')
    parser.add_argument('-p', '--port', type=int, default=7473)
    args = parser.parse_args()

    # If ips and ports were provided parse them
    ipports = None if args.managers is None  else parse_ipports(args.managers)

    manager = Manager(args.port, args.num_workers, ipports)
    manager._run()

