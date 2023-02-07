"""This module implements the AttachedServer runtime."""
import os
os.environ['OMP_NUM_THREADS'] = "1"

from multiprocessing.connection import Client, Listener, Connection, wait
from multiprocessing import Pipe, Process
import signal
import sys
import time
import traceback
from typing import List, cast
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask
from bqskit.runtime.worker import start_worker
from bqskit.runtime.message import RuntimeMessage


class Manager:
    """
    BQSKit Runtime Manager.

    A Manager is a middle node in the process hierarchy and is responsible
    for managing workers or other managers. The manager is part of the
    detached architecture. Here managers are started individually as 
    separate processes, which in turn start their own workers. Then,
    if necessary, more managers can be started to manager the level-1
    managers, and so on, until finally, a detached server is started and
    clients can connect.
    """

    def __init__(self, num_workers: int = -1, ipports: list[tuple[str, int]] | None = None) -> None:
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
        self._listen_once()

        self.workers: list[tuple[Connection, Process | None]] = []
        self.worker_resources: list[int] = []

        if ipports is None: # Case 1: spawn and manage workers
            self._spawn_workers(num_workers)

        else: # Case 2: Connect to managers at ipports
            sub_range = (self.upper_id_bound - self.lower_id_bound) // len(ipports)
            for i, (ip, port) in enumerate(ipports):
                lb = self.lower_id_bound + (i*sub_range)
                ub = min(self.lower_id_bound + ((i+1)*sub_range), self.upper_id_bound)
                self._connect_to_manager(ip, port, lb, ub)
        
        # Task tracking data structure
        self.total_resources = sum(self.worker_resources)
        self.total_idle_resources = 0
        self.worker_idle_resources: list[int] = [r for r in self.worker_resources]

        # Ready and inform upstream
        self.upstream.send((RuntimeMessage.STARTED, self.total_resources))

    def _listen_once(self) -> None:
        listener = Listener(('localhost', 7472))
        self.upstream = listener.accept()
        listener.close()

        # Handshake
        msg, payload = self.upstream.recv()
        assert msg == RuntimeMessage.CONNECT
        self.lower_id_bound = payload[0]
        self.upper_id_bound = payload[1]

    def _spawn_workers(self, num_workers = -1):
        if num_workers == -1:
            num_workers = os.cpu_count()
    
        for i in range(num_workers):
            p, q = Pipe()
            if self.lower_id_bound + i == self.upper_id_bound:
                raise RuntimeError("Insufficient id range for workers.")
            self.workers.append((p, Process(target=start_worker, args=(self.lower_id_bound + i, q))))
            self.workers[-1][1].start()
            self.worker_resources.append(1)

        for wconn, _ in self.workers:
            assert wconn.recv() == ((RuntimeMessage.STARTED, None))

    def _connect_to_manager(self, ip, port, lb, ub) -> None:
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
                msg, payload = conn.recv()
                assert msg == RuntimeMessage.STARTED
                self.worker_resources.append(payload)
                return
        raise RuntimeError("Worker connection refused")
    
    def __del__(self) -> None:
        """Shutdown the manager and clean up spawned processes."""
        # Instruct workers to shutdown
        for wconn, wproc in self.workers:
            try:
                wconn.send((RuntimeMessage.SHUTDOWN, None))
            except:
                pass
            if wproc is not None:
                os.kill(wproc.pid, signal.SIGUSR1)
        
        # Clean up processes
        for _, wproc in self.workers:
            if wproc is None:
                continue

            if wproc.exitcode is None:
                os.kill(wproc.pid, signal.SIGKILL)
            wproc.join()

    def _run(self) -> None:
        """Main server loop."""
        connections = [self.upstream] + [wconn for wconn, _ in self.workers]
        
        try:
            while True:
                for conn in wait(connections):
                    msg, payload = conn.recv()

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
                            self._handle_cancel(payload)

                    else:

                        if msg == RuntimeMessage.SUBMIT:
                            task = cast(RuntimeTask, payload)
                            self._recieve_new_task(task)

                        elif msg == RuntimeMessage.SUBMIT_BATCH:
                            self._recieve_new_tasks(tasks)
                                    
                        elif msg == RuntimeMessage.RESULT:
                            result = cast(RuntimeResult, payload)
                            self._handle_result_going_up(result)
                    
                        elif msg == RuntimeMessage.ERROR:
                            self.upstream.send((msg, payload))
                            return

                        elif msg == RuntimeMessage.LOG:
                            self.upstream.send((msg, payload))

                        elif msg == RuntimeMessage.CANCEL:
                            self.upstream.send((msg, payload))

        except Exception as e:
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            self.upstream.send((RuntimeMessage.ERROR, error_str))

    def _recieve_new_task(self, task: RuntimeTask) -> None:
        """Either send the task upstream or schedule it downstream."""
        if self.total_idle_resources > 0:
            self._assign_new_task(task)
        else:
            self.upstream.send((RuntimeMessage.SUBMIT, task))

    def _assign_new_task(self, task: RuntimeTask) -> None:
        """Schedule a task on a worker."""
        # select worker
        min_tasks = min(self.worker_idle_resources)
        best_id = self.worker_idle_resources.index(min_tasks)
        self.worker_idle_resources[best_id] -= 1
        
        # assign work
        worker = self.workers[best_id]
        worker[0].send((RuntimeMessage.SUBMIT, task))

    def _recieve_new_tasks(self, tasks: list[RuntimeTask]) -> None:
        """Either send the tasks upstream or schedule them downstream."""
        if len(tasks) < self.total_idle_resources:
            self._assign_new_tasks(tasks)
        else:
            self.upstream.send((RuntimeMessage.SUBMIT_BATCH, tasks))

    def _assign_new_tasks(self, tasks: list[RuntimeTask]) -> None:
        """Schedule many tasks between the workers."""
        assignments = [[] for _ in self.workers]
        for task in tasks:
            # select worker
            min_tasks = min(self.worker_idle_resources)
            best_id = self.worker_idle_resources.index(min_tasks)
            self.worker_idle_resources[best_id] -= 1
            assignments[best_id].append(task)
        
        # assign work
        for i, assignment in enumerate(assignments):
            if len(assignment) == 0:
                continue

            elif len(assignment) == 1:
                msg = (RuntimeMessage.SUBMIT, assignment[0])
                self.workers[i][0].send(msg)

            else:
                msg = (RuntimeMessage.SUBMIT_BATCH, assignment)
                self.workers[i][0].send(msg)

    def _handle_result_coming_down(self, result: RuntimeResult) -> None:
        wid = result.return_address.worker_id - self.lower_id_bound
        assert wid > 0 and wid < (self.upper_id_bound - self.lower_id_bound)
        self.workers[wid][0].send((RuntimeMessage.RESULT, result))

    def _handle_result_going_up(self, result: RuntimeResult) -> None:
        """Either store the result here or ship it to the destination worker"""
        self.worker_idle_resources[result.completed_by] += 1
        dest_wid = result.return_address.worker_id

        # If it isn't for me
        if dest_wid < self.lower_id_bound or dest_wid >= self.upper_id_bound:
            self.upstream.send((RuntimeMessage.RESULT, result))
            return
        else:
            self._handle_result_coming_down(result)

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Cancel a compilation task or a runtime task in the system."""
        for wconn, p in self.workers:
            if p is not None:
                os.kill(p.pid, signal.SIGUSR1)
            wconn.send((RuntimeMessage.CANCEL, addr))

def start_manager(*args, **kwargs) -> None:
    """Start a runtime manager."""
    Manager(*args, **kwargs)._run()
