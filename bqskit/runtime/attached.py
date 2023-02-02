"""This module implements the AttachedServer runtime."""
import os
from multiprocessing.connection import Listener, Connection, wait
from multiprocessing import Pipe, Process
import signal
import sys
import traceback
from typing import Any, List, cast
import uuid
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask

from bqskit.runtime.worker import start_worker
from bqskit.runtime.message import RuntimeMessage
from bqskit.compiler.task import CompilationTask
from threadpoolctl import threadpool_limits


def _listen_once() -> Connection:
    listener = Listener(('localhost', 7472))
    client = listener.accept()
    listener.close()
    assert client.recv() == (RuntimeMessage.CONNECT, None)
    return client

class AttachedServer:
    """
    BQSKit Runtime Server in attached mode.

    In attached mode, the runtime is owned by a client and is typical
    started and stopped transparently in the background when a Compiler
    object is created and destroyed. In the attached architecture,
    the server spawns and directly manages the workers.
    """

    def __init__(self, num_workers: int = -1) -> None:
        """Create a server with `num_workers` workers."""
        self.tasks: dict[uuid.UUID, int] = {}
        self.mailbox_to_task_dict: dict[int, uuid.UUID] = {}
        self.mailboxes: dict[int, Any] = {}
        self.mailbox_counter = 0

        # Start workers
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.workers: list[tuple[Connection, Process]] = []
        for i in range(num_workers):
            p, q = Pipe()
            self.workers.append((p, Process(target=start_worker, args=(i, q, -1))))
            self.workers[-1][1].start()
        self.num_tasks = [0 for _ in self.workers]

        for wconn, _ in self.workers:
            assert wconn.recv() == ((RuntimeMessage.STARTED, None))

        # Connect to client
        self.client = _listen_once()
    
    def __del__(self) -> None:
        """Shutdown the server and clean up spawned processes."""
        # Instruct workers to shutdown
        for wconn, wproc in self.workers:
            try:
                wconn.send((RuntimeMessage.SHUTDOWN, None))
            except:
                pass
            os.kill(wproc.pid, signal.SIGUSR1)
        
        # Clean up processes
        for _, wproc in self.workers:
            if wproc.exitcode is None:
                os.kill(wproc.pid, signal.SIGKILL)
            wproc.join()

    def _run(self) -> None:
        """Main server loop."""
        connections = [self.client] + [wconn for wconn, _ in self.workers]
        
        try:
            while True:
                for conn in wait(connections):
                    msg, payload = conn.recv()

                    if msg == RuntimeMessage.DISCONNECT:
                        return

                    elif msg == RuntimeMessage.SUBMIT:
                        if isinstance(payload, CompilationTask):
                            self._recieve_new_comp_task(payload)

                        elif isinstance(payload, RuntimeTask):
                            self._recieve_new_task(payload)

                    elif msg == RuntimeMessage.REQUEST:
                        request = cast(uuid.UUID, payload)
                        self._handle_request(request)
                
                    elif msg == RuntimeMessage.SUBMIT_BATCH:
                        tasks = cast(List[RuntimeTask], payload)
                        self._recieve_new_tasks(tasks)
                                    
                    elif msg == RuntimeMessage.RESULT:
                        result = cast(RuntimeResult, payload)
                        self._handle_result(result)
                    
                    elif msg == RuntimeMessage.ERROR:
                        self.client.send((msg, payload[1]))
                        return
                    
                    elif msg == RuntimeMessage.STATUS:
                        self._handle_status(payload)

                    elif msg == RuntimeMessage.LOG:
                        self.client.send((msg, payload[1]))

                    elif msg == RuntimeMessage.CANCEL:
                        self._handle_cancel(payload)

        except Exception as e:
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            self.client.send((RuntimeMessage.ERROR, error_str))

    def _recieve_new_comp_task(self, task: CompilationTask) -> None:
        """Convert a :class:`CompilationTask` into an internal one."""
        mailbox_id = self._get_new_mailbox_id()
        self.tasks[task.task_id] = mailbox_id
        self.mailbox_to_task_dict[mailbox_id] = task.task_id
        self.mailboxes[mailbox_id] = [None, False]
        addr = RuntimeAddress(-1, -1, mailbox_id, 0)
        fnargs = (CompilationTask.run, (task,), {})
        internal_task = RuntimeTask(
            fnargs,
            addr,
            mailbox_id,
            tuple(),
            task.logging_level,
            task.max_logging_depth,
        )
        self._recieve_new_task(internal_task)

    def _recieve_new_task(self, task: RuntimeTask) -> None:
        """Schedule a task on a worker."""
        # select worker
        min_tasks = min(self.num_tasks)
        best_id = self.num_tasks.index(min_tasks)
        self.num_tasks[best_id] += 1

        # assign work
        worker = self.workers[best_id]
        worker[0].send((RuntimeMessage.SUBMIT, task))

    def _recieve_new_tasks(self, tasks: list[RuntimeTask]) -> None:
        """Schedule many tasks between the workers."""
        assignments = [[] for _ in self.workers]
        for task in tasks:
            # select worker
            min_tasks = min(self.num_tasks)
            best_id = self.num_tasks.index(min_tasks)
            self.num_tasks[best_id] += 1
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

    def _handle_request(self, request: uuid.UUID) -> None:
        """Record the requested task, and ship it as soon as it's ready."""
        mailbox_id = self.tasks[request]
        box = self.mailboxes[mailbox_id]
        if box[0] is None:
            box[1] = True
        else:
            self.client.send((RuntimeMessage.RESULT, box[0]))
            self.tasks.pop(request)
            self.mailboxes.pop(mailbox_id)
            self.mailbox_to_task_dict.pop(mailbox_id)

    def _handle_result(self, result: RuntimeResult) -> None:
        """Either store the result here or ship it to the destination worker"""
        self.num_tasks[result.completed_by] -= 1
        dest_worker = result.return_address.worker_id

        if dest_worker != -1:  # If it isn't for me
            msg = (RuntimeMessage.RESULT, result)
            self.workers[dest_worker][0].send(msg)
            return

        # If it is for me
        mailbox_id = result.return_address.mailbox_slot
        box = self.mailboxes[mailbox_id]
        box[0] = result.result
        if box[1]:
            self.client.send((RuntimeMessage.RESULT, box[0]))
            self.tasks.pop(self.mailbox_to_task_dict[mailbox_id])
            self.mailboxes.pop(mailbox_id)
            self.mailbox_to_task_dict.pop(mailbox_id)

    def _handle_status(self, request: uuid.UUID) -> None:
        """Inform the client if the task is finished or not."""
        try:
            mailbox_id = self.tasks[request]
            box = self.mailboxes[mailbox_id]
            self.client.send((RuntimeMessage.STATUS, box[0] is not None))
        except Exception as e:
            msg = f"Invalid task: {request}."
            self.client.send((RuntimeMessage.ERROR, msg))

    def _handle_cancel(self, request: uuid.UUID | RuntimeAddress) -> None:
        """Cancel a compilation task or a runtime task in the system."""
        if isinstance(request, uuid.UUID):
            addr = RuntimeAddress(-1, -1, self.tasks[request], 0)
        else:
            addr = request

        for wconn, p in self.workers:
            os.kill(p.pid, signal.SIGUSR1)
            wconn.send((RuntimeMessage.CANCEL, addr))
        
        self.client.send((RuntimeMessage.CANCEL, addr))

    def _get_new_mailbox_id(self) -> int:
        """Unique mailbox id counter."""
        new_id = self.mailbox_counter
        self.mailbox_counter += 1
        return new_id

def start_attached_server(*args, **kwargs) -> None:
    """Start a runtime server in attached mode."""
    # When the server is started using fork instead of spawn
    # global variables are shared. This can leak erroneous logging
    # configurations into the workers. We clear the information here:
    import logging
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.PlaceHolder):
            continue
        logger.handlers.clear()
    
    # Ignore interrupt signals
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Start and run the server
    with threadpool_limits(limits=1):
        AttachedServer(*args, **kwargs)._run()