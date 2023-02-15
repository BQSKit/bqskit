"""This module implements BQSKit Runtime's Worker."""
from __future__ import annotations
from dataclasses import dataclass


# Enable low-level fault handling: system crashes print a minimal trace.
import faulthandler
faulthandler.enable()


# Disable multi-threading in BLAS libraries.
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['RUST_BACKTRACE'] = '1'


import logging
import signal
import sys
import time
import traceback
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from queue import Queue
from types import FrameType
from typing import Any
from typing import Callable
from typing import cast
from typing import List

from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.future import RuntimeFuture
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


# Global variable containing reference to this process's worker object.
_worker = None


@dataclass
class WorkerMailbox:
    """
    A mailbox on a worker is a final destination for a task's result.

    When a task is created, a mailbox is also created with an associated
    future. The parent task can await on the future, letting the worker's
    event loop know it is waiting on the associated result. When a result
    arrives, it is placed in the appropriate mailbox and the waiting
    task is placed into the ready queue.
    """
    expecting_single_result: bool = False
    total_num_results: int = 0
    result: Any = None
    num_results: int = 0
    dest_addr: RuntimeAddress | None = None

    @property
    def ready(self) -> bool:
        """Return true if the mailbox has all expected results."""
        return (
            self.num_results >= self.total_num_results
            and self.num_results != 0
        )
    
    @staticmethod
    def new_mailbox(num_results: int | None = None) -> WorkerMailbox:
        """
        Create a new mailbox with `num_results` slots.

        If `num_results` is None (by default), then the mailbox will
        only have one slot and expect one result.
        """
        if num_results is None:
            return WorkerMailbox(True, 1)
        
        return WorkerMailbox(False, num_results, [None] * num_results)


class Worker:
    """
    BQSKit Runtime's Worker.

    BQSKit Runtime utilizes a single-threaded worker to accept, execute,
    pause, spawn, resume, and complete tasks in a custom event loop built
    with python's async await mechanisms. Each worker receives and sends
    tasks and results to the greater system through a single duplex
    connection with a runtime server or manager.

    At start-up, the worker receives an ID and waits for its first task.
    An executing task may use the `submit` and `map` methods to spawn child
    tasks and distribute them across the whole system. Once completed,
    those child tasks will have their results shipped back to the worker
    who created them. When a task awaits a child task, it is removed from
    the ready queue until the desired results come in.

    All created log records are shipped back to the client's process.
    This feature ensures compatibility with applications like jupyter
    that only print messages from the client process's stdout. Additionally,
    it allows BQSKit users seamless integration with the standard python
    logging module. From a user's perspective, they can configure any
    standard python logger from their process like usual and have the
    entire system honor that configuration. Lastly, we do support an
    additional logging option for maximum task depth. Tasks with more
    ancestors than the maximum logging depth will not produce any logs.

    Workers handle python errors by capturing and bubbling them up. For
    system-level crashes and errors, the worker will attempt to print a
    stack trace and initiate a system-wide shutdown; however, note that
    these issues can go unhandled and cause the runtime to deadlock or
    crash.

    Workers perform very minimal scheduling of tasks. Newly created tasks
    are directly forwarded upwards to a manager or server, which in turn
    assigns them to workers. New tasks from above are commonly received
    in batches. In this case, all but one task from a batch is delayed.
    Delayed tasks are moved (in LIFO order) to the ready queue if a worker
    has no other work to complete. This mechanism encourages completing
    deeply-nested tasks first and prevents flooding the system with active
    tasks, which usually require much more memory than delayed ones.
    """

    def __init__(self, id: int, conn: Connection) -> None:
        """
        Initialize a worker with no tasks.

        Args:
            id (int): This worker's id.

            conn (Connection): This worker's duplex channel to a manager
                or a server.
        """
        self._id = id
        self._conn = conn

        self._outgoing: list[tuple[RuntimeMessage, Any]] = []
        """Stores outgoing messages to be handled by the event loop."""

        self._tasks: dict[RuntimeAddress, RuntimeTask] = {}
        """Tracks all started, unfinished tasks on this worker."""

        self._delayed_tasks: list[RuntimeTask] = []
        """Store all delayed tasks in LIFO order."""

        self._ready_tasks: Queue[RuntimeAddress] = Queue()
        """Tasks queued up for execution."""

        self._cancelled_tasks: set[RuntimeAddress] = set()
        """To ensure newly-recieved cancelled tasks are never started."""

        self._active_task: RuntimeTask | None = None
        """Handle on active task if one is running."""

        self._running = False
        """Controls if the loop is running."""

        self._communicating = False
        """True if worker is currently communicating with management."""

        self._mailboxes: dict[int, WorkerMailbox] = {}
        """Mailboxes are used to store expected results."""

        self._mailbox_counter = 0
        """This count ensures every mailbox has a unique id."""

        # Send out every emitted log message upstream
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            active_task = get_worker()._active_task
            if active_task is not None:
                tid = active_task.comp_task_id
            else:
                tid = -1
            self._outgoing.append((RuntimeMessage.LOG, (tid, record)))
            return record

        logging.setLogRecordFactory(record_factory)

        # Communicate that this worker is ready
        self._communicating = True
        self._conn.send((RuntimeMessage.STARTED, None))
        self._communicating = False

    def _loop(self) -> None:
        """Main worker event loop."""
        self._running = True
        while self._running:
            self._try_idle()
            self._handle_comms()
            self._step_next_ready_task()

    def _try_idle(self) -> None:
        """If there is nothing to do, wait until we recieve a message."""
        empty_out_box = len(self._outgoing) == 0
        no_ready_tasks = self._ready_tasks.empty()
        no_delayed_tasks = len(self._delayed_tasks) == 0

        if empty_out_box and no_ready_tasks and no_delayed_tasks:
            wait([self._conn])

    def _handle_comms(self) -> None:
        """Handle all incoming and outgoing messages."""
        # Handle outgoing communication
        self._communicating = True
        # We use self._communication as a gaurd when sending out messages
        # to prevent sending duplicate messages. This could otherwise
        # happen if a cancel signal is handled after `self._conn.send`,
        # but before `self._outgoing.clear`.
        for out_msg in self._outgoing:
            self._conn.send(out_msg)
        self._outgoing.clear()
        self._communicating = False

        # Handle incomming communication
        while True:
            self._communicating = True
            # We use self._communication as a gaurd on os resources.

            if not self._conn.poll():
                self._communicating = False
                break

            msg, payload = self._conn.recv()
            self._communicating = False

            # Process message
            if msg == RuntimeMessage.SHUTDOWN:
                self._running = False
                return

            elif msg == RuntimeMessage.SUBMIT:
                task = cast(RuntimeTask, payload)
                self._add_task(task)

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                tasks = cast(List[RuntimeTask], payload)
                self._add_task(tasks.pop())  # Submit one task
                self._delayed_tasks.extend(tasks)  # Delay rest 

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self._handle_result(result)

            elif msg == RuntimeMessage.CANCEL:
                addr = cast(RuntimeAddress, payload)
                self._handle_cancel(addr)

    def _add_task(self, task: RuntimeTask) -> None:
        """Start a task and add it to the loop."""
        self._tasks[task.return_address] = task
        task.start()
        self._ready_tasks.put(task.return_address)

    def _handle_result(self, result: RuntimeResult) -> None:
        """Insert result into appropriate mailbox and wake waiting task."""
        box = self._mailboxes[result.return_address.mailbox_index]
        box.num_results += 1

        if box.expecting_single_result:
            box.result = result.result
        else:
            box.result[result.return_address.mailbox_slot] = result.result

        if box.ready and box.dest_addr is not None:
            # if task waiting on this result
            self._tasks[box.dest_addr].send = box.result
            self._mailboxes.pop(result.return_address.mailbox_index)
            self._ready_tasks.put(box.dest_addr) # Wake it

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Remove `addr` and its children tasks from this worker."""
        self._cancelled_tasks.add(addr)

        # Remove all tasks that are children of `addr` from initialized tasks
        to_remove: list[Any] = []
        for key, task in self._tasks.items():
            if task.is_descendant_of(addr):
                to_remove.append(key)

        for key in to_remove:
            self._tasks.pop(key)

        # Remove all tasks that are children of `addr` from delayed tasks
        to_remove.clear()
        for task in self._delayed_tasks:
            if task.is_descendant_of(addr):
                to_remove.append(task)

        for task in to_remove:
            self._delayed_tasks.remove(task)

    def _step_next_ready_task(self) -> None:
        """Select a task to run, and advance it one step."""

        # Get next ready task
        # This is done in a lockless way in regards to the cancel signal
        # handler such that the active task should never be cancelled.
        addr = None
        task = None
        while True:
            if self._ready_tasks.empty():
                if len(self._delayed_tasks) > 0:
                    self._add_task(self._delayed_tasks.pop())
                return

            _addr = self._ready_tasks.get()

            if _addr in self._cancelled_tasks:
                continue

            try:
                _task = self._tasks[_addr]
            except KeyError:
                # _addr can be removed from self._tasks by a signal handler
                continue

            if any(bcb in self._cancelled_tasks for bcb in _task.breadcrumbs):
                continue

            addr, task = _addr, _task
            break

        assert addr is not None and task is not None

        try:
            self._active_task = task

            # Check again to ensure cancelled task is not started
            if task.return_address in self._cancelled_tasks:
                return

            if any(bcb in self._cancelled_tasks for bcb in task.breadcrumbs):
                return

            # Set logging level
            if len(task.breadcrumbs) <= task.max_logging_depth:
                logging.getLogger().setLevel(task.logging_level)
            else:
                logging.getLogger().setLevel(30)

            # Step it
            result = task.step()

            # Reset send value, if set
            if task.send is not None:
                task.send = None

            # Handle an await on a RuntimeFuture
            if isinstance(result, tuple) and result[0] == 'BQSKIT_MAIL_ID':
                box = self._mailboxes[result[1]]
                if box.ready:
                    task.send = box.result
                    self._mailboxes.pop(result[1])
                    self._ready_tasks.put(addr)
                else:
                    box.dest_addr = addr

            else:
                # If not waiting on a RuntimeFuture then ready to run again
                self._ready_tasks.put(addr)

        except StopIteration as e:
            # Task finished running, package and send out result
            task_result = RuntimeResult(addr, e.value, self._id)
            self._outgoing.append((RuntimeMessage.RESULT, task_result))

            # Remove task
            self._tasks.pop(addr)

            # Start delayed task
            if len(self._delayed_tasks) > 0:  # TODO: try only adding if ready queue is empty
                self._add_task(self._delayed_tasks.pop())

        except RuntimeCancelException:
            # Start delayed task
            if len(self._delayed_tasks) > 0:
                self._add_task(self._delayed_tasks.pop())

        except Exception:
            assert self._active_task is not None

            # Bubble up errors
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            self._outgoing.append(
                (
                    RuntimeMessage.ERROR,
                    (self._active_task.comp_task_id, error_str),
                ),
            )

        finally:
            self._active_task = None

    def _get_new_mailbox_id(self) -> int:
        """Return a new unique mailbox id."""
        new_id = self._mailbox_counter
        self._mailbox_counter += 1
        return new_id

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Submit `fn` as a task to the runtime."""
        assert self._active_task is not None
        # Group fnargs together
        fnarg = (fn, args, kwargs)

        # Create a new mailbox
        mailbox_id = self._get_new_mailbox_id()
        self._mailboxes[mailbox_id] = WorkerMailbox.new_mailbox()

        # Create the task
        task = RuntimeTask(
            fnarg,
            RuntimeAddress(self._id, mailbox_id, 0),
            self._active_task.comp_task_id,
            self._active_task.breadcrumbs + (self._active_task.return_address,),
            self._active_task.logging_level,
            self._active_task.max_logging_depth,
        )

        # Submit the task (on the next cycle)
        self._outgoing.append((RuntimeMessage.SUBMIT, task))

        # Return future pointing to the mailbox
        return RuntimeFuture(mailbox_id)

    def map(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Map `fn` over the input arguments distributed across the runtime."""
        assert self._active_task is not None
        # Group fnargs together
        fnargs = []
        if len(args) == 1:
            for arg in args[0]:
                fnargs.append((fn, (arg,), kwargs))

        else:
            for subargs in zip(*args):
                fnargs.append((fn, subargs, kwargs))

        # Create a new mailbox
        mailbox_id = self._get_new_mailbox_id()
        self._mailboxes[mailbox_id] = WorkerMailbox.new_mailbox(len(fnargs))

        # Create the tasks
        breadcrumbs = self._active_task.breadcrumbs
        breadcrumbs += (self._active_task.return_address,)
        tasks = [
            RuntimeTask(
                fnarg,
                RuntimeAddress(self._id, mailbox_id, i),
                self._active_task.comp_task_id,
                breadcrumbs,
                self._active_task.logging_level,
                self._active_task.max_logging_depth,
            )
            for i, fnarg in enumerate(fnargs)
        ]

        # Submit the tasks
        self._outgoing.append((RuntimeMessage.SUBMIT_BATCH, tasks))

        # Return future pointing to the mailbox
        return RuntimeFuture(mailbox_id)


class RuntimeCancelException(Exception):
    """Raised to pre-empt a cancelled task."""
    pass


def cancel_signal_handler(signum: int, frame: FrameType | None) -> Any:
    """Pause the worker, handle cancel messages, and resume worker."""
    if _worker is None:
        return

    # Pre-empt to help ensure cancel message arrival
    time.sleep(0)

    if _worker._communicating:
        # If the worker is currently sending out messages, then it will
        # process the cancel message before stepping any task.
        return

    _worker._handle_comms() # Expecting a cancel message; process it now

    # if active task is cancelled, pre-empt it
    if _worker._in_task:
        cur_task = _worker._active_task
        if cur_task is not None:
            cur_task_cancelled = cur_task.return_address in _worker._cancelled_tasks
            cur_task_ancestor_cancelled = any(
                bcb in _worker._cancelled_tasks
                for bcb in cur_task.breadcrumbs
            )
            if cur_task_cancelled or cur_task_ancestor_cancelled:
                _worker._in_task = False
                raise RuntimeCancelException()


def start_worker(*args: Any, **kwargs: Any) -> None:
    """Start this process's worker."""
    signal.signal(signal.SIGUSR1, cancel_signal_handler)
    global _worker
    _worker = Worker(*args, **kwargs)
    _worker._loop()


def get_worker() -> Worker:
    """Return a handle on this process's worker."""
    if _worker is None:
        raise RuntimeError('Worker has not been started.')
    return _worker
