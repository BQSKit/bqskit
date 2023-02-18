"""This module implements BQSKit Runtime's Worker."""
from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from queue import Queue
from typing import Any
from typing import Callable
from typing import cast
from typing import List

from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.future import RuntimeFuture
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


@dataclass
class WorkerMailbox:
    """
    A mailbox on a worker is a final destination for a task's result.

    When a task is created, a mailbox is also created with an associated future.
    The parent task can await on the future, letting the worker's event loop
    know it is waiting on the associated result. When a result arrives, it is
    placed in the appropriate mailbox and the waiting task is placed into the
    ready queue.
    """
    expecting_single_result: bool = False
    total_num_results: int = 0
    result: Any = None
    num_results: int = 0
    dest_addr: RuntimeAddress | None = None
    wake_on_next: bool = False

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

        If `num_results` is None (by default), then the mailbox will only have
        one slot and expect one result.
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
                lvl = active_task.logging_level
                if lvl is None or lvl <= record.levelno:
                    tid = active_task.comp_task_id
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
        for out_msg in self._outgoing:
            self._conn.send(out_msg)
        self._outgoing.clear()

        # Handle incomming communication
        while self._conn.poll():
            msg, payload = self._conn.recv()

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
        mailbox_id = result.return_address.mailbox_index
        if mailbox_id not in self._mailboxes:
            # If the mailbox has been dropped due to a cancel, ignore result
            return

        box = self._mailboxes[mailbox_id]
        box.num_results += 1
        slot_id = result.return_address.mailbox_slot

        if box.expecting_single_result:
            box.result = result.result
        else:
            box.result[slot_id] = result.result

        # If a task is waiting on this result
        if box.dest_addr is not None:
            task = self._tasks[box.dest_addr]

            if task.wake_on_next:
                if task.send is None:
                    task.send = [(slot_id, result.result)]
                    self._ready_tasks.put(box.dest_addr)
                    # Only first result in, wakes the task

                elif isinstance(task.send, list):
                    task.send.append((slot_id, result.result))
                    # more results may arrive before task starts again

                else:
                    raise RuntimeError("Unexpected send type.")

            elif box.ready:
                self._tasks[box.dest_addr].send = box.result
                self._tasks[box.dest_addr].owned_mailboxes.remove(mailbox_id)
                self._mailboxes.pop(mailbox_id)
                self._ready_tasks.put(box.dest_addr)  # Wake it

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """Remove `addr` and its children tasks from this worker."""
        self._cancelled_tasks.add(addr)

        # Remove all tasks that are children of `addr` from initialized tasks
        to_remove: list[Any] = []
        for key, task in self._tasks.items():
            if task.is_descendant_of(addr):
                to_remove.append(key)

        for key in to_remove:
            for mailbox_id in self._tasks[key].owned_mailboxes:
                self._mailboxes.pop(mailbox_id)
            self._tasks[key].owned_mailboxes.clear()
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
        if not self._running:
            return

        if self._ready_tasks.empty():
            if len(self._delayed_tasks) > 0:
                self._add_task(self._delayed_tasks.pop())
            return

        addr = self._ready_tasks.get()

        if addr in self._cancelled_tasks or addr not in self._tasks:
            return

        task = self._tasks[addr]

        if any(bcb in self._cancelled_tasks for bcb in task.breadcrumbs):
            return

        try:
            self._active_task = task
                
            # Step it
            result = task.step()

            # Handle an await on a RuntimeFuture
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and result[0] in ['BQSKIT_MAIL_ID', 'BQSKIT_WAIT_ID']
            ):
                mailbox_id = result[1]
                if mailbox_id not in self._mailboxes:
                    raise RuntimeError("Cannot await on a canceled task.")
                box = self._mailboxes[mailbox_id]
                if box.ready and result[0] == 'BQSKIT_MAIL_ID':
                    task.send = box.result
                    task.owned_mailboxes.remove(result[1])
                    self._mailboxes.pop(result[1])
                    self._ready_tasks.put(addr)
                else:
                    box.dest_addr = addr
                    if result[0] == 'BQSKIT_WAIT_ID':
                        task.wake_on_next = True
            else:
                raise RuntimeError("Can only await on a BQSKit RuntimeFuture.")

        except StopIteration as e:
            # Task finished running, package and send out result
            task_result = RuntimeResult(addr, e.value, self._id)
            self._outgoing.append((RuntimeMessage.RESULT, task_result))

            # Remove task
            self._tasks.pop(addr)

            # Cancel any open tasks
            for mailbox_id in self._active_task.owned_mailboxes:
                # If task is complete, simply discard result
                if mailbox_id in self._mailboxes:
                    if self._mailboxes[mailbox_id].ready:
                        self._mailboxes.pop(mailbox_id)
                        continue
                # Otherwise send a cancel message
                self.cancel(RuntimeFuture(mailbox_id))

            # Start delayed task
            if self._ready_tasks.empty() and len(self._delayed_tasks) > 0:
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
        self._active_task.owned_mailboxes.append(mailbox_id)

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
        self._active_task.owned_mailboxes.append(mailbox_id)

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

    def cancel(self, future: RuntimeFuture) -> None:
        """Cancel all tasks associated with `future`."""
        num_slots = self._mailboxes[future.mailbox_id].total_num_results
        self._active_task.owned_mailboxes.remove(future.mailbox_id)
        self._mailboxes.pop(future.mailbox_id)
        addrs = [
            RuntimeAddress(self._id, future.mailbox_id, slot_id)
            for slot_id in range(num_slots)
        ]
        msgs = [(RuntimeMessage.CANCEL, addr) for addr in addrs]
        self._outgoing.extend(msgs)
    
    async def wait(self, future: RuntimeFuture) -> list[tuple[int, Any]]:
        """
        Wait for and return the next batch of results from a map task.

        Returns:
            (list[tuple[int, Any]]): A list of the results that arrived
                while the task was waiting. Each result is paired with
                the index of its arguments in the original map call.
        """
        if future.done:
            raise RuntimeError("Cannot wait on an already completed result.")
        future.wait_flag = True
        next_result_batch = await future
        future.wait_flag = False
        return next_result_batch  # type: ignore


# Global variable containing reference to this process's worker object.
_worker = None


def start_worker(*args: Any, **kwargs: Any) -> None:
    """Start this process's worker."""
    global _worker
    _worker = Worker(*args, **kwargs)
    _worker._loop()


def get_worker() -> Worker:
    """Return a handle on this process's worker."""
    if _worker is None:
        raise RuntimeError('Worker has not been started.')
    return _worker
