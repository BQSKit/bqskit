"""This module implements BQSKit Runtime's Worker."""
from __future__ import annotations

import logging
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Client
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
    expected_num_results: int = 0
    result: Any = None
    num_results: int = 0
    dest_addr: RuntimeAddress | None = None
    fresh_results: list[Any] | None = None

    @property
    def ready(self) -> bool:
        """Return true if the mailbox has all expected results."""
        return (
            self.num_results >= self.expected_num_results
            and self.num_results != 0
        )

    @property
    def has_task_waiting(self) -> bool:
        """Return True if a task is waiting on the result of this box."""
        return self.dest_addr is not None

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

    def get_new_results(self) -> list[tuple[int, Any]]:
        """Return and reset the results that have come in since previous
        call."""
        assert self.fresh_results is not None
        out = self.fresh_results
        self.fresh_results = []
        return out

    def deposit_result(self, result: RuntimeResult) -> None:
        """Store the result in the mailbox."""
        self.num_results += 1
        slot_id = result.return_address.mailbox_slot

        # Record as fresh result
        if self.fresh_results is None:
            self.fresh_results = []
        self.fresh_results.append((slot_id, result.result))

        if self.expecting_single_result:
            self.result = result.result
        else:
            self.result[slot_id] = result.result


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
    operating system-level crashes and errors -- such as seg-faults in
    client code -- the worker will attempt to print a stack trace and
    initiate a system-wide shutdown. However, note that these issues
    can go unhandled and cause the runtime to deadlock or crash.

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

        self._ready_task_ids: Queue[RuntimeAddress] = Queue()
        """Tasks queued up for execution."""

        self._cancelled_task_ids: set[RuntimeAddress] = set()
        """To ensure newly-received cancelled tasks are never started."""

        self._active_task: RuntimeTask | None = None
        """The currently executing task if one is running."""

        self._running = False
        """Controls if the event loop is running."""

        self._mailboxes: dict[int, WorkerMailbox] = {}
        """Map from mailbox ids to worker mailboxes."""

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
        self._conn.send((RuntimeMessage.STARTED, self._id))

    def _loop(self) -> None:
        """Main worker event loop."""
        self._running = True
        while self._running:
            self._try_step_next_ready_task()
            self._try_idle()
            self._handle_comms()

    def _try_idle(self) -> None:
        """If there is nothing to do, wait until we receive a message."""
        empty_outgoing = len(self._outgoing) == 0
        no_ready_tasks = self._ready_task_ids.empty()
        no_delayed_tasks = len(self._delayed_tasks) == 0

        if empty_outgoing and no_ready_tasks and no_delayed_tasks:
            self._conn.send((RuntimeMessage.WAITING, 1))
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
                # Delayed tasks have no context and are stored (more-or-less)
                # as a function pointer together with the arguments.
                # When it gets started, it consumes much more memory,
                # so we delay the task start until necessary (at no cost)

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
        self._ready_task_ids.put(task.return_address)

    def _handle_result(self, result: RuntimeResult) -> None:
        """Insert result into appropriate mailbox and wake waiting task."""
        mailbox_id = result.return_address.mailbox_index
        if mailbox_id not in self._mailboxes:
            # If the mailbox has been dropped due to a cancel, ignore result
            return

        box = self._mailboxes[mailbox_id]
        box.deposit_result(result)

        if box.has_task_waiting:
            assert box.dest_addr is not None
            task = self._tasks[box.dest_addr]

            if task.wake_on_next or box.ready:
                self._ready_task_ids.put(box.dest_addr)  # Wake it

    def _handle_cancel(self, addr: RuntimeAddress) -> None:
        """
        Remove `addr` and its children tasks from this worker.

        Notes:
            Since `self._ready_task_ids' is a queue, it is more efficient
            to discard cancelled tasks when popping from it. Therefore, we
            do not do anything with `self._ready_task_ids` here.

            Also, we also don't need to send out cancel messages for
            cancelled children tasks since other workers can evaluate that
            for themselves using breadcrumbs and the original `addr` cancel
            message.
        """
        self._cancelled_task_ids.add(addr)

        # Remove all tasks that are children of `addr` from initialized tasks
        for key, task in self._tasks.items():
            if task.is_descendant_of(addr):
                for mailbox_id in self._tasks[key].owned_mailboxes:
                    self._mailboxes.pop(mailbox_id)
        self._tasks = {
            a: t for a, t in self._tasks.items()
            if not t.is_descendant_of(addr)
        }

        # Remove all tasks that are children of `addr` from delayed tasks
        self._delayed_tasks = [
            t for t in self._delayed_tasks
            if not t.is_descendant_of(addr)
        ]

    def _get_next_ready_task(self) -> RuntimeTask | None:
        """Return the next ready task if one exists, otherwise None."""
        while True:
            if self._ready_task_ids.empty():
                if len(self._delayed_tasks) > 0:
                    self._add_task(self._delayed_tasks.pop())
                    continue
                return None

            addr = self._ready_task_ids.get()

            if addr in self._cancelled_task_ids or addr not in self._tasks:
                # When a task is cancelled on the worker it is not removed
                # from the ready queue because it is much cheaper to just
                # discard cancelled tasks as they come out.
                continue

            task = self._tasks[addr]

            if any(bcb in self._cancelled_task_ids for bcb in task.breadcrumbs):
                # If any of the selected tasks ancestor tasks are cancelled
                # then discard this one too. Each breadcrumb (bcb) is a
                # task address (unique system-wide task id) of an ancestor
                # task.
                continue

            return task

    def _try_step_next_ready_task(self) -> None:
        """Select a task to run, and advance it one step."""
        task = self._get_next_ready_task()

        if task is None:
            # Nothing to do
            return

        try:
            self._active_task = task

            # Perform a step of the task and get the future it awaits on
            future = task.step(self._get_desired_result(task))

            self._process_await(task, future)

        except StopIteration as e:
            self._process_task_completion(task, e.value)

        except Exception:
            assert self._active_task is not None  # for type checker

            # Bubble up errors
            exc_info = sys.exc_info()
            error_str = ''.join(traceback.format_exception(*exc_info))
            error_payload = (self._active_task.comp_task_id, error_str)
            self._outgoing.append((RuntimeMessage.ERROR, error_payload))

        finally:
            self._active_task = None

    def _process_await(self, task: RuntimeTask, future: RuntimeFuture) -> None:
        """Process a task's await request."""
        if not isinstance(future, RuntimeFuture):
            raise RuntimeError('Can only await on a BQSKit RuntimeFuture.')

        if future.mailbox_id not in self._mailboxes:
            raise RuntimeError('Cannot await on a canceled task.')

        box = self._mailboxes[future.mailbox_id]

        # Let the mailbox know this task is waiting
        box.dest_addr = task.return_address
        task.desired_box_id = future.mailbox_id

        if future._next_flag:
            # Set from Worker.next, implies the task wants the next result
            if box.ready:
                m = 'Cannot wait for next results on a complete task.'
                raise RuntimeError(m)
            task.wake_on_next = True

        elif box.ready:
            self._ready_task_ids.put(task.return_address)

    def _process_task_completion(self, task: RuntimeTask, result: Any) -> None:
        """Package and send out task result."""
        assert task is self._active_task
        packaged_result = RuntimeResult(task.return_address, result, self._id)

        if task.return_address.worker_id == self._id:
            self._handle_result(packaged_result)
            self._outgoing.append((RuntimeMessage.UPDATE, -1))
            # Let manager know this worker has one less task
            # without sending a result
        else:
            self._outgoing.append((RuntimeMessage.RESULT, packaged_result))

        # Remove task
        self._tasks.pop(task.return_address)

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
        if self._ready_task_ids.empty() and len(self._delayed_tasks) > 0:
            self._add_task(self._delayed_tasks.pop())

    def _get_desired_result(self, task: RuntimeTask) -> Any:
        """Retrieve the task's desired result from the mailboxes."""
        if task.desired_box_id is None:
            return None

        box = self._mailboxes[task.desired_box_id]

        if task.wake_on_next:
            fresh_results = box.get_new_results()
            assert len(fresh_results) > 0
            return fresh_results

        assert box.ready
        task.owned_mailboxes.remove(task.desired_box_id)
        return self._mailboxes.pop(task.desired_box_id).result

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

        if len(fnargs) == 0:
            raise RuntimeError('Unable to map 0 tasks.')

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
        assert self._active_task is not None
        num_slots = self._mailboxes[future.mailbox_id].expected_num_results
        self._active_task.owned_mailboxes.remove(future.mailbox_id)
        self._mailboxes.pop(future.mailbox_id)
        addrs = [
            RuntimeAddress(self._id, future.mailbox_id, slot_id)
            for slot_id in range(num_slots)
        ]
        msgs = [(RuntimeMessage.CANCEL, addr) for addr in addrs]
        self._outgoing.extend(msgs)

    async def next(self, future: RuntimeFuture) -> list[tuple[int, Any]]:
        """
        Wait for and return the next batch of results from a map task.

        Returns:
            (list[tuple[int, Any]]): A list of the results that arrived
                since the last time this was called. On the first call,
                all results that have arrived since the task started are
                returned. Each result is paired with the index of its
                arguments in the original map call.
        """
        if future._done:
            raise RuntimeError('Cannot wait on an already completed result.')

        future._next_flag = True
        next_result_batch = await future
        future._next_flag = False
        return next_result_batch


# Global variable containing reference to this process's worker object.
_worker = None


def start_worker(id: int, conn: Connection | int) -> None:
    """Start this process's worker."""
    # Ignore interrupt signals on workers, boss will handle it for us
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Purge all standard python logging configurations
    for _, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.PlaceHolder):
            continue
        logger.handlers.clear()
    logging.Logger.manager.loggerDict = {}

    if isinstance(conn, int):
        # On windows, the workers create a socket connection themselves
        max_retries = 5
        wait_time = .25
        for _ in range(max_retries):
            try:
                assert isinstance(conn, int)
                conn = Client(('localhost', conn), 'AF_INET')
            except ConnectionRefusedError:
                time.sleep(wait_time)
                wait_time *= 2
            else:
                break

    if not isinstance(conn, Connection):
        raise RuntimeError('Unable to establish connection with manager.')

    global _worker
    _worker = Worker(id, conn)
    _worker._loop()


def get_worker() -> Worker:
    """Return a handle on this process's worker."""
    if _worker is None:
        raise RuntimeError('Worker has not been started.')
    return _worker
