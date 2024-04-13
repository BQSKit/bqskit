"""This module implements BQSKit Runtime's Worker."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.connection import Client
from multiprocessing.connection import Connection
from queue import Empty
from queue import Queue
from threading import Lock
from threading import Thread
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import Sequence

from bqskit.runtime import default_worker_port
from bqskit.runtime import set_blas_thread_counts
from bqskit.runtime.address import RuntimeAddress
from bqskit.runtime.future import RuntimeFuture
from bqskit.runtime.message import RuntimeMessage
from bqskit.runtime.result import RuntimeResult
from bqskit.runtime.task import RuntimeTask


_logger = logging.getLogger(__name__)


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

    BQSKit Runtime utilizes a dual-threaded worker to accept, execute,
    pause, spawn, resume, and complete tasks in a custom event loop built
    with python's async await mechanisms. Each worker receives and sends
    tasks and results to the greater system through a single duplex
    connection with a runtime server or manager. One thread performs
    work and sends outgoing messages, while the other thread handles
    incoming messages.

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

        self._running = True
        """Controls if the event loop is running."""

        self._mailboxes: dict[int, WorkerMailbox] = {}
        """Map from mailbox ids to worker mailboxes."""

        self._mailbox_counter = 0
        """This count ensures every mailbox has a unique id."""

        self._cache: dict[str, Any] = {}
        """Local worker cache."""

        self.most_recent_read_submit: RuntimeAddress | None = None
        """Tracks the most recently processed submit message from above."""

        self.read_receipt_mutex = Lock()
        """A lock to ensure waiting messages's read receipt is correct."""

        # Send out every client emitted log message upstream
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            active_task = self._active_task
            if not record.name.startswith('bqskit.runtime'):
                if active_task is not None:
                    lvl = active_task.logging_level
                    if lvl is None or lvl <= record.levelno:
                        if lvl <= logging.DEBUG:
                            record.msg += f' [wid={self._id}'
                            items = active_task.log_context.items()
                            if len(items) > 0:
                                record.msg += ', '
                            con_str = ', '.join(f'{k}={v}' for k, v in items)
                            record.msg += con_str
                            record.msg += ']'
                        tid = active_task.comp_task_id
                        self._conn.send((RuntimeMessage.LOG, (tid, record)))
            return record

        logging.setLogRecordFactory(record_factory)

        # Start incoming thread
        self.incoming_thread = Thread(target=self.recv_incoming)
        self.incoming_thread.daemon = True
        self.incoming_thread.start()
        _logger.debug('Started incoming thread.')

        # Communicate that this worker is ready
        self._conn.send((RuntimeMessage.STARTED, self._id))

    def _loop(self) -> None:
        """Main worker event loop."""
        while self._running:
            try:
                self._try_step_next_ready_task()
            except Exception:
                self._running = False
                exc_info = sys.exc_info()
                error_str = ''.join(traceback.format_exception(*exc_info))
                _logger.error(error_str)
                try:
                    self._conn.send((RuntimeMessage.ERROR, error_str))
                except Exception:
                    pass

    def recv_incoming(self) -> None:
        """Continuously receive all incoming messages."""
        while self._running:
            # Receive message
            try:
                msg, payload = self._conn.recv()
            except Exception:
                _logger.debug('Crashed due to lost connection')
                if sys.platform == 'win32':
                    os.kill(os.getpid(), 9)
                else:
                    os.kill(os.getpid(), signal.SIGKILL)
                exit()

            _logger.debug(f'Received message {msg.name}.')
            _logger.log(1, f'Payload: {payload}')

            # Process message
            if msg == RuntimeMessage.SHUTDOWN:
                if sys.platform == 'win32':
                    os.kill(os.getpid(), 9)
                else:
                    os.kill(os.getpid(), signal.SIGKILL)

            elif msg == RuntimeMessage.SUBMIT:
                self.read_receipt_mutex.acquire()
                task = cast(RuntimeTask, payload)
                self.most_recent_read_submit = task.unique_id
                self._add_task(task)
                self.read_receipt_mutex.release()

            elif msg == RuntimeMessage.SUBMIT_BATCH:
                self.read_receipt_mutex.acquire()
                tasks = cast(List[RuntimeTask], payload)
                self.most_recent_read_submit = tasks[0].unique_id
                self._add_task(tasks.pop())  # Submit one task
                self._delayed_tasks.extend(tasks)  # Delay rest
                # Delayed tasks have no context and are stored (more-or-less)
                # as a function pointer together with the arguments.
                # When it gets started, it consumes much more memory,
                # so we delay the task start until necessary (at no cost)
                self.read_receipt_mutex.release()

            elif msg == RuntimeMessage.RESULT:
                result = cast(RuntimeResult, payload)
                self._handle_result(result)

            elif msg == RuntimeMessage.CANCEL:
                addr = cast(RuntimeAddress, payload)
                self._handle_cancel(addr)
                # TODO: preempt?

            elif msg == RuntimeMessage.IMPORTPATH:
                paths = cast(List[str], payload)
                for path in paths:
                    if path not in sys.path:
                        sys.path.append(path)

    def _add_task(self, task: RuntimeTask) -> None:
        """Start a task and add it to the loop."""
        self._tasks[task.return_address] = task
        task.start()
        self._ready_task_ids.put(task.return_address)

    def _handle_result(self, result: RuntimeResult) -> None:
        """Insert result into appropriate mailbox and wake waiting task."""
        assert result.return_address.worker_id == self._id

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
                # print(f'Worker {self._id} is waking task
                # {task.return_address}, with {task.wake_on_next=},
                # {box.ready=}')
                self._ready_task_ids.put(box.dest_addr)  # Wake it
                box.dest_addr = None  # Prevent double wake

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
        # TODO: Send update message?
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
        """Return the next ready task if one exists, otherwise block."""
        while True:
            if self._ready_task_ids.empty() and len(self._delayed_tasks) > 0:
                self._add_task(self._delayed_tasks.pop())
                continue

            self.read_receipt_mutex.acquire()
            try:
                addr = self._ready_task_ids.get_nowait()

            except Empty:
                payload = (1, self.most_recent_read_submit)
                self._conn.send((RuntimeMessage.WAITING, payload))
                self.read_receipt_mutex.release()
                addr = self._ready_task_ids.get()

            else:
                self.read_receipt_mutex.release()

            # Handle a shutdown request that occured while waiting
            if not self._running:
                return None

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
                # TODO: do I need to manually remove addr from self._tasks?
                continue

            return task

    def _try_step_next_ready_task(self) -> None:
        """Select a task to run, and advance it one step."""
        task = self._get_next_ready_task()

        if task is None:
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
            self._conn.send((RuntimeMessage.ERROR, error_payload))

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

        # if future._next_flag:
        #     # Set from Worker.next, implies the task wants the next result
        #     # if box.ready:
        #     #     m = 'Cannot wait for next results on a complete task.'
        #     #     raise RuntimeError(m)
        #     task.wake_on_next = True
        task.wake_on_next = future._next_flag
        # print(f'Worker {self._id} is waiting on task
        # {task.return_address}, with {task.wake_on_next=}')

        if box.ready:
            self._ready_task_ids.put(task.return_address)

    def _process_task_completion(self, task: RuntimeTask, result: Any) -> None:
        """Package and send out task result."""
        assert task is self._active_task
        packaged_result = RuntimeResult(task.return_address, result, self._id)

        if task.return_address not in self._tasks:
            # print(f'Task was cancelled: {task.return_address},
            # {task.fnargs[0].__name__}')
            return

        if task.return_address.worker_id == self._id:
            self._handle_result(packaged_result)
            self._conn.send((RuntimeMessage.UPDATE, -1))
            # Let manager know this worker has one less task
            # without sending a result
        else:
            self._conn.send((RuntimeMessage.RESULT, packaged_result))

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

    def _get_desired_result(self, task: RuntimeTask) -> Any:
        """Retrieve the task's desired result from the mailboxes."""
        if task.desired_box_id is None:
            return None

        box = self._mailboxes[task.desired_box_id]

        if task.wake_on_next:
            fresh_results = box.get_new_results()
            # assert len(fresh_results) > 0
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
        task_name: str | None = None,
        log_context: dict[str, str] = {},
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Submit `fn` as a task to the runtime."""
        assert self._active_task is not None

        if task_name is not None and not isinstance(task_name, str):
            raise RuntimeError('task_name must be a string.')

        if not isinstance(log_context, dict):
            raise RuntimeError('log_context must be a dictionary.')

        for k, v in log_context.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise RuntimeError(
                    'log_context must be a map from strings to strings.',
                )

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
            task_name,
            self._active_task.log_context | log_context,
        )

        # Submit the task (on the next cycle)
        self._conn.send((RuntimeMessage.SUBMIT, task))

        # Return future pointing to the mailbox
        return RuntimeFuture(mailbox_id)

    def map(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_name: Sequence[str | None] | str | None = None,
        log_context: Sequence[dict[str, str]] | dict[str, str] = {},
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Map `fn` over the input arguments distributed across the runtime."""
        assert self._active_task is not None

        if task_name is None or isinstance(task_name, str):
            task_name = [task_name] * len(args[0])

        if len(task_name) != len(args[0]):
            raise RuntimeError(
                'task_name must be a string or a list of strings equal'
                'in length to the number of tasks.',
            )

        if isinstance(log_context, dict):
            log_context = [log_context] * len(args[0])

        if len(log_context) != len(args[0]):
            raise RuntimeError(
                'log_context must be a dictionary or a list of dictionaries'
                ' equal in length to the number of tasks.',
            )

        for context in log_context:
            for k, v in context.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise RuntimeError(
                        'log_context must be a map from strings to strings.',
                    )

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
                task_name[i],
                self._active_task.log_context | log_context[i],
            )
            for i, fnarg in enumerate(fnargs)
        ]

        # Submit the tasks
        self._conn.send((RuntimeMessage.SUBMIT_BATCH, tasks))

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
        for addr in addrs:
            self._conn.send((RuntimeMessage.CANCEL, addr))

    def get_cache(self) -> dict[str, Any]:
        """
        Retrieve worker's local cache.

        Returns:
            (dict[str, Any]): The worker's local cache. This cache can be
                used to store large or unserializable objects within a
                worker process' memory. Passes on the same worker that use
                the same object can load the object from this cache. If
                there are multiple workers, those workers will load their
                own copies of the object into their own cache.
        """
        return self._cache

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
        # if future._done:
        if future.mailbox_id not in self._mailboxes:
            raise RuntimeError('Cannot wait on an already completed result.')

        future._next_flag = True
        next_result_batch = await future
        future._next_flag = False
        return next_result_batch


# Global variable containing reference to this process's worker object.
_worker = None


def start_worker(
    w_id: int | None,
    port: int,
    cpu: int | None = None,
    logging_level: int = logging.WARNING,
    num_blas_threads: int = 1,
) -> None:
    """Start this process's worker."""
    if w_id is not None:
        # Ignore interrupt signals on workers, boss will handle it for us
        # If w_id is None, then we are being spawned separately.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # TODO: check what needs to be done on win

    # Set number of BLAS threads
    set_blas_thread_counts(num_blas_threads)

    # Enforce no default logging
    logging.lastResort = logging.NullHandler()  # type: ignore  # TODO: should I report this as a type bug?  # noqa: E501
    logging.getLogger().handlers.clear()

    # Pin worker to cpu
    if cpu is not None:
        if sys.platform == 'win32':
            raise RuntimeError('Cannot pin worker to cpu on windows.')
        os.sched_setaffinity(0, [cpu])

    # Connect to manager
    max_retries = 7
    wait_time = .1
    conn: Connection | None = None
    family = 'AF_INET' if sys.platform == 'win32' else None
    for _ in range(max_retries):
        try:
            conn = Client(('localhost', port), family)
        except (ConnectionRefusedError, TimeoutError):
            time.sleep(wait_time)
            wait_time *= 2
        else:
            break

    if conn is None:
        raise RuntimeError('Unable to establish connection with manager.')

    # If id isn't provided, wait for assignment
    if w_id is None:
        msg, w_id = conn.recv()
        assert isinstance(w_id, int)
        assert msg == RuntimeMessage.STARTED

    # Set up runtime logging
    _runtime_logger = logging.getLogger('bqskit.runtime')
    _runtime_logger.propagate = False
    _runtime_logger.setLevel(logging_level)
    _handler = logging.StreamHandler()
    _handler.setLevel(0)
    _fmt_header = '%(asctime)s.%(msecs)03d - %(levelname)-8s |'
    _fmt_message = f' [wid={w_id}]: %(message)s'
    _fmt = _fmt_header + _fmt_message
    _formatter = logging.Formatter(_fmt, '%H:%M:%S')
    _handler.setFormatter(_formatter)
    _runtime_logger.addHandler(_handler)

    # Build and start worker
    global _worker
    _worker = Worker(w_id, conn)
    _worker._loop()


def get_worker() -> Worker:
    """Return a handle on this process's worker."""
    if _worker is None:
        raise RuntimeError('Worker has not been started.')
    return _worker


def _check_positive(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            '%s is an invalid positive int value' % value,
        )
    return ivalue


def start_worker_rank() -> None:
    """Entry point for spawning a rank of runtime worker processes."""
    parser = argparse.ArgumentParser(
        prog='bqskit-worker',
        description='Launch a rank of BQSKit runtime worker processes.',
    )
    parser.add_argument(
        'num_workers',
        type=_check_positive,
        help='The number of workers to spawn.',
    )
    parser.add_argument(
        '--cpus', '-c',
        nargs='+',
        type=int,
        help='Either one number or a list of numbers equal in length to the'
        ' number of workers. The workers will be pinned to specified logical'
        ' cpus. If a single-number is given, then all cpu indices are'
        ' enumerated starting at that number.',
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=default_worker_port,
        help='The port the workers will try to connect to a manager on.',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Enable logging of increasing verbosity, either -v, -vv, or -vvv.',
    )
    parser.add_argument(
        '-t', '--num_blas_threads',
        type=int,
        default=1,
        help='The number of threads to use in BLAS libraries.',
    )
    args = parser.parse_args()

    if args.cpus is not None:
        if len(args.cpus) == 1:
            cpus = [args.cpus[0] + i for i in range(args.num_workers)]

        elif len(args.cpus) == args.num_workers:
            cpus = args.cpus

        else:
            raise RuntimeError(
                'The specified logical cpus are invalid. Expected either'
                ' a single number or a list of numbers equal in length to'
                ' the number of workers.',
            )

    else:
        cpus = [None for _ in range(args.num_workers)]

    logging_level = [30, 20, 10, 1][min(args.verbose, 3)]

    # Spawn worker process
    procs = []
    for cpu in cpus:
        pargs = (None, args.port, cpu, logging_level, args.num_blas_threads)
        procs.append(Process(target=start_worker, args=pargs))
        procs[-1].start()

    # Join them
    for proc in procs:
        proc.join()
