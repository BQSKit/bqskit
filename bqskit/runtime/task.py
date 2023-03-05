"""This module implements the RuntimeTask class."""
from __future__ import annotations

import inspect
import logging
from typing import Any
from typing import Coroutine

from bqskit.runtime.address import RuntimeAddress


class RuntimeTask:
    """
    BQSKit Runtime's internal task structure.

    The task exists in two phases:
        1. In the initial phase, the task is a context-less function
        and arguments grouped with a return address. It is expected
        when the task is finished executing, to ship the result to
        the return address.

        2. When the task has been scheduled to run on a worker, it is
        initialized into a coroutine that can be executed in steps.
    """

    task_counter = 0
    """Per-process task id counter."""

    def __init__(
        self,
        fnargs: tuple[Any, Any, Any],
        return_address: RuntimeAddress,
        comp_task_id: int,
        breadcrumbs: tuple[RuntimeAddress, ...],
        logging_level: int | None = None,
        max_logging_depth: int = -1,
    ) -> None:
        """Create the task with a new id and return address."""
        RuntimeTask.task_counter += 1
        self.task_id = RuntimeTask.task_counter

        self.fnargs = fnargs
        """Tuple of function pointer, arguments, and keyword arguments."""

        self.return_address = return_address
        """Where the result of this task should be sent."""

        self.logging_level = logging_level
        """Logs with levels >= to this get emitted, if None always emit."""

        self.comp_task_id = comp_task_id
        """The mailbox id of the this task's root task."""

        self.breadcrumbs = breadcrumbs
        """All of this task's parent tasks' addresses in order."""

        self.max_logging_depth = max_logging_depth
        """Logs are not emitted for tasks with this many or more parents."""

        self.coro: Coroutine[Any, Any, Any] | None = None
        """The coroutine containing this tasks code."""

        # self.send: Any = None
        # """A register that both the coroutine and task have access to."""

        self.desired_box_id: int | None = None
        """When waiting on a mailbox, this stores that mailbox's id."""

        self.owned_mailboxes: list[int] = []
        """The mailbox ids that this task owns."""

        self.wake_on_next: bool = False
        """Set to true if this task should wake immediately on a result."""

    def step(self, send_val: Any = None) -> Any:
        """Execute one step of the task."""
        if self.coro is None:
            raise RuntimeError('Task has not been initialized.')

        # Reset previously set await flags
        self.wake_on_next = False
        self.desired_box_id = None

        # Set logging level
        old_level = logging.getLogger().getEffectiveLevel()
        if (
            self.max_logging_depth < 0
            or len(self.breadcrumbs) <= self.max_logging_depth
        ):
            logging.getLogger().setLevel(0)

        # Execute a task step
        to_return = self.coro.send(send_val)

        # Reset logging
        logging.getLogger().setLevel(old_level)

        return to_return

    def start(self) -> None:
        """Initialize the task."""
        self.coro = self.run()

    async def run(self) -> Any:
        """Task coroutine wrapper."""
        if inspect.iscoroutinefunction(self.fnargs[0]):
            return await self.fnargs[0](*self.fnargs[1], **self.fnargs[2])
        return self.fnargs[0](*self.fnargs[1], **self.fnargs[2])

    def is_descendant_of(self, addr: RuntimeAddress) -> bool:
        """Return true if `addr` identifies a parent (or this) task."""
        return addr == self.return_address or addr in self.breadcrumbs
