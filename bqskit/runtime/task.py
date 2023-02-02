"""This module implements the RuntimeTask class."""
import inspect
from typing import Any

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
        fnargs: Any,
        return_address: RuntimeAddress,
        comp_task_id: int,
        breadcrumbs: tuple[RuntimeAddress],
        logging_level = 30,
        max_logging_depth = -1,
    ) -> None:
        """Create the task with a new id and return address."""
        RuntimeTask.task_counter += 1
        self.task_id = RuntimeTask.task_counter
        self.fnargs = fnargs
        self.return_address = return_address
        self.logging_level = logging_level
        self.comp_task_id = comp_task_id
        self.breadcrumbs = breadcrumbs
        self.max_logging_depth = max_logging_depth
        self.coro = None
        self.send = None

    def step(self):
        """Execute one step of the task."""
        if self.coro is None:
            raise RuntimeError("Task has not been initialized.")

        return self.coro.send(self.send)
    
    def start(self):
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
