"""This package implements the BQSKit Runtime."""
from __future__ import annotations

import faulthandler
import os
from typing import Any
from typing import Callable
from typing import Protocol
from typing import TYPE_CHECKING

# Enable low-level fault handling: system crashes print a minimal trace.
faulthandler.enable()


# Disable multi-threading in BLAS libraries.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['RUST_BACKTRACE'] = '1'


if TYPE_CHECKING:
    from bqskit.runtime.future import RuntimeFuture


class RuntimeHandle(Protocol):
    """
    A structural type capturing BQSKit Runtime capabilities.

    This protocol represents the functionality exposed to the BQSKit pass
    designer by the runtime system. Primarily, one can create tasks in the
    system via the submit and map methods. Additionally, it is possible to
    cancel tasks in the system, however, this is asynchronous and non-pre-
    emptive. The cancel will return immediately, the future it represents will
    become dead, and at some point in the future, all tasks and data associated
    with the cancelled one will be remove from the system. Lastly, one can wait
    on the first batch of result to arrive from a map task.
    """

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RuntimeFuture:
        ...

    def map(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RuntimeFuture:
        ...

    def cancel(self, future: RuntimeFuture) -> None:
        ...

    async def wait(self, future: RuntimeFuture) -> list[tuple[int, Any]]:
        ...


def get_runtime() -> RuntimeHandle:
    """Return a handle on the active runtime."""
    from bqskit.runtime.worker import get_worker
    return get_worker()


__all__ = ['get_runtime']
