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


def get_runtime() -> RuntimeHandle:
    from bqskit.runtime.worker import get_worker
    return get_worker()


__all__ = ['get_runtime']
