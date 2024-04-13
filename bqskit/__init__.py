"""The Berkeley Quantum Synthesis Toolkit Python Package."""
from __future__ import annotations

from typing import Any

from bqskit._logging import disable_logging
from bqskit._logging import enable_logging
from bqskit._version import __version__  # noqa: F401
from bqskit._version import __version_info__  # noqa: F401


def __getattr__(name: str) -> Any:
    # Lazy imports
    if name == 'compile':
        from bqskit.compiler.compile import compile
        return compile

    if name == 'Circuit':
        from bqskit.ir.circuit import Circuit
        return Circuit

    if name == 'MachineModel':
        from bqskit.compiler.machine import MachineModel
        return MachineModel

    raise AttributeError(f'module {__name__} has no attribute {name}')


__all__ = [
    'compile',
    'MachineModel',
    'Circuit',
    'enable_logging',
    'disable_logging',
]
