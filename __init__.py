"""The Berkeley Quantum Synthesis Toolkit Python Package."""
from __future__ import annotations

import logging
from sys import stdout as _stdout

from bqskit.compiler.compile import compile
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.lang import register_language as _register_language
from bqskit.ir.lang.qasm2 import OPENQASM2Language as _qasm

# Initialize Logging
_logging_initialized = False


def enable_logging(verbose: bool = False) -> None:
    """
    Enable logging for BQSKit.

    Args:
        verbose (bool): If set to True, will print more verbose messages.
            Defaults to False.
    """
    global _logging_initialized
    if not _logging_initialized:
        _logger = logging.getLogger('bqskit')
        _handler = logging.StreamHandler(_stdout)
        _handler.setLevel(0)
        _fmt_header = '%(asctime)s.%(msecs)03d - %(levelname)-8s |'
        _fmt_message = ' %(name)s: %(message)s'
        _fmt = _fmt_header + _fmt_message
        _formatter = logging.Formatter(_fmt, '%H:%M:%S')
        _handler.setFormatter(_formatter)
        _logger.addHandler(_handler)
        _logging_initialized = True

    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger('bqskit').setLevel(level)


def enable_dashboard() -> None:
    import warnings
    warnings.warn(
        'Dask has been removed from BQSKit. As a result, the'
        ' enable_dashboard method has been removed.'
        'This warning will turn into an error in a future update.',
        DeprecationWarning,
    )


def disable_dashboard() -> None:
    import warnings
    warnings.warn(
        'Dask has been removed from BQSKit. As a result, the'
        ' disable_dashboard method has been removed.'
        'This warning will turn into an error in a future update.',
        DeprecationWarning,
    )


def disable_parallelism() -> None:
    import warnings
    warnings.warn(
        'The disable_parallelism method has been removed.'
        ' Instead, set the "num_workers" parameter to 1 during '
        'Compiler construction. This warning will turn into'
        'an error in a future update.',
        DeprecationWarning,
    )


def enable_parallelism() -> None:
    import warnings
    warnings.warn(
        'The enable_parallelism method has been removed.'
        ' Instead, set the "num_workers" parameter to 1 during '
        'Compiler construction. This warning will turn into'
        'an error in a future update.',
        DeprecationWarning,
    )


__all__ = [
    'compile',
    'MachineModel',
    'Circuit',
    'enable_logging',
]

# Register supported languages
_register_language('qasm', _qasm())
