"""The Berkeley Quantum Synthesis Toolkit Python Package."""
from __future__ import annotations

import logging

from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.lang import register_language
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


# Initialize Logging
_logger = logging.getLogger('bqskit')
_logger.setLevel(logging.CRITICAL)
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_fmt = '%(asctime)s.%(msecs)03d - %(levelname)-8s | %(name)s: %(message)s'
_formatter = logging.Formatter(_fmt, '%H:%M:%S')
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)


def enable_logging(verbose: bool = False) -> None:
    """
    Enable logging for BQSKit.

    Args:
        verbose (bool): If set to True, will print more verbose messages.
            Defaults to False.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger('bqskit').setLevel(level)


# Register supported languages
register_language('qasm', OPENQASM2Language())


__all__ = [
    'CompilationTask',
    'Compiler',
    'Circuit',
    'UnitaryMatrix',
    'enable_logging',
]
