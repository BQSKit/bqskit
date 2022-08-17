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
_logger = logging.getLogger('bqskit')
_handler = logging.StreamHandler(_stdout)
_handler.setLevel(0)
_fmt_header = '(%(thread)d): %(asctime)s.%(msecs)03d - %(levelname)-8s |'
_fmt_message = ' %(name)s: %(message)s'
_fmt = _fmt_header + _fmt_message
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


__all__ = [
    'compile',
    'MachineModel',
    'Circuit',
    'enable_logging',
]

# Register supported languages
_register_language('qasm', _qasm())
