from __future__ import annotations

import logging

from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.compiler.task import TaskResult
from bqskit.compiler.task import TaskStatus
from bqskit.ir.circuit import Circuit

__all__ = [
    'CompilationTask',
    'Compiler',
    'TaskStatus',
    'TaskResult',
    'Circuit',
]

# Initialize Logging
_logger = logging.getLogger('bqskit')
_logger.setLevel(logging.CRITICAL)
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_fmt = '%(levelname)-8s | %(message)s'
_formatter = logging.Formatter(_fmt)
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
