"""
===========================================
Berkeley Quantum Synthesis Toolkit (BQSKit)
===========================================

BQSKit is a superoptimizing quantum compiler that aims to provide
easy to use and quick to extend software around quantum synthesis.
This is accomplished by first building a quantum circuit intermediate
representation designed to work efficiently with numerical optimizer
based synthesis algorithms, and second bundling a compiler infrastructure
and algorithm framework that can run many algorithms efficiently over
a cluster of computers.
"""

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
