"""
================================================
Compiler Infrastructure (:mod:`bqskit.compiler`)
================================================

The `bqskit.compiler` package implements the BQSKit's compiler framework.
This includes classes for defining, combining, executing, and managing
compilation algorithms.

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:

    BasePass
    Compiler
    CompilationTask
    MachineModel
"""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.executor import Executor
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.task import CompilationTask

__all__ = [
    'compile',
    'BasePass',
    'Compiler',
    'Executor',
    'CompilationTask',
    'MachineModel',
]
