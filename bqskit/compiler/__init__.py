"""
================================================
Compiler Infrastructure (:mod:`bqskit.compiler`)
================================================

The `bqskit.compiler` package implements the BQSKit's compiler framework.
This includes classes for defining, combining, executing, and managing
compilation algorithms. It also includes the standard :func:`compile`
function definition.

.. rubric:: Standard BQSKit Compile Function

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:

    compile

.. rubric:: Compiler Infrastructure

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:

    BasePass
    Compiler
    MachineModel
    GateSet
    GateSetLike
    PassData
    CompilationStatus
    CompilationTask
    Workflow
    WorkflowLike
"""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.gateset import GateSet
from bqskit.compiler.gateset import GateSetLike
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.compiler.status import CompilationStatus
from bqskit.compiler.task import CompilationTask
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike

__all__ = [
    'BasePass',
    'compile',
    'Compiler',
    'GateSet',
    'GateSetLike',
    'MachineModel',
    'PassData',
    'CompilationStatus',
    'CompilationTask',
    'Workflow',
    'WorkflowLike',
]
