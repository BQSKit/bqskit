"""
=============================================================
Circuit Execution Module (:mod:`bqskit.exec`)
=============================================================

.. currentmodule:: bqskit.exec

BQSKit supports a variety of ways to execute circuits through
CircuitRunner.

.. rubric:: Core Classes

.. autosummary::
    :toctree: autogen
    :recursive:

    CircuitRunner
    RunnerResults

.. rubric:: CircuitRunners

.. autosummary::
    :toctree: autogen
    :recursive:

    QuestRunner
    IBMQRunner
    SimulationRunner
"""
from __future__ import annotations

__all__ = [
    'CircuitRunner',
    'RunnerResults',
    'QuestRunner',
    'SimulationRunner',
]

from bqskit.exec.results import RunnerResults
from bqskit.exec.runner import CircuitRunner
try:
    from bqskit.exec.runners.ibmq import IBMQRunner
    __all__ += ['IBMQRunner']
except ImportError:
    pass
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
