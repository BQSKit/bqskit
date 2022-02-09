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

from bqskit.exec.results import RunnerResults
from bqskit.exec.runner import CircuitRunner
from bqskit.exec.runners.ibmq import IBMQRunner
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner

__all__ = [
    'CircuitRunner',
    'RunnerResults',
    'QuestRunner',
    'IBMQRunner',
    'SimulationRunner',
]
