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
"""
from __future__ import annotations

from bqskit.exec.results import RunnerResults
from bqskit.exec.runner import CircuitRunner

__all__ = ['CircuitRunner', 'RunnerResults']
