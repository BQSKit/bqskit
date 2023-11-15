"""This package implements utility passes."""
from __future__ import annotations

from bqskit.passes.util.compress import CompressPass
from bqskit.passes.util.conversion import BlockConversionPass
from bqskit.passes.util.converttou3 import ToU3Pass
from bqskit.passes.util.converttovar import ToVariablePass
from bqskit.passes.util.extend import ExtendBlockSizePass
from bqskit.passes.util.fill import FillSingleQuditGatesPass
from bqskit.passes.util.log import LogErrorPass
from bqskit.passes.util.log import LogPass
from bqskit.passes.util.random import SetRandomSeedPass
from bqskit.passes.util.record import RecordStatsPass
from bqskit.passes.util.structure import StructureAnalysisPass
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.passes.util.update import UpdateDataPass

__all__ = [
    'CompressPass',
    'RecordStatsPass',
    'SetRandomSeedPass',
    'UnfoldPass',
    'UpdateDataPass',
    'ToU3Pass',
    'BlockConversionPass',
    'LogPass',
    'ExtendBlockSizePass',
    'LogErrorPass',
    'FillSingleQuditGatesPass',
    'ToVariablePass',
    'StructureAnalysisPass',
]
