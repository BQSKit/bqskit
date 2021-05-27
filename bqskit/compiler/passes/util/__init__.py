"""This package implements utility passes."""
from __future__ import annotations

from bqskit.compiler.passes.util.compress import CompressPass
from bqskit.compiler.passes.util.inject import InjectDataPass
from bqskit.compiler.passes.util.random import SetRandomSeedPass
from bqskit.compiler.passes.util.record import RecordStatsPass
from bqskit.compiler.passes.util.unfold import UnfoldPass

__all__ = [
    'CompressPass',
    'InjectDataPass',
    'SetRandomSeedPass',
    'RecordStatsPass',
    'UnfoldPass',
]
