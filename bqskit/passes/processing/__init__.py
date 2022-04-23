"""This package implements BQSKit's post-processing passes."""
from __future__ import annotations

from bqskit.passes.processing.exhaustive import ExhaustiveGateRemovalPass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.substitute import SubstitutePass
from bqskit.passes.processing.window import WindowOptimizationPass

__all__ = [
    'ExhaustiveGateRemovalPass',
    'ScanningGateRemovalPass',
    'SubstitutePass',
    'WindowOptimizationPass',
]
