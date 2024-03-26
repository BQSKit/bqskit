"""This package implements BQSKit's post-processing passes."""
from __future__ import annotations

from bqskit.passes.processing.exhaustive import ExhaustiveGateRemovalPass
from bqskit.passes.processing.iterative import IterativeScanningGateRemovalPass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.substitute import SubstitutePass

__all__ = [
    'ExhaustiveGateRemovalPass',
    'IterativeScanningGateRemovalPass',
    'ScanningGateRemovalPass',
    'SubstitutePass',
]
