"""This package implements BQSKit's post-processing passes."""
from __future__ import annotations

from bqskit.passes.processing.exhaustive import ExhaustiveGateRemovalPass
from bqskit.passes.processing.iterative import IterativeScanningGateRemovalPass
from bqskit.passes.processing.rebase import Rebase2QuditGatePass  # TODO: Remove
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.treescan import TreeScanningGateRemovalPass
from bqskit.passes.processing.substitute import SubstitutePass

__all__ = [
    'ExhaustiveGateRemovalPass',
    'IterativeScanningGateRemovalPass',
    'ScanningGateRemovalPass',
    'TreeScanningGateRemovalPass',
    'SubstitutePass',
    'Rebase2QuditGatePass',
]
