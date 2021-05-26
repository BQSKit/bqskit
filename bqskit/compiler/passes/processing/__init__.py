"""This package implements BQSKit's post-processing passes."""
from __future__ import annotations

from bqskit.compiler.passes.processing.scan import ScanningGateRemovalPass
from bqskit.compiler.passes.processing.window import WindowOptimizationPass

__all__ = ['ScanningGateRemovalPass', 'WindowOptimizationPass']
