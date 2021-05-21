"""This package implements synthesis passes and synthesis related classes."""
from __future__ import annotations

from bqskit.compiler.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.compiler.passes.synthesis.qfast import QFASTDecompositionPass
from bqskit.compiler.passes.synthesis.qsearch import QSearchSynthesisPass

__all__ = [
    'QFASTDecompositionPass',
    'QSearchSynthesisPass',
    'LEAPSynthesisPass',
]
