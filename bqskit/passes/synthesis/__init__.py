"""This package implements synthesis passes and synthesis related classes."""
from __future__ import annotations

from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.qfast import QFASTDecompositionPass
from bqskit.passes.synthesis.qpredict import QPredictDecompositionPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass

__all__ = [
    'LEAPSynthesisPass',
    'QFASTDecompositionPass',
    'QPredictDecompositionPass',
    'QSearchSynthesisPass',
    'SynthesisPass',
]
