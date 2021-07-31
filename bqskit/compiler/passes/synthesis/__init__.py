"""This package implements synthesis passes and synthesis related classes."""
from __future__ import annotations

from bqskit.compiler.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.compiler.passes.synthesis.qfast import QFASTDecompositionPass
from bqskit.compiler.passes.synthesis.qpredict import QPredictDecompositionPass
from bqskit.compiler.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.compiler.passes.synthesis.synthesis import SynthesisPass

__all__ = [
    'LEAPSynthesisPass',
    'QFASTDecompositionPass',
    'QPredictDecompositionPass',
    'QSearchSynthesisPass',
    'SynthesisPass',
]
