"""This package implements synthesis passes and synthesis related classes."""
from __future__ import annotations

from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.pas import PermutationAwareSynthesisPass
from bqskit.passes.synthesis.qfast import QFASTDecompositionPass
from bqskit.passes.synthesis.qpredict import QPredictDecompositionPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.passes.synthesis.target import SetTargetPass
from bqskit.passes.synthesis.qsd import QSDPass
from bqskit.passes.synthesis.full_qsd import FullQSDPass

__all__ = [
    'LEAPSynthesisPass',
    'QFASTDecompositionPass',
    'QPredictDecompositionPass',
    'QSearchSynthesisPass',
    'SynthesisPass',
    'SetTargetPass',
    'PermutationAwareSynthesisPass',
    'QSDPass',
    'FullQSDPass'
]
