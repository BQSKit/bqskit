"""This package implements synthesis passes and synthesis related classes."""
from __future__ import annotations

from bqskit.passes.synthesis.bzxz import BlockZXZPass
from bqskit.passes.synthesis.bzxz import FullBlockZXZPass
from bqskit.passes.synthesis.diagonal import WalshDiagonalSynthesisPass
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.pas import PermutationAwareSynthesisPass
from bqskit.passes.synthesis.qfast import QFASTDecompositionPass
from bqskit.passes.synthesis.qpredict import QPredictDecompositionPass
from bqskit.passes.synthesis.qsd import FullQSDPass
from bqskit.passes.synthesis.qsd import MGDPass
from bqskit.passes.synthesis.qsd import QSDPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.passes.synthesis.target import SetTargetPass

__all__ = [
    'LEAPSynthesisPass',
    'QFASTDecompositionPass',
    'QPredictDecompositionPass',
    'QSearchSynthesisPass',
    'SynthesisPass',
    'SetTargetPass',
    'PermutationAwareSynthesisPass',
    'WalshDiagonalSynthesisPass',
    'FullQSDPass',
    'MGDPass',
    'QSDPass',
    'BlockZXZPass',
    'FullBlockZXZPass',
]
