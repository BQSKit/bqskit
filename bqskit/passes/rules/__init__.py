"""This package implements BQSKit's rule-based passes."""
from __future__ import annotations

from bqskit.passes.rules.ch2cnot import CHToCNOTPass
from bqskit.passes.rules.cnot2ch import CNOTToCHPass
from bqskit.passes.rules.cnot2cy import CNOTToCYPass
from bqskit.passes.rules.cnot2cz import CNOTToCZPass
from bqskit.passes.rules.cy2cnot import CYToCNOTPass
from bqskit.passes.rules.swap2cnot import SwapToCNOTPass
from bqskit.passes.rules.u3 import U3Decomposition
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition

__all__ = [
    'CHToCNOTPass',
    'CNOTToCZPass',
    'CNOTToCHPass',
    'CNOTToCYPass',
    'CYToCNOTPass',
    'SwapToCNOTPass',
    'U3Decomposition',
    'ZXZXZDecomposition',
]
