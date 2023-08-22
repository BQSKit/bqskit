"""
The Unitary package.

This package exports the Unitary base class and several children
classes: UnitaryMatrix, UnitaryBuilder, DifferentiableUnitary, and
LocallyOptimizableUnitary.
"""
from __future__ import annotations

from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
# from bqskit.qis.unitary.unitary import Sequence[int]

__all__ = [
    'Unitary',
    'RealVector',
    # 'Sequence[int]',
    'UnitaryLike',
    'UnitaryMatrix',
    'UnitaryBuilder',
    'DifferentiableUnitary',
    'LocallyOptimizableUnitary',
]
