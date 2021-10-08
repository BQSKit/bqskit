"""
=======================================================
Quantum Information Science Objects (:mod:`bqskit.qis`)
=======================================================

.. currentmodule:: bqskit.qis

The `bqskit.qis` package contains class definitions
for common objects found in quantum information science.

The most widely used will be the `UnitaryMatrix` object which
represents a pure quantum operation in matrix form.

.. jupyter-execute::
    :linenos:

    from bqskit.qis import UnitaryMatrix

    # Create a unitary object containing the cnot operation
    cnot = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )

    # Sample a random 2-qubit unitary from the Haar distribution
    utry = UnitaryMatrix.random(2)

    # The `UnitaryMatrix` class implement the NumPy API
    result = cnot @ utry
    print(result)

.. rubric:: Abstract Base Classes

.. autosummary::
    :toctree: autogen
    :recursive:

    Unitary
    DifferentiableUnitary
    LocallyOptimizableUnitary

.. rubric:: Core Classes

.. autosummary::
    :toctree: autogen
    :recursive:

    UnitaryMatrix
    StateVector
    StateVectorMap
    PauliMatrices
    PermutationMatrix
"""
from __future__ import annotations

from bqskit.qis.pauli import PauliMatrices
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.state import StateLike
from bqskit.qis.state import StateVector
from bqskit.qis.state import StateVectorMap
from bqskit.qis.unitary import DifferentiableUnitary
from bqskit.qis.unitary import LocallyOptimizableUnitary
from bqskit.qis.unitary import Unitary
from bqskit.qis.unitary import UnitaryLike
from bqskit.qis.unitary import UnitaryMatrix

__all__ = [
    'Unitary',
    'DifferentiableUnitary',
    'LocallyOptimizableUnitary',
    'UnitaryMatrix',
    'StateVector',
    'UnitaryLike',
    'StateVectorMap',
    'PauliMatrices',
    'StateLike',
    'PermutationMatrix',
]
