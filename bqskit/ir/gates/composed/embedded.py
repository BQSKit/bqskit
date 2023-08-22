"""This module implements the ControlledGate class."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_integer
#from bqskit.utils.typing import is_valid_radixes


class EmbeddedGate(ComposedGate, QuditGate, DifferentiableUnitary):
    """
    An arbitrary qudit gate composed from embedding a lower qudit gate into a
    higher qudit, eg qubit gate into qutrit gate.

    Given any qudit gate, EmbeddedGate can emded it into a higher qudit gate.
    """

    def __init__(
        self,
        gate: Gate,
        target_radixes: int | Sequence[int],
        level_map: Sequence[int] | Sequence[Sequence[int]],
    ):
        """

        Raises:
            TypeError: If gate is not of type Gate
            TypeError: If target_radixes is not an integer or sequence of integers
            TypeError: If level map is not a sequence of integers, or senquence of sequence of integers


            ValueError: If target_radixes is invalid (length of gate.radixes != length of target_radixes)
                        If gate.radixes[i]>target_radixes[i] for i in 1:len(target_radixes)
                        If length of level_map is not equal to target_radixes
                        If length of level_map[i] is not equal to target_radixes[i] for i in 1:len(level_map)
                        If np.any(level_map[i] >= target_radixes[i])

        Examples: (update)
            XGate for qutrits:
                ```
                > x_qutrit_gate = ControlledGate(XGate(),[3],[0,2])
                > x_qutrit_gate.get_unitary()
                > [[0, 0, sqrt(2)/2],
                   [0, 0, 0],
                   [sqrt(2)/2, 0, 0]]
                ```
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s.' % type(gate))

        if not is_integer(target_radixes) and type(target_radixes) != Sequence[int]:
            raise TypeError(
                'Expected target radixes to be integer or a sequence of integers, got %s.' % type(
                    target_radixes,
                ),
            )

        if type(level_map) != Sequence[int] and type(level_map) != Sequence[Sequence[int]]:
            raise TypeError(
                'Expected level map to a sequence of integers or a sequence of sequences of integers, got %s.' % type(
                    level_map,
                ),
            )

        if is_integer(target_radixes):
            _target_radixes = [target_radixes for i in range(len(gate.radixes))]
        else
            _target_radixes = target_radixes

        if len(_target_radixes)!=len(gate.radixes):
            raise ValueError(
                'Target radixes must have the same length as gate.radixes.',
            )

        for i in range(len(_target_radixes)):
            if _target_radixes[i] < gate.radixes[i]:
                raise ValueError(
                    'Target radix at index %s should be greater than or equal to gate radix at same index.' % (
                        i
                    ),
                )

        if len(gate.radixes) == 1:
            if len(level_map) != 1 and len(level_map) != gate.radixes[0]:
                raise ValueError(
                    f'Level map must be a sequnce of {gate.radixes[0]} integers, or a sequence with one sequence element of {gate.radixes[0]} integers.',
                )
        elif len(level_map) != len(gate.radixes):
            raise ValueError(
                'Level map must have the same length as gate.radixes.',
            )

        for i in range(len(level_map)):
            if len(level_map[i] != gate.radixes[i]):
                raise ValueError(
                    'Level map at index {} must have {} integers to match respective radix of gate'.format(
                        gate.radixes[i], gate.radixes[i],
                    ),
                )
            if np.any(level_map[i] >= gate.radixes[i]):
                raise ValueError(
                    'One or more elements of level map at index %s is greater than gate radix at the same index with value %s.'(
                        i, gate.radixes[i],
                    ),
                )

        self.gate = gate
        self._level_map = level_map
        self._num_qudits = gate._num_qudits
        self._name = 'Embedded(%s)' % self.gate.name
        self._num_params = self.gate._num_params
        self._radixes = tuple(target_radixes)

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            M = np.eye(np.prod(self.radixes), dtype=np.complex128)
            M = self._map_gate_to_target_radixes(
                M, self.gate.get_unitary().numpy(),
            )
            self.utry = UnitaryMatrix(M, self.radixes)

    def _map_gate_to_target_radixes(self, Matrix: npt.NDArray[np.complex128], Unitary: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Construct the matrix for the target radixes from the original gate
        input.

        gate.radixes[i] -> radixes[i] the j-th level of radixes[i] ->
        level_map[i][j]
        """
        if len(self.gate.radixes) == 1:
            if len(self._level_map) == 1:
                level_map = self._level_map[0]
            elif len(self._level_map) == self.gate.radixes:
                level_map = self._level_map
            for i in range(self.gate.radixes[0]):
                for j in range(self.gate.radixes[0]):
                    Matrix[level_map[i], level_map[j]] = Unitary[i, j]
        elif len(self.gate.radixes) == 2:
            for i1 in range(self.gate.radixes[0]):
                for j1 in range(self.gate.radixes[1]):
                    for i2 in range(self.gate.radixes[0]):
                        for j2 in range(self.gate.radixes[1]):
                            Matrix[
                                level_map[0][i1] * self.radixes[1] +
                                level_map[1][j1],
                                level_map[0][i2] * self.radixes[1] +
                                level_map[1][j2],
                            ] = Unitary[i1 * self.gate.radixes[1] + j1, i2 * self.gate.radixes[1] + j2]
        elif len(self.gate.radixes) == 3:
            for i1 in range(self.gate.radixes[0]):
                for j1 in range(self.gate.radixes[1]):
                    for k1 in range(self.gate.radixes[2]):
                        for i2 in range(self.gate.radixes[0]):
                            for j2 in range(self.gate.radixes[1]):
                                for k2 in range(self.gate.radixes[2]):
                                    Matrix[
                                        level_map[0][i1] * self.radixes[1] * self.radixes[2] +
                                        level_map[1][j1] *
                                        self.radixes[2] + level_map[2][k1],
                                        level_map[0][i2] * self.radixes[1] * self.radixes[2] +
                                        level_map[1][j2] *
                                        self.radixes[2] + level_map[2][k2],
                                    ] =\
                                        Unitary[
                                            i1 *
                                        self.gate.radixes[1] * self.gate.radixes[2] +
                                            j1 * self.gate.radixes[2] + k1,
                                            i2 *
                                        self.gate.radixes[1] * self.gate.radixes[2] +
                                            j2 * self.gate.radixes[2] + k2,
                                        ]
        else:
            raise NotImplementedError(
                'Currently we support up to 3 qudit gate embedding',
            )
        return Matrix

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, 'utry'):
            return self.utry

        M = np.eye(np.prod(self.radixes), dtype=np.complex128)
        M = self._map_gate_to_target_radixes(M, self.gate.get_unitary(params))
        return UnitaryMatrix(M, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry') or self._num_params == 0:
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        M = np.zeros(
            (
                np.prod(self.radixes), np.prod(
                    self.radixes,
                ),
            ), dtype=np.complex128,
        )
        result = []
        for i in range(len(grads)):
            result.append(self._map_gate_to_target_radixes(M, grads[i]))
        return np.array(result, dtype=np.complex128)

    def get_unitary_and_grad(self, params: RealVector = []) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return self.utry, np.array([])
        elif self._num_params == 0:
            return np.array([]), np.array([])

        U = self.get_unitary(params)
        G = self.get_grad(params)  # type: ignore

        return U, G

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, EmbeddedGate)
            and self.gate == other.gate
            and self._level_map == other._level_map
            and self.radixes == other.radixes
        )

    def __hash__(self) -> int:
        return hash((self.gate, self.radixes))
