"""This module implements the VariableLocationGate."""
from __future__ import annotations

from typing import cast
from typing import Sequence

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import softmax


class VariableLocationGate(ComposedGate):
    """
    Gate that can multiplex multiple placements of another gate.

    A VariableLocationGate continuously interpolates between multiple locations
    for another gate. To do this this composed gate becomes as large as the sum
    of all locations.
    """

    def __init__(
        self,
        gate: Gate,
        locations: Sequence[CircuitLocationLike],
        radixes: Sequence[int] = [],
    ) -> None:
        """
        Create a gate that has parameterized location.

        Args:
            gate (Gate): The gate to parameterize location for.

            locations (Sequence[CircuitLocationLike]): A sequence of locations.
                Each location represents a valid placement for gate.

            radixes (Sequence[int]): The number of orthogonal
                states for each qudit. Attempts to infer it from `gate`
                and `locations` if not specified.

        Raises:
            ValueError: If there are not enough locations or the locations
                are incorrectly sized.

        Notes:
            The locations are calculated in their own space and are not
            relative to a circuit. This means you should consider the
            VariableLocationGate as its own circuit when deciding the
            locations. For example, if you want to multiplex the (2, 3)
            and (3, 5) placements of a CNOT on a 6-qubit circuit, then
            you would give the VariableLocationGate the (0, 1) and (1, 2)
            locations and place the VariableLocationGate on qubits
            (2, 3, 5) on the circuit.
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))

        if not all(CircuitLocation.is_location(l) for l in locations):
            raise TypeError('Expected a sequence of valid locations.')

        locations = [CircuitLocation(l) for l in locations]

        if not all(len(l) == gate.num_qudits for l in locations):
            raise ValueError('Invalid sized location.')

        if len(locations) < 1:
            raise ValueError('VLGs require at least 1 locations.')

        self.gate = gate
        name_str_tuple = (gate.name, locations, radixes)
        self._name = 'VariableLocationGate(%s, %s, %s)' % name_str_tuple
        self.locations = list(locations)
        self._num_qudits = len(set(sum((tuple(l) for l in locations), tuple())))

        if len(radixes) == 0:
            # Calculate radixes
            radix_map: dict[int, int | None] = {
                i: None for i in range(self.num_qudits)
            }
            for l in locations:
                for radix, qudit_index in zip(gate.radixes, l):
                    if radix_map[qudit_index] is None:
                        radix_map[qudit_index] = radix
                    elif radix_map[qudit_index] != radix:
                        raise ValueError(
                            'Gate cannot be applied to all locations'
                            ' due to radix mismatch.',
                        )

            if None in radix_map.values():
                raise ValueError('VariableUnitaryGate cannot infer radixes.')

            self._radixes = tuple(radix_map.values())  # type: ignore
        else:
            for l in locations:
                for radix, qudit_index in zip(gate.radixes, l):
                    if radixes[qudit_index] != radix:
                        raise ValueError(
                            'Gate cannot be applied to all locations'
                            ' due to radix mismatch.',
                        )

            self._radixes = tuple(radixes)
            self._num_qudits = len(self.radixes)

        self._num_params = self.gate.num_params + len(locations)

        self.extension_size = self.num_qudits - self.gate.num_qudits
        # TODO: This needs to changed for radixes
        self.I = np.identity(2 ** self.extension_size)
        self.perms = np.array([
            PermutationMatrix.from_qubit_location(self.num_qudits, l)
            for l in self.locations
        ])

    def get_location(self, params: RealVector) -> tuple[int, ...]:
        """Returns the gate's location."""
        idx = int(np.argmax(self.split_params(params)[1]))
        return tuple(self.locations[idx])

    def split_params(
        self,
        params: RealVector,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Split params into subgate params and location params."""
        return (
            np.array(params[:self.gate.num_params]),
            np.array(params[self.gate.num_params:]),
        )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        a, l = self.split_params(params)
        l = softmax(l, 10)

        P = np.sum([a * s for a, s in zip(l, self.perms)], 0)
        G = self.gate.get_unitary(a)
        PGPT = P.T @ np.kron(G, self.I) @ P
        return UnitaryMatrix.closest_to(PGPT, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        return self.get_unitary_and_grad(params)[1]

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)
        a, l = self.split_params(params)
        l = softmax(l, 10)

        P = np.sum([a * s for a, s in zip(l, self.perms)], 0)
        G = self.gate.get_unitary(a)
        G = np.kron(G, self.I)
        PG = P @ G
        GPT = G @ P.T
        PGPT = P @ GPT

        dG = cast(DifferentiableUnitary, self.gate).get_grad(a)
        dG = np.kron(dG, self.I)
        dG = P @ dG @ P.T

        perm_array = np.array([perm for perm in self.perms])
        dP = perm_array @ GPT + PG @ perm_array.transpose((0, 2, 1)) - 2 * PGPT
        dP = np.array([10 * x * y for x, y in zip(l, dP)])
        U = UnitaryMatrix.closest_to(PGPT, self.radixes)
        return U, np.concatenate([dG, dP])

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VariableLocationGate)
            and self.gate == other.gate
            and self.locations == other.locations
            and self.radixes == other.radixes
        )

    def __hash__(self) -> int:
        return hash((self.gate, tuple(self.locations), self.radixes))
