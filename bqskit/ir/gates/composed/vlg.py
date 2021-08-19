"""This module implements the VariableLocationGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import softmax


class VariableLocationGate(Gate):
    """
    The VariableLocationGate class.

    A VariableLocationGate continuously encodes multiple locations for another
    gate.
    """

    def __init__(
        self,
        gate: Gate,
        locations: Sequence[CircuitLocationLike],
        radixes: Sequence[int],
    ) -> None:
        """
        Create a gate that has parameterized location.

        Args:
            gate (Gate): The gate to parameterize location for.

            locations (Sequence[CircuitLocationLike]): A sequence of locations.
                Each location represents a valid placement for gate.

            radixes (Sequence[int]): The number of orthogonal
                states for each qudit. Defaults to qubits.

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

        if not all(len(l) == gate.get_size() for l in locations):
            raise ValueError('Invalid sized location.')

        if len(locations) < 1:
            raise ValueError('VLGs require at least 1 locations.')

        self.gate = gate
        self.name = 'VariableLocationGate(%s)' % gate.get_name()
        self.locations = list(locations)

        if radixes is None:
            # Calculate radixes
            radix_map: dict[int, int | None] = {
                i: None for i in range(self.size)
            }
            for l in locations:
                for radix, qudit_index in zip(gate.get_radixes(), l):
                    if radix_map[qudit_index] is None:
                        radix_map[qudit_index] = radix
                    elif radix_map[qudit_index] != radix:
                        raise ValueError(
                            'Gate cannot be applied to all locations'
                            ' due to radix mismatch.',
                        )

            self.radixes = tuple(radix_map.values())
        else:
            for l in locations:
                for radix, qudit_index in zip(gate.get_radixes(), l):
                    if radixes[qudit_index] != radix:
                        raise ValueError(
                            'Gate cannot be applied to all locations'
                            ' due to radix mismatch.',
                        )

            self.radixes = tuple(radixes)

        self.size = len(self.radixes)
        self.num_params = self.gate.get_num_params() + len(locations)

        self.extension_size = self.size - self.gate.get_size()
        # TODO: This needs to changed for radixes
        self.I = np.identity(2 ** self.extension_size)
        self.perms = np.array([
            PermutationMatrix.from_qubit_location(self.size, l)
            for l in self.locations
        ])

    def get_location(self, params: Sequence[float]) -> tuple[int, ...]:
        """Returns the gate's location."""
        idx = int(np.argmax(self.split_params(params)[1]))
        return tuple(self.locations[idx])

    def split_params(
            self, params: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split params into subgate params and location params."""
        return (
            np.array(params[:self.gate.get_num_params()]),
            np.array(params[self.gate.get_num_params():]),
        )

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)
        a, l = self.split_params(params)
        l = softmax(l, 10)

        P = np.sum([a * s for a, s in zip(l, self.perms)], 0)
        G = self.gate.get_unitary(a)  # type: ignore
        # TODO: Change get_unitary params to be union with np.ndarray
        PGPT = P @ np.kron(G, self.I) @ P.T
        return UnitaryMatrix.closest_to(PGPT, self.radixes)

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Unitary for more info."""
        return self.get_unitary_and_grad(params)[1]

    def get_unitary_and_grad(
        self,
        params: Sequence[float] = [],
    ) -> tuple[UnitaryMatrix, np.ndarray]:
        """Returns the unitary and gradient for this gate."""
        self.check_parameters(params)
        a, l = self.split_params(params)
        l = softmax(l, 10)

        P = np.sum([a * s for a, s in zip(l, self.perms)], 0)
        G = self.gate.get_unitary(a)  # type: ignore
        G = np.kron(G, self.I)
        PG = P @ G
        GPT = G @ P.T
        PGPT = P @ GPT

        dG = self.gate.get_grad(a)  # type: ignore
        dG = np.kron(dG, self.I)
        dG = P @ dG @ P.T

        perm_array = np.array([perm for perm in self.perms])
        dP = perm_array @ GPT + PG @ perm_array.transpose((0, 2, 1)) - 2 * PGPT
        dP = np.array([10 * x * y for x, y in zip(l, dP)])
        U = UnitaryMatrix.closest_to(PGPT, self.get_radixes())
        return U, np.concatenate([dG, dP])

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        raise NotImplementedError()
