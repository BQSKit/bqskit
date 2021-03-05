"""This module implements the VariableLocationGate."""
from __future__ import annotations

from typing import Sequence

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_valid_location


class VariableLocationGate(Gate):
    """
    The VariableLocationGate class.

    A VariableLocationGate continuously encodes multiple locations for
    another gate.
    """

    def __init__(self, gate: Gate, locations: Sequence[Sequence[int]]) -> None:
        """
        Create a gate that has parameterized location.

        Args:
            gate (Gate): The gate to parameterize location for.

            locations (Sequence[Sequence[int]]): A sequence of locations.
                Each location represents a valid placement for gate.
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))

        if not all(is_valid_location(l) for l in locations):
            raise TypeError('Expected a sequence of locations.')

        if not all(len(l) == gate.get_size() for l in locations):
            raise TypeError('Invalid sized location.')

        min_qudit_idx = min(min(l) for l in locations)
        max_qudit_idx = max(max(l) for l in locations)

        self.gate = gate
        self.name = 'VariableLocationGate(%s)' % gate.get_name()
        self.size = max_qudit_idx - min_qudit_idx + 1
        self.num_params = self.gate.get_num_params() + len(locations)

        self.radixes = gate.get_radixes()
        # How do we handle size and radixes

        self.extension_size = self.size - self.gate.get_size()

    def split_params(
            self, params: Sequence[float],
    ) -> tuple[Sequence[float], Sequence[float]]:
        """Split params into subgate params and location params."""
        return (
            params[:self.gate.get_num_params()],
            params[self.gate.get_num_params() + 1:],
        )

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)
        a, l = self.split_params(params)
        return UnitaryMatrix.identity(2)  # TODO

    # def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
    #     """
    #     Returns the gradient for this gate, see Gate for more info.

    #     Notes:
    #         The derivative of the conjugate transpose of matrix is equal
    #         to the conjugate transpose of the derivative.
    #     """
    #     self.check_parameters(params)
    #     if hasattr(self, 'utry'):
    #         return np.array([])

    #     return np.transpose(self.gate.get_grad(params).conj(), (0, 2, 1))
