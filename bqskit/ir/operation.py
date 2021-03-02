"""This module implements the Operation class."""

from __future__ import annotations

import numpy as np
from bqskit.utils.cachedclass import CachedClass
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary import Unitary
from bqskit.utils.typing import is_valid_location

from typing import Any, Sequence

from bqskit.ir.gate import Gate
from bqskit.ir.gates import FrozenParameterGate


class Operation(Unitary, CachedClass):
    """
    The Operation class.

    A Operation groups together a gate, its parameters and location.
    """

    def __init__(
        self, gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
        """
        Operation Constructor.
s
        Args:
            gate (Gate): The cell's gate.

            location (Sequence[int]):  The set of qudits this gate affects.

            params (Sequence[float]): The parameters for the gate.
        
        Raises:
            ValueError: If `gate`'s size doesn't match `location`'s length.

            ValueError: If `gate`'s size doesn't match `params`'s length.
        """

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate, got %s.' % type(gate))

        if not is_valid_location(location):
            raise TypeError('Invalid location.')

        if len(location) != gate.get_size():
            raise ValueError('Gate and location size mismatch.')

        self.num_params = gate.get_num_params()
        self.radixes = gate.get_radixes()
        self.size = gate.get_size()

        self.check_parameters(params)
        
        self._gate = gate
        self._location = location
        self._params = list(params)
    
    @property
    def gate(self) -> Gate:
        return self._gate
    
    @property
    def location(self) -> Sequence[int]:
        return self._location
    
    @property
    def params(self) -> Sequence[float]:
        return self._params

    def get_qasm(self) -> str:
        """Returns the qasm string for this operation."""

        if isinstance(self.gate, FrozenParameterGate):
            full_params = self.gate.get_full_params(self.params)
        else:
            full_params = self.params

        return '{}({}) q[{}];'.format(
            self.gate.get_qasm_name(),
            ', '.join([str(p) for p in full_params]),
            '], q['.join([str(q) for q in self.location]),
        )
    
    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Return the op's unitary, see Unitary for more info."""
        if params:
            self.check_parameters(params)
            return self.gate.get_unitary(params)
        return self.gate.get_unitary(self.params)

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Return the op's gradient, see Unitary for more info."""
        if params:
            self.check_parameters(params)
            return self.gate.get_grad(params)
        return self.gate.get_grad(self.params)

    def get_unitary_and_grad(self, params: Sequence[float] = []) -> tuple[UnitaryMatrix, np.ndarray]:
        """Return the op's unitary and gradient, see Unitary for more info."""
        if params:
            self.check_parameters(params)
            return self.gate.get_unitary_and_grad(params)
        return self.gate.get_unitary_and_grad(self.params)
    
    def __eq__(self, rhs: Any) -> bool:
        """Check for equality."""
        if self is rhs:
            return True
        
        if not isinstance(rhs, Operation):
            return NotImplemented
        
        return (
            self.gate == rhs.gate
            and all(x == y for x, y in zip(self.params, rhs.params))
            and all(x == y for x, y in zip(self.location, rhs.location))
        )

    def __str__(self) -> str:
        pass  # TODO

    def __repr__(self) -> str:
        pass  # TODO
