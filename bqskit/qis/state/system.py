"""This module implements the StateSystem class."""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

from bqskit.qis.state.state import StateVector

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing import TypeGuard


class StateSystem(Mapping[StateVector, StateVector]):
    """A system of input and output states."""

    def __init__(self, system: StateSystemLike) -> None:
        """Construct a state system."""
        if isinstance(system, StateSystem):
            self._system: dict[StateVector, StateVector] = system._system
            self._radixes: tuple[int, ...] = system._radixes
            self._dim: int = system._dim
            self._vec_count: int = system._vec_count
            self.target: npt.NDArray[np.complex128] = system.target
            return

        self._system = {
            StateVector(k): StateVector(v)
            for k, v in system.items()
        }
        self._radixes = list(self._system.keys())[0].radixes
        self._dim = list(self._system.keys())[0].dim
        self._vec_count = len(self._system)

        for k, v in system.items():
            if k.radixes != self.radixes:
                raise ValueError('States in system have mismatch in dimension.')
            if v.radixes != self.radixes:
                raise ValueError('States in system have mismatch in dimension.')

        # Check overlap matrices
        V = np.column_stack(list(self._system.keys()))
        W = np.column_stack(list(self._system.values()))
        Ov = V.conj().T @ V
        Wv = W.conj().T @ W
        if not np.allclose(Ov, Wv):
            raise ValueError(
                'State system is unsolvable:'
                ' input and output overlap matrices do not match.',
            )
        self.target = W @ V.conj().T

    @property
    def num_qudits(self) -> int:
        """The number of qudits in the state."""
        return len(self.radixes)

    @property
    def dim(self) -> int:
        """The vector dimension for this state."""
        return self._dim

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return self._radixes

    def __iter__(self) -> Iterator[StateVector]:
        return self._system.__iter__()

    def __len__(self) -> int:
        return self._system.__len__()

    def __getitem__(self, _key: StateVector) -> StateVector:
        return self._system.__getitem__(_key)

    def __contains__(self, _key: object) -> bool:
        return self._system.__contains__(_key)

    def is_qubit_only(self) -> bool:
        """Return true if this unitary can only act on qubits."""
        return all([radix == 2 for radix in self.radixes])

    def is_qutrit_only(self) -> bool:
        """Return true if this unitary can only act on qutrits."""
        return all([radix == 3 for radix in self.radixes])

    @staticmethod
    def is_state_system(V: Any) -> TypeGuard[StateSystemLike]:
        """
        Check if V is a system of pure states.

        Args:
            V (Any): The vector to check.

        Returns:
            bool: True if V is a system of pure states.
        """
        if isinstance(V, StateSystem):
            return True

        if not isinstance(V, dict):
            return False

        for k, v in V.items():
            if not StateVector.is_pure_state(k):
                return False

            if not StateVector.is_pure_state(v):
                return False

        return True


StateSystemLike = Union[StateSystem, Dict[Any, Any]]
