"""This module implements the PassData class."""
from __future__ import annotations

import copy
import itertools as it
from typing import Any
from typing import Iterator
from typing import MutableMapping
from typing import Sequence

from bqskit.compiler.gateset import GateSet
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from bqskit.utils.typing import is_sequence


class PassData(MutableMapping[str, Any]):
    """
    A dictionary wrapper shared between all passes in a compilation run.

    This class reserves certain keywords and supplies default initial values for
    them. Other than that, it behaves very similar to a normal dictionary object
    mapping `str` keys to any type of value.
    """

    _reserved_keys = [
        'target',
        'model',
        'placement',
        'error',
        'seed',
        'machine_model',
        'initial_mapping',
        'final_mapping',
    ]

    def __init__(self, circuit: Circuit) -> None:
        """Initialize a PassData object from `circuit`."""
        self._target: Circuit | StateVector | UnitaryMatrix | StateSystem
        if circuit.num_qudits <= 8:
            try:
                self._target = circuit.get_unitary()
            except:
                self._target = circuit
        else:
            self._target = circuit  # Lazy evaluation

        self._error = 0.0
        self._model = MachineModel(circuit.num_qudits)
        self._placement = list(range(circuit.num_qudits))
        self._initial_mapping = list(range(circuit.num_qudits))
        self._final_mapping = list(range(circuit.num_qudits))
        self._data: dict[str, Any] = {}
        self._seed: int | None = None

    @property
    def target(self) -> StateVector | UnitaryMatrix | StateSystem:
        """Return the current target unitary or state."""
        if isinstance(self._target, Circuit):
            self._target = self._target.get_unitary()

        return self._target

    @target.setter
    def target(self, _val: StateVector | UnitaryMatrix | StateSystem) -> None:
        if not isinstance(_val, (StateVector, UnitaryMatrix, StateSystem)):
            raise TypeError(
                f'Cannot assign type {type(_val)} to target.'
                ' Expected either a StateVector, StateSystem,'
                ' or UnitaryMatrix.',
            )
        if len(self.placement) != _val.num_qudits:
            self.placement = list(range(_val.num_qudits))
        self._target = _val

    @property
    def error(self) -> float:
        """Return the current target unitary or state."""
        return self._error

    @error.setter
    def error(self, _val: float) -> None:
        if not is_real_number(_val):
            raise TypeError(
                f'Cannot assign type {type(_val)} to error.'
                ' Expected a real number.',
            )

        self._error = _val

    @property
    def model(self) -> MachineModel:
        """Return the current target MachineModel."""
        return self._model

    @model.setter
    def model(self, _val: MachineModel) -> None:
        if not isinstance(_val, MachineModel):
            raise TypeError(
                f'Cannot set model to {type(_val)}.'
                ' Expected a MachineModel.',
            )

        self._model = _val

    @property
    def gate_set(self) -> GateSet:
        """Return the current target MachineModel's GateSet."""
        return self._model.gate_set

    @gate_set.setter
    def gate_set(self, _val: GateSet) -> None:
        if not isinstance(_val, GateSet):
            raise TypeError(
                f'Cannot set gate_set to {type(_val)}.'
                ' Expected a GateSet.',
            )

        self._model.gate_set = _val

    @property
    def placement(self) -> list[int]:
        """Return the current placement of circuit qudits on model qudits."""
        return self._placement

    @placement.setter
    def placement(self, _val: Sequence[int]) -> None:
        if not is_sequence(_val):
            raise TypeError(
                f'Cannot set placement to {type(_val)}.'
                ' Expected a sequence of integers.',
            )

        if not all(is_integer(x) for x in _val):
            raise TypeError(
                'Cannot set placement. Expected a sequence of integers.',
            )

        self._placement = list(int(x) for x in _val)

    @property
    def initial_mapping(self) -> list[int]:
        """Return the initial mapping of logical to physical qudits."""
        return self._initial_mapping

    @initial_mapping.setter
    def initial_mapping(self, _val: Sequence[int]) -> None:
        if not is_sequence(_val):
            raise TypeError(
                f'Cannot set initial_mapping to {type(_val)}.'
                ' Expected a sequence of integers.',
            )

        if not all(is_integer(x) for x in _val):
            raise TypeError(
                'Cannot set initial_mapping. Expected a sequence of integers.',
            )

        self._initial_mapping = list(int(x) for x in _val)

    @property
    def final_mapping(self) -> list[int]:
        """Return the final mapping of logical to physical qudits."""
        return self._final_mapping

    @final_mapping.setter
    def final_mapping(self, _val: Sequence[int]) -> None:
        if not is_sequence(_val):
            raise TypeError(
                f'Cannot set final_mapping to {type(_val)}.'
                ' Expected a sequence of integers.',
            )

        if not all(is_integer(x) for x in _val):
            raise TypeError(
                'Cannot set final_mapping. Expected a sequence of integers.',
            )

        self._final_mapping = list(int(x) for x in _val)

    @property
    def seed(self) -> int | None:
        """Return the pass's seed."""
        return self._seed

    @seed.setter
    def seed(self, _val: int | None) -> None:
        if _val is not None and not is_integer(_val):
            raise TypeError(
                f'Cannot set seed to {type(_val)}.'
                ' Expected an integer or none.',
            )
        self._seed = _val

    @property
    def connectivity(self) -> CouplingGraph:
        """Retrieve the physical connectivity of the circuit qudits."""
        return self.model.coupling_graph.get_subgraph(self.placement)

    def __getitem__(self, _key: str) -> Any:
        """Retrieve the value associated with `_key` from the pass data."""
        if _key in self._reserved_keys:
            if _key == 'machine_model':
                _key = 'model'
            return self.__getattribute__(_key)

        return self._data.__getitem__(_key)

    def __setitem__(self, _key: str, _val: Any) -> None:
        """Update the value associated with `_key` in the pass data."""
        if _key in self._reserved_keys:
            if _key == 'machine_model':
                _key = 'model'
            return self.__setattr__(_key, _val)

        return self._data.__setitem__(_key, _val)

    def __delitem__(self, _key: str) -> None:
        """Delete the key-value pair associated with `_key`."""
        if _key in self._reserved_keys:
            raise RuntimeError(f'Cannot delete {_key} from data.')

        return self._data.__delitem__(_key)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over all keys in the pass data."""
        return it.chain(self._reserved_keys.__iter__(), self._data.__iter__())

    def __len__(self) -> int:
        """Return the number of key-value pairs in the pass data."""
        return self._data.__len__() + len(self._reserved_keys)

    def __contains__(self, _o: object) -> bool:
        """Return true if `_o` is a key in the pass data."""
        in_resv = self._reserved_keys.__contains__(_o)
        in_data = self._data.__contains__(_o)
        return in_resv or in_data

    def copy(self) -> PassData:
        """Returns a deep copy of the data."""
        return copy.deepcopy(self)

    def become(self, other: PassData, deepcopy: bool = False) -> None:
        """Become a copy of `other`."""
        if deepcopy:
            self._target = copy.deepcopy(other._target)
            self._error = copy.deepcopy(other._error)
            self._model = copy.deepcopy(other._model)
            self._placement = copy.deepcopy(other._placement)
            self._data = copy.deepcopy(other._data)
            self._seed = copy.deepcopy(other._seed)
        else:
            self._target = copy.copy(other._target)
            self._error = copy.copy(other._error)
            self._model = copy.copy(other._model)
            self._placement = copy.copy(other._placement)
            self._data = copy.copy(other._data)
            self._seed = copy.copy(other._seed)
