"""This module implements the BasePass abstract base class."""
from __future__ import annotations

import abc
import logging
from typing import Any
from typing import Callable
from typing import TypeVar

from distributed import get_client
from distributed import rejoin
from distributed import secede

from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)

T = TypeVar('T')


class BasePass(abc.ABC):
    """
    The abstract base for BQSKit compiler passes.

    All BQSKit algorithms must inherit from BasePass to be run within
    the compiler framework. Each child class will need to implement its
    algorithm inside of the :func:`run` method.

    Examples:
        >>> class PrintCNOTCountPass(BasePass):
        ...     def run(self, circ: Circuit, data: dict[str, Any] = {}) -> None:
        ...         print(f"Number of CNOTs: {circ.count(CNOTGate())}")
    """

    @property
    def name(self) -> str:
        """The name of the pass."""
        return self.__class__.__name__

    @abc.abstractmethod
    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """
        Perform the pass's operation on `circuit`.

        Args:
            circuit (Circuit): The circuit to operate on.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to get auxillary information from previous
                passes and to store information for future passes.
                This function should never error based on what is in
                this dictionary.

        Note:
            - This function should be self-contained and have no side effects.
              This is because it will be potentially run in parallel.
        """

    @staticmethod
    def get_model(
        circuit: Circuit | UnitaryMatrix,
        data: dict[str, Any],
    ) -> MachineModel:
        """
        Retrieve the machine model from the data dictionary.

        Args:
            circuit (Circuit): The pass circuit.

            data (dict[str, Any]): The data dictionary.

        Returns:
            MachineModel: The machine model in the data dictionary, or
                a default one.
        """

        if len(data) == 0:
            return MachineModel(circuit.num_qudits)

        if 'machine_model' not in data:
            data['machine_model'] = MachineModel(circuit.num_qudits)

        if (
            not isinstance(data['machine_model'], MachineModel)
            or data['machine_model'].num_qudits < circuit.num_qudits
        ):
            _logger.warning('Expected machine_model to be a valid model.')
            return MachineModel(circuit.num_qudits)

        return data['machine_model']

    @staticmethod
    def get_placement(circuit: Circuit, data: dict[str, Any]) -> list[int]:
        """
        Retrieve the logical to physical qubit map from the data dictionary.

        Args:
            circuit (Circuit): The pass circuit.

            data (dict[str, Any]): The data dictionary.

        Returns:
            (list[int]): A list with length equal to circuit.num_qudits.
            Logical qubit i is mapped to the physical qubit described
            by the i-th element in the list.
        """
        # Default placement is trivial map: i -> i
        default_placement = list(range(circuit.num_qudits))
        model = BasePass.get_model(circuit, data)
        sg = model.coupling_graph.get_subgraph(default_placement)

        # Check in data for existing placement
        if 'placement' in data:
            p = data['placement']

            # Check it's valid
            if is_sequence(p) and len(p) == circuit.num_qudits:
                return data['placement']

        # If none found try to return default placement
        if sg.is_fully_connected():
            if len(data) != 0:
                data['placement'] = default_placement
            return default_placement

        raise RuntimeError(
            "No valid placement found and trivial one doesn't work."
            '\nConsider running a placement pass before the current pass.',
        )

    @staticmethod
    def get_target(circuit: Circuit, data: dict[str, Any]) -> UnitaryMatrix:
        """
        Retrieve the target unitary from the data dictionary.

        Args:
            circuit (Circuit): The pass circuit.

            data (dict[str, Any]): The data dictionary.

        Returns:
            UnitaryMatrix: The target unitary.
        """
        if len(data) == 0:
            return circuit.get_unitary()

        if 'target_unitary' not in data:
            data['target_unitary'] = circuit.get_unitary()

        if not isinstance(data['target_unitary'], UnitaryMatrix):
            _logger.warning('Expected target_unitary to be a unitary.')
            return circuit.get_unitary()

        return data['target_unitary']

    @staticmethod
    def in_parallel(data: dict[str, Any]) -> bool:
        """Return true if pass is being executed in a parallel environment."""
        if 'parallel' not in data:
            return False

        if not isinstance(data['parallel'], bool):
            _logger.warning('Expected parallel to be a bool.')
            return False

        return data['parallel']

    @staticmethod
    def execute(
        data: dict[str, Any],
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> list[T]:
        """Map a function over iterable arguments in parallel."""

        if not all(is_iterable(arg) for arg in args):
            raise ValueError('Each argument must be a sized iterable')

        if not all(len(arg) == len(args[0]) for arg in args):
            raise ValueError('Each iterable argument must have same length.')

        # Execute using dask if dask is started and there are multiple tasks
        if BasePass.in_parallel(data) and len(args[0]) > 1:
            if fn == Circuit.instantiate:
                kwargs['parallel'] = True

            client = get_client()
            futures = client.map(fn, *args, **kwargs)
            secede()
            results = client.gather(futures)
            rejoin()
            return results

        # Otherwise execute in current process
        else:
            results = []
            if len(args) == 1:
                for arg in args[0]:
                    results.append(fn(arg, **kwargs))
            else:
                for subargs in zip(*args):
                    results.append(fn(*subargs, **kwargs))
            return results
