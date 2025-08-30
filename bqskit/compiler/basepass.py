"""This module implements the BasePass abstract base class."""
from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING
import pickle

if TYPE_CHECKING:
    from typing import Any
    from bqskit.compiler.machine import MachineModel
    from bqskit.compiler.passdata import PassData
    from bqskit.compiler.workflow import Workflow
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.graph import CouplingGraph
    from bqskit.qis.state.system import StateSystem
    from bqskit.qis.state.state import StateVector
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class BasePass(abc.ABC):
    """
    The abstract base for BQSKit compiler passes.

    All BQSKit algorithms must inherit from BasePass to be run within
    the compiler framework. Each child class will need to implement its
    algorithm inside of the :func:`run` method.

    Examples:
        >>> class PrintCNOTCountPass(BasePass):
        ...     async def run(self, circuit: Circuit, data: PassData) -> None:
        ...         print(f"Number of CNOTs: {circuit.count(CNOTGate())}")
    """

    @property
    def name(self) -> str:
        """The name of the pass."""
        return self.__class__.__name__

    @abc.abstractmethod
    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Perform the pass's operation on `circuit`.

        Args:
            circuit (Circuit): The circuit to operate on.

            data (PassData): Associated data for the pass.
                Can be used to get auxillary information from previous
                passes and to store information for future passes.

        Note:
            - This function should be self-contained and have no side effects.
              This is because it will be potentially run in parallel.
        """

    def __str__(self) -> str:
        """Return a string representation of the pass."""
        return self.name

    @staticmethod
    def get_model(_: Any, data: PassData) -> MachineModel:
        """
        Retrieve the machine model from the data dictionary.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.model

    @staticmethod
    def get_placement(_: Any, data: PassData) -> list[int]:
        """
        Retrieve the logical to physical qubit map from the data dictionary.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.placement

    @staticmethod
    def get_connectivity(_: Any, data: PassData) -> CouplingGraph:
        """
        Retrieve the current connectivity of the circuit.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.connectivity

    @staticmethod
    def get_target(
        _: Any,
        data: PassData,
    ) -> UnitaryMatrix | StateVector | StateSystem:
        """
        Retrieve the target from the data dictionary.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to retrieve elements from the data dictionary'
            ' are now deprecated. We have upgraded the features of the data'
            ' dictionary. It can now be used directly to retrieve elements'
            " and provide defaults if they don't exist. This warning will"
            ' become an error in the future.',
            DeprecationWarning,
        )
        return data.target

    @staticmethod
    def in_parallel(data: dict[str, Any]) -> bool:
        """
        Return true if pass is being executed in a parallel.

        (Deprecated)
        """
        warnings.warn(
            'BasePass calls to `in_parallel` are deprecated and will always'
            ' return True. This warning will become an error in the future.',
            DeprecationWarning,
        )
        return True

    @staticmethod
    def execute(*args: Any, **kwargs: Any) -> Any:
        """
        Map a function over iterable arguments in parallel.

        (Defunct)
        """
        raise RuntimeError(
            'Since Dask was removed, the execute function became defunct.'
            ' We have switched to using `get_runtime().map(...)`. You can'
            ' import `get_runtime` from `bqskit.runtime`, and alongside a'
            ' an `await` keyword, should be a  drop-in replacement for the'
            ' execute function. See the following link for more info: '
            'https://bqskit.readthedocs.io/en/latest/source/runtime.html'
            'In a future version, this error will become an AttributeError.',
        )
    
    def checkpoint_finished(self, data: PassData, checkpoint_key: str) -> bool:
        """
        Check if we are checkpointing this pass. If so, check if the
        checkpoint has finished.

        Args:
            data (PassData): The data dictionary.
            checkpoint_key (str): The key to check for in the data dictionary.

        Returns:
            bool: True if the pass should checkpoint and has finished.
        """
        if "checkpoint_dir" in data:
            if data.get(checkpoint_key, False):
                return True
            
        return False

    def finish_checkpoint(self, 
                          circuit: Circuit, 
                          data: PassData, 
                          checkpoint_key: str,
                          remove_key: str | None = None) -> None:
        """
        Set the checkpoint key to True and save the data and circuit.

        Args:
            circuit (Circuit): The circuit to save.
            data (PassData): The data dictionary.
            checkpoint_key (str): The key to set to True.
            remove_key (str | None): If not None, remove this key from the data
                dictionary before saving.
        """
        if "checkpoint_dir" in data:
            data[checkpoint_key] = True
            if remove_key is not None:
                data.pop(remove_key)
            save_data_file = data["checkpoint_data_file"]
            save_circuit_file = data["checkpoint_circ_file"]
            pickle.dump(data, open(save_data_file, "wb"))
            pickle.dump(circuit, open(save_circuit_file, "wb"))

    def restart_checkpoint(self,
                           circuit: Circuit,
                           data: PassData, 
                           checkpoint_key: str) -> Any | None:
        """
        Load the saved data and circuit from the checkpoint.

        This will modify (in-place) the passed in circuit and data dictionary.

        Args:
            data (PassData): The data dictionary.
            checkpoint_key (str): The key to check for in the data dictionary.

        Returns:
            Any | None: If the checkpoint exists, it returns whatever data is 
            stored at the checkpoint key. Otherwise, it returns None.
        """
        if "checkpoint_dir" in data:
            load_data_file = data["checkpoint_data_file"]
            load_circuit_file = data["checkpoint_circ_file"]
            new_data = pickle.load(open(load_data_file, "rb"))
            new_circuit = pickle.load(open(load_circuit_file, "rb"))
            data.update(new_data)
            circuit.become(new_circuit)
            return data.get(checkpoint_key, None)

        return None

    def checkpoint_save(self, circuit: Circuit, data: PassData) -> None:
        """
        Save the circuit and data to the checkpoint.

        Args:
            circuit (Circuit): The circuit to save.
            data (PassData): The data dictionary.
        """
        if "checkpoint_dir" in data:
            save_data_file = data["checkpoint_data_file"]
            save_circuit_file = data["checkpoint_circ_file"]
            pickle.dump(data, open(save_data_file, "wb"))
            pickle.dump(circuit, open(save_circuit_file, "wb"))

async def _sub_do_work(
    workflow: Workflow,
    circuit: Circuit,
    data: PassData,
) -> tuple[Circuit, PassData]:
    """Execute a sequence of passes on circuit."""
    if 'calculate_error_bound' in data and data['calculate_error_bound']:
        old_utry = circuit.get_unitary()

    await workflow.run(circuit, data)

    if 'calculate_error_bound' in data and data['calculate_error_bound']:
        new_utry = circuit.get_unitary()
        data.error = new_utry.get_distance_from(old_utry)

    return circuit, data
