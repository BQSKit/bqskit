"""This module implements the Instantiater base class."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateLike
    from bqskit.qis.state.system import StateSystemLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryLike


class Instantiater(abc.ABC):
    """
    The Instantiater class.

    An Instantiater is responsible for instantiating circuit templates such that
    the resulting circuit bests implements the desired target.
    """

    @abc.abstractmethod
    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        x0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Instantiate `circuit` to best implement `target`.

        Args:
            circuit (Circuit): The circuit template to instantiate.

            target (UnitaryMatrix | StateVector | StateSystem): The unitary
                matrix to implement or state to prepare.

            x0 (np.ndarray): Initial point to use during instantiation.

        Returns:
            (np.ndarray): The paremeters for the circuit that makes the
                circuit best implement `target`.

        Notes:
            This method should be side-effect free. This is necessary since
            many instantiate calls to the same circuit using the same
            Instantiater object may happen in parallel.
        """

    def multi_start_instantiate(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> Circuit:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        Args:
            circuit (Circuit): The circuit template to instantiate.

            target (UnitaryMatrix | StateVector | StateSystem): The unitary
                matrix to implement or state to prepare.

            num_starts (int): The number of starting points to attempt
                instantiation with.

        Returns:
            (Circuit): A circuit copy with the best parameters with
                respect to `target`.

        Notes:
            This method should be side-effect free. This is necessary since
            many instantiate calls to the same circuit using the same
            Instantiater object may happen in parallel.
        """
        _circuit = circuit.copy()
        self.multi_start_instantiate_inplace(_circuit, target, num_starts)
        return _circuit

    def multi_start_instantiate_inplace(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> None:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        See :func:`multi_start_instantiate` for more info.

        Notes:
            This method is a version of :func:`multi_start_instantiate`
            that modifies `circuit` in place rather than returning a copy.
        """
        target = self.check_target(target)
        start_gen = RandomStartGenerator()
        starts = start_gen.gen_starting_points(num_starts, circuit, target)
        cost_fn = HilbertSchmidtCostGenerator().gen_cost(circuit, target)
        params_list = [self.instantiate(circuit, target, x0) for x0 in starts]
        params = sorted(params_list, key=lambda x: cost_fn(x))[0]
        circuit.set_params(params)

    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> Circuit:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        See :func:`multi_start_instantiate` for more info.

        Notes:
            This method is an async version of :func:`multi_start_instantiate`
            and designed to parallelize the instantiation calls in the
            BQSKit Runtime during pass execution.
        """
        from bqskit.runtime import get_runtime
        target = self.check_target(target)
        start_gen = RandomStartGenerator()
        starts = start_gen.gen_starting_points(num_starts, circuit, target)
        cost_fn = HilbertSchmidtCostGenerator().gen_cost(circuit, target)
        params_list = await get_runtime().map(
            self.instantiate,
            [circuit] * num_starts,
            [target] * num_starts,
            starts,
        )
        params = sorted(params_list, key=lambda x: cost_fn(x))[0]
        circuit.set_params(params)
        return circuit

    @staticmethod
    @abc.abstractmethod
    def is_capable(circuit: Circuit) -> bool:
        """Return true if the circuit can be instantiated."""

    @staticmethod
    @abc.abstractmethod
    def get_violation_report(circuit: Circuit) -> str:
        """
        Return a message explaining why `circuit` cannot be instantiated.

        Args:
            circuit (Circuit): Generate a report for this circuit.

        Raises:
            ValueError: If `circuit` can be instantiated with this
                instantiater.
        """

    def check_target(
        self,
        target: UnitaryLike | StateLike | StateSystemLike,
    ) -> UnitaryMatrix | StateVector | StateSystem:
        """Check `target` to be valid and return it casted."""
        # Check `target`
        try:
            if UnitaryMatrix.is_unitary(target):
                target = UnitaryMatrix(target)

            elif StateVector.is_pure_state(target):
                target = StateVector(target)

            elif StateSystem.is_state_system(target):
                target = StateSystem(target)

            else:
                raise TypeError(
                    'Target is neither a unitary, a state system'
                    f', nor a state. Got {type(input)}.',
                )
        except Exception as e:
            raise TypeError(
                'Unable to determine type of input.'
                ' Ensure that you are trying to compile a valid'
                ' circuit, unitary, or state.',
            ) from e

        return target

    @staticmethod
    @abc.abstractmethod
    def get_method_name() -> str:
        """Return the name of this method."""
