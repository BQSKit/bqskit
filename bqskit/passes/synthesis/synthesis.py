"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SynthesisPass(BasePass):
    """
    SynthesisPass abstract class.

    The SynthesisPass is a base class that exposes an abstract synthesize
    function. Inherit from this class and implement the synthesize function to
    create a synthesis tool.

    A SynthesisPass will synthesize a new circuit targeting the input circuit's
    unitary.
    """

    @abstractmethod
    async def synthesize(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """
        Synthesis abstract method to synthesize a UnitaryMatrix into a Circuit.

        Args:
            utry (UnitaryMatrix): The unitary to synthesize.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to provide auxillary information from
                previous passes. This function should never error based
                on what is in this dictionary.

        Note:
            This function should be self-contained and have no side effects.
        """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        circuit.become(await self.synthesize(data.target, data))
