"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass

if TYPE_CHECKING:
    from typing import Any
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SynthesisPass(BasePass):
    """
    SynthesisPass abstract class.

    The SynthesisPass is a base class that exposes an abstract
    synthesize function. Inherit from this class and implement the
    synthesize function to create a synthesis tool.

    A SynthesisPass will synthesize a new circuit targeting the input
    circuit's unitary.
    """

    @abstractmethod
    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
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

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if len(data) == 0:
            data = dict()

        target_utry = self.get_target(circuit, data)
        circuit.become(self.synthesize(target_utry, data))
