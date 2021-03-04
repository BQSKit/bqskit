"""This module implements the QFASTPass class."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.passes.synthesis import SynthesisPass
from bqskit.ir.circuit import Circuit
from bqskit.qis.unitarymatrix import UnitaryMatrix


class QFASTPass(SynthesisPass):
    """
    The QFASTPass class.

    Performs one QFAST decomposition step breaking down a large unitary
    into a sequence of smaller operations.
    """

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        pass
        # circuit = Circuit(utry.get_size, utry.get_radixes)
        # while True:
        #     self.expand(circuit)
        #     circuit.optimize()
        #     if ObjectiveFunction.get_distance(
        #             circuit, utry,
        #     ) < self.success_threshold:
        #         break
        # self.finalize(circuit)
        # return circuit

    def expand(self, circuit: Circuit) -> None:
        """Expand the circuit."""
        # if isinstance(circuit[-1].gate, VariableLocationGate):
        #     loc = circuit[-1].gate.get_location(circuit[-1].params)
        #     # circuit.insert_gate(the right gate to insert)
        # else:
        #     circuit.append_gate(VariableLocationGate)

    def finalize(self, circuit: Circuit) -> None:
        # replace VariableLocationGate with FixedLocationGate
        # optimize
        pass
