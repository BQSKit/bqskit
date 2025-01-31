"""This module implements the OPENQASM2Language class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.ir.lang.language import LangException
from bqskit.ir.lang.language import Language
from bqskit.ir.lang.qasm2.parser import parse
from bqskit.ir.lang.qasm2.visitor import OPENQASMVisitor
from bqskit.ir.gates.measure import MeasurementPlaceholder

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


class OPENQASM2Language(Language):
    """The OPENQASM2Language class."""

    def encode(self, circuit: Circuit) -> str:
        """Write `circuit` in this language."""
        if not circuit.is_qubit_only():
            raise LangException('Only qubit circuits can be wrriten to qasm.')

        source = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
        source += f'qreg q[{circuit.num_qudits}];\n'
        is_classical_register_present = False
        for gate in circuit.gate_set:
            # add at maximum one classical register definition per gate
            if isinstance(gate, MeasurementPlaceholder):
                if is_classical_register_present:
                    # skip all subsequent classical register definitions
                    continue

            if isinstance(gate, MeasurementPlaceholder):
                is_classical_register_present = True

            print(gate)
            print("Gate definition:")
            print(gate.get_qasm_gate_def())
            source += gate.get_qasm_gate_def()

        for op in circuit:
            source += op.get_qasm()

        return source

    def decode(self, source: str) -> Circuit:
        """Parse `source` into a circuit."""
        tree = parse(source)
        visitor = OPENQASMVisitor()
        visitor.visit_topdown(tree)
        return visitor.get_circuit()
