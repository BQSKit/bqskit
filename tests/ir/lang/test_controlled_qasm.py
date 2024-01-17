from __future__ import annotations

from bqskit.ir.lang.qasm2 import OPENQASM2Language


class TestControlledQASM:
    def test_cu1(self) -> None:

        input_qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'cu1(3.1415) q[0], q[1];\n'
        )
        circuit = OPENQASM2Language().decode(input_qasm)

        output_qasm = circuit.to('qasm')

        assert input_qasm == output_qasm

    def test_cu3(self) -> None:

        input_qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'cu3(3.1415, 0.0, -4.0) q[0], q[1];\n'
        )
        circuit = OPENQASM2Language().decode(input_qasm)

        output_qasm = circuit.to('qasm')

        assert input_qasm == output_qasm
