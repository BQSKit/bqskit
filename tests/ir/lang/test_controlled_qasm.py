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

    def test_cu2(self) -> None:

        input_qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'cu2(3.1415, 0.0) q[0], q[1];\n'
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

    def test_cswap(self) -> None:

        input_qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[3];\n'
            'cswap q[0], q[1], q[2];\n'
        )
        circuit = OPENQASM2Language().decode(input_qasm)

        output_qasm = circuit.to('qasm')

        assert input_qasm == output_qasm

    def test_c3x(self) -> None:

        input_qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[4];\n'
            'c3x q[0], q[1], q[2], q[3];\n'
        )
        circuit = OPENQASM2Language().decode(input_qasm)

        output_qasm = circuit.to('qasm')

        assert input_qasm == output_qasm

    def test_c4x(self) -> None:

        input_qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[5];\n'
            'c4x q[0], q[1], q[2], q[3], q[4];\n'
        )
        circuit = OPENQASM2Language().decode(input_qasm)

        output_qasm = circuit.to('qasm')

        assert input_qasm == output_qasm

    def test_ch(self) -> None:

        input_qasm = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'ch q[0], q[1];\n'
        )
        try:
            OPENQASM2Language().decode(input_qasm)
        except ValueError:
            assert True
