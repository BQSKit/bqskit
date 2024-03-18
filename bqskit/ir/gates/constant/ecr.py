"""This module implements the ECRGate."""
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
import math

class ECRGate(ConstantGate, QubitGate):
    _name = 'ecr'
    _num_qudits = 2
    _qasm_name = 'ecr'
    _utry = UnitaryMatrix([
            [0, 0, 1*1/math.sqrt(2), 1j*1/math.sqrt(2)],
            [0, 0, 1j*1/math.sqrt(2), 1*1/math.sqrt(2)],
            [1*1/math.sqrt(2), -1j*1/math.sqrt(2), 0, 0],
            [-1j*1/math.sqrt(2), 1*1/math.sqrt(2), 0, 0]
        ])

    def __init__(self) -> None:
        pass 

    def get_qasm_gate_def(self) -> str:
        qasm_rzx = "gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }"
        qasm_ecr = "gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }"
        return qasm_rzx +"\n" + qasm_ecr


