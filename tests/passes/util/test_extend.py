from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.passes import ExtendBlockSizePass
from bqskit.passes import QuickPartitioner


def test_extend_3(r6_qudit_circuit: Circuit) -> None:
    in_utry = r6_qudit_circuit.get_unitary()
    QuickPartitioner(3).run(r6_qudit_circuit)
    ExtendBlockSizePass(3).run(r6_qudit_circuit)
    out_utry = r6_qudit_circuit.get_unitary()
    assert all(isinstance(op.gate, CircuitGate) for op in r6_qudit_circuit)
    assert all(op.num_qudits == 3 for op in r6_qudit_circuit)
    assert in_utry == out_utry


def test_extend_4(r6_qudit_circuit: Circuit) -> None:
    in_utry = r6_qudit_circuit.get_unitary()
    QuickPartitioner(3).run(r6_qudit_circuit)
    ExtendBlockSizePass(4).run(r6_qudit_circuit)
    out_utry = r6_qudit_circuit.get_unitary()
    assert all(isinstance(op.gate, CircuitGate) for op in r6_qudit_circuit)
    assert all(op.num_qudits == 4 for op in r6_qudit_circuit)
    assert in_utry == out_utry
