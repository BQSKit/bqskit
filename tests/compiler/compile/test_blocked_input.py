from __future__ import annotations

from bqskit import compile
from bqskit import MachineModel
from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import U3Gate


def test_compile_blocked_input_circuit_unfold_rebase_correctly(
        compiler: Compiler,
) -> None:
    circuit = Circuit(2)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    cg = CircuitGate(circuit, True)
    blocked_circuit = Circuit(2)
    blocked_circuit.append_gate(cg, (0, 1))
    model = MachineModel(2, gate_set={CZGate(), U3Gate()})
    out_circuit = compile(blocked_circuit, model, compiler=compiler)
    assert len(out_circuit.gate_set) == 2
    assert CZGate() in out_circuit.gate_set
    assert U3Gate() in out_circuit.gate_set
    assert CNOTGate() not in out_circuit.gate_set
