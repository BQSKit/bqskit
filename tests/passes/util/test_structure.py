from __future__ import annotations

from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.structure import CircuitStructure
from bqskit.passes import QuickPartitioner
from bqskit.passes.util.structure import StructureAnalysisPass


def test_list_structures(compiler: Compiler) -> None:

    partitioner = QuickPartitioner(2)
    analyzer = StructureAnalysisPass()
    circuit = Circuit(3)

    true_structures = []
    num_blocks = 3
    for block_num in range(num_blocks):
        block = Circuit(2)
        for _ in range(block_num + 1):
            block.append_gate(CNOTGate(), (0, 1))
        structure = CircuitStructure(block)
        true_structures.append(structure)

    for edge in [(0, 1), (1, 2), (1, 2), (0, 2), (0, 2), (0, 2)]:
        circuit.append_gate(CNOTGate(), edge)
    circuit, data = compiler.compile(
        circuit, [partitioner, analyzer], request_data=True,
    )

    pass_structures = data['structures']
    assert len(pass_structures) == 3
    for ts, ps in zip(true_structures, pass_structures.keys()):
        assert ts == ps
        assert pass_structures[ps] == 1
