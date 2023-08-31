from __future__ import annotations

from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.passes import ExtendBlockSizePass
from bqskit.passes import QuickPartitioner
from bqskit.passes.util.structure import StructurePass


def test_list_structures(compiler: Compiler) -> None:

    partitioner = QuickPartitioner(2)
    analyzer = StructurePass()
    circuit = Circuit(3)
    for edge in [(0,1),(1,2),(1,2),(0,2),(0,2),(0,2)]:
        circuit.append_gate(CNOTGate(), edge)

    circ,data = compiler.compile(
        circuit,[partitioner, analyzer], request_data=True
    )

    structures = data['structures']
    assert len(structures) == 3
    assert structures[0].depth == 1
    assert structures[1].depth == 2
    assert structures[2].depth == 3

