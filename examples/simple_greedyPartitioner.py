from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.passes.greedypartitioner import GreedyPartitioner

num_q = 5
coup_map = {(0, 1), (1, 2), (2, 3), (3, 4)}
circ = Circuit(num_q)
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [3, 2])
circ.append_gate(CNOTGate(), [2, 1])
part = GreedyPartitioner()

print('Num Cycles Before:', circ.num_cycles)
print('Num Ops Before:', circ.num_operations)

data = {'multi_gate_score': 1, 'single_gate_score': 1}
part.run(circ, data)

# for point, op in circ.operations_with_points():
#     print(op, point)

print('Num Cycles After:', circ.num_cycles)
print('Num Ops After:', circ.num_operations)
