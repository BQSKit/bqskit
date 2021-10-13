"""This example script showcases the SimplePartitioner."""
from __future__ import annotations

import numpy as np
from numpy import pi

from bqskit.compiler.machine import MachineModel
from bqskit.ir import Circuit
from bqskit.ir.gates.composed.controlled import ControlledGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.gates.parameterized.u1 import U1Gate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.iterator import CircuitIterator
from bqskit.passes.partitioning.scan import ScanPartitioner


def make_line(n):  # type: ignore
    edge_set = set()
    for i in range(0, n - 1):
        edge_set.add((i, i + 1))
    return edge_set


def input_state(circ, n):  # type: ignore
    for j in range(n):
        circ.append_gate(HGate(), [j])
        circ.append_gate(U1Gate(), [j], [-np.pi / float(2**(j))])


def qft(circ, n):  # type: ignore
    for j in range(n):
        circ.append_gate(HGate(), [j])
        for k in range(j + 1, n):
            circ.append_gate(
                ControlledGate(U1Gate()), [
                    k, j,
                ], [np.pi / float(2**(k - j))],
            )
    for j in range(int(np.floor(n / 2))):
        circ.append_gate(SwapGate(), [j, n - j - 1])


# QFT 5
num_q = 5
coup_map = make_line(num_q)
circ = Circuit(num_q)
# input_state(circ, num_q)
# qft(circ, num_q)

# Make QFT circuit
circ.append_gate(CNOTGate(), [1, 0])
circ.append_gate(U3Gate(), [0], [0, 0, 7 * pi / 4])
circ.append_gate(CNOTGate(), [1, 0])
circ.append_gate(U3Gate(), [0], [pi / 2, pi / 4, 5 * pi / 4])
circ.append_gate(U3Gate(), [1], [0, 0, pi / 8])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(U3Gate(), [2], [0, 0, 15 * pi / 8])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(U3Gate(), [1], [0, 0, pi / 16])
circ.append_gate(U3Gate(), [2], [0, 0, pi / 8])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [2, 1])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(U3Gate(), [1], [0, 0, 7 * pi / 4])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(U3Gate(), [0], [0, 0, pi / 8])
circ.append_gate(U3Gate(), [1], [pi / 2, pi / 4, 5 * pi / 4])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [1, 0])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(U3Gate(), [3], [0, 0, 6.0868358])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(U3Gate(), [2], [0, 0, pi / 32])
circ.append_gate(U3Gate(), [3], [0, 0, pi / 16])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [3, 2])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(U3Gate(), [2], [0, 0, 15 * pi / 8])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(U3Gate(), [1], [0, 0, pi / 16])
circ.append_gate(U3Gate(), [2], [0, 0, pi / 8])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [2, 1])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(U3Gate(), [1], [0, 0, 7 * pi / 4])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(U3Gate(), [0], [0, 0, pi / 8])
circ.append_gate(U3Gate(), [1], [pi / 2, pi / 4, 5 * pi / 4])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(U3Gate(), [4], [0, 0, 6.1850105])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(U3Gate(), [4], [0, 0, pi / 32])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(U3Gate(), [4], [0, 0, 6.0868358])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(U3Gate(), [4], [0, 0, pi / 16])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [4, 3])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [3, 2])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(U3Gate(), [2], [0, 0, 15 * pi / 8])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(U3Gate(), [2], [0, 0, pi / 8])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(U3Gate(), [2], [0, 0, 7 * pi / 4])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(U3Gate(), [2], [pi / 2, 0, 5 * pi / 4])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [3, 2])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [2, 1])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [2, 1])
circ.append_gate(CNOTGate(), [4, 3])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [4, 3])

# Do partitioning
mach = MachineModel(num_q, coup_map)
part = ScanPartitioner(3)
data = {}  # type: ignore

part.run(circ, data)

circ_iter = CircuitIterator(circ)
count = 0
for op in circ_iter:
    print('unitary %d' % (count))
    print(op.get_unitary())  # type: ignore
    count += 1
