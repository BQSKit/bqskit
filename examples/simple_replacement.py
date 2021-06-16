"""This module tests the CircuitIterator class."""
from __future__ import annotations

import logging

from numpy import pi

from bqskit.compiler.passes.partitioning.scan import ScanPartitioner
from bqskit.compiler.passes.synthesis import LEAPSynthesisPass
from bqskit.compiler.passes.util.variabletou3 import VariableToU3Pass
from bqskit.compiler.search.generators.simple import SimpleLayerGenerator
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate

# Enable logging
logging.getLogger('bqskit.compiler').setLevel(logging.DEBUG)

circuit = Circuit(3)
circuit.append_gate(U3Gate(), [0], [pi, pi / 2, pi / 2])
circuit.append_gate(U3Gate(), [1], [pi, pi / 2, pi / 2])
circuit.append_gate(CNOTGate(), [0, 1])
circuit.append_gate(U3Gate(), [2], [pi, pi / 2, pi / 2])
circuit.append_gate(U3Gate(), [0], [pi, pi / 2, pi / 2])
instantiate_options = {
    'min_iters': 0,
    'diff_tol_r': 1e-5,
    'dist_tol': 1e-11,
    'max_iters': 2500,
}
layer_generator = SimpleLayerGenerator(
    single_qudit_gate_1=VariableUnitaryGate(1),
)
synthesizer = LEAPSynthesisPass(
    layer_generator=layer_generator,
    instantiate_options=instantiate_options,
)
partitioner = ScanPartitioner()
converter = VariableToU3Pass()

pre_synth = circuit.get_unitary()

for cycle, op in circuit.operations_with_cycles():
    print('(', cycle, ',', op.location[0], ')', ': ', op)

partitioner.run(circuit, {})
synthesizer.run(circuit, {})

old_unis = []
for cycle, op in circuit.operations_with_cycles():
    print('(', cycle, ',', op.location[0], ')', ': ', op)
    old_unis.append(op.get_unitary())

converter.run(circuit, {})

new_unis = []
for cycle, op in circuit.operations_with_cycles():
    print('(', cycle, ',', op.location[0], ')', ': ', op)
    new_unis.append(op.get_unitary())

for i in range(len(old_unis)):
    print('\t', new_unis[i].get_distance_from(old_unis[i]))

post_synth = circuit.get_unitary()

print('Total error: ', post_synth.get_distance_from(pre_synth))
