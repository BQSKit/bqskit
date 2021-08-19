"""This script is contains a simple use case of the QFAST synthesis method."""
from __future__ import annotations

import logging

from qiskit.quantum_info import OneQubitEulerDecomposer

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passes.partitioning.scan import ScanPartitioner
from bqskit.compiler.passes.synthesis import LEAPSynthesisPass
from bqskit.compiler.search.generators.simple import SimpleLayerGenerator
from bqskit.ir import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint

if __name__ == '__main__':
    # Enable logging
    logging.getLogger('bqskit').setLevel(logging.DEBUG)

    circuit = Circuit(5)
    circuit.append_gate(CNOTGate(), [0, 1])
    circuit.append_gate(HGate(), [1])
    circuit.append_gate(CNOTGate(), [0, 1])
    circuit.append_gate(HGate(), [0])
    circuit.append_gate(HGate(), [0])
    circuit.append_gate(HGate(), [1])
    circuit.append_gate(HGate(), [1])

    num_q = 5
    data = {
        'machine_model': MachineModel(
            num_q, [(i, i + 1)
                    for i in range(num_q - 1)],
        ),
    }
    partitioner = ScanPartitioner(3)

    # Combine all passes
    instantiate_options = {
        'min_iters': 0,
        'diff_tol_r': 1e-5,
        'dist_tol': 1e-11,
        'max_iters': 2500,
    }
    layer_generator = SimpleLayerGenerator(
        single_qudit_gate_1=VariableUnitaryGate(1),
    )

    passes = [
        partitioner,
        LEAPSynthesisPass(
            layer_generator=layer_generator,
            instantiate_options=instantiate_options,
        ),
        # WindowOptimizationPass(
        #    window_size=11,
        #    synthesispass=QSearchSynthesisPass(
        #        layer_generator=layer_generator,
        #        instantiate_options=instantiate_options,
        #    ),
        # ),
        # ScanningGateRemovalPass(),
    ]

    # We will now define the CompilationTask we want to run.
    task = CompilationTask(circuit, passes)

    from numpy.linalg import norm

    # Finally let's create create the compiler and execute the CompilationTask.
    with Compiler() as compiler:
        compiled_circuit = compiler.compile(task)
        for cycle, op in compiled_circuit.operations_with_cycles():
            if isinstance(op.gate, VariableUnitaryGate):
                utry = op.get_unitary()
                params = OneQubitEulerDecomposer('U3').angles(utry)

                new_op = Operation(U3Gate(), op.location, list(params))
                new_utry = new_op.get_unitary()
                new_params = OneQubitEulerDecomposer('U3').angles(new_utry)

                u1 = Operation(
                    U3Gate(), op.location, list(
                        params,
                    ),
                ).get_unitary()
                u2 = Operation(
                    U3Gate(), op.location, list(
                        new_params,
                    ),
                ).get_unitary()

                print(norm(u1 - u2))
                print(params)
                print(new_params)
                compiled_circuit.replace(
                    CircuitPoint(
                        cycle, op.location[0],
                    ), new_op,
                )
            else:
                print(op)
