"""This script is contains a simple use case of the QFAST synthesis method."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qiskit.quantum_info import OneQubitEulerDecomposer
from qsearch import assemblers
from qsearch import compiler
from qsearch import leap_compiler
from qsearch import options
from qsearch import post_processing
from threadpoolctl import threadpool_limits

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.ir import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.partitioning.scan import ScanPartitioner
from bqskit.passes.processing import ScanningGateRemovalPass
from bqskit.passes.processing import WindowOptimizationPass
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.synthesis import LEAPSynthesisPass
from bqskit.passes.synthesis import QSearchSynthesisPass
from bqskit.passes.synthesis import SynthesisPass
from bqskit.passes.util import UnfoldPass
from bqskit.qis.unitary import UnitaryMatrix


class OldLeap(SynthesisPass):
    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        utry = utry.numpy

        # Pass options into qsearch, being maximally quiet,
        # and set the target to utry
        opts = options.Options()
        opts.target = utry
        opts.verbosity = 0
        opts.write_to_stdout = False
        opts.reoptimize_size = 7

        # use the LEAP compiler, which scales better than normal qsearch
        # compiler = leap_compiler.LeapCompiler()
        c = compiler.SearchCompiler()
        output = c.compile(opts)

        # LEAP requires some post-processing
        # pp = post_processing.LEAPReoptimizing_PostProcessor()
        # output = pp.post_process_circuit( output, opts )
        output = assemblers.ASSEMBLER_IBMOPENQASM.assemble(output)
        return OPENQASM2Language().decode(output)


if __name__ == '__main__':
    # Enable logging
    logging.getLogger('bqskit').setLevel(logging.DEBUG)

    # circuit = Circuit.from_file("scratch/xy-10-20.qasm")
    # ScanPartitioner().run(circuit, {})
    # # ForEachBlockPass([LEAPSynthesisPass()]).run(circuit, {})
    # ForEachBlockPass([OldLeap()]).run(circuit, {})
    # UnfoldPass().run(circuit, {})
    # print(circuit.count(CNOTGate()))

    # with threadpool_limits(limits=1):
    #     task = CompilationTask(circuit, [
    #         ScanPartitioner(),
    #         ForEachBlockPass([LEAPSynthesisPass(), WindowOptimizationPass()]),
    #         UnfoldPass(),
    #     ])
    #     with Compiler() as compiler:
    #         compiled_circuit = compiler.compile(task)
    #     print(compiled_circuit.count(CNOTGate()))

    # circuit = Circuit.from_file("scratch/tfim-4-10.qasm")
    utry = np.loadtxt('scratch/benchmarks/toffoli.unitary', dtype=np.complex128)
    circuit = Circuit.from_unitary(utry)

    OldLeap().run(circuit, {})
    # QSearchSynthesisPass().run(circuit, {})
    # ScanningGateRemovalPass().run(circuit, {})
    print(circuit.count(CNOTGate()))
    assert circuit.get_unitary().get_distance_from(utry) < 1e-5

    # task = CompilationTask(circuit, [QSearchSynthesisPass()])
    # with Compiler() as compiler:
    #     compiled_circuit = compiler.compile(task)
    # print(compiled_circuit.count(CNOTGate()))

    # assert compiled_circuit.get_unitary().get_distance_from(utry) < 1e-5
