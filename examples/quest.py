"""This script synthesizes and executes a circuit using the Quest algorithm."""
from __future__ import annotations

import logging

from bqskit.compiler import Compiler
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit.ir import Circuit
from bqskit.passes import ForEachBlockPass
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QFASTDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix

if __name__ == '__main__':
    # Enable logging
    logging.getLogger('bqskit').setLevel(logging.INFO)

    # Let's create a random 4-qubit unitary to synthesize and add it to a
    # circuit.
    circuit = Circuit.from_unitary(UnitaryMatrix.random(4))

    workflow = [
        QFASTDecompositionPass(),
        ForEachBlockPass([LEAPSynthesisPass(), ScanningGateRemovalPass()]),
        UnfoldPass(),
    ]

    # Finally let's create create the compiler and execute the CompilationTask.
    with Compiler() as compiler:
        compiled_circuit = compiler.compile(circuit, workflow)

        # Execute the circuit with Quest
        quest_runner = QuestRunner(SimulationRunner(), compiler=compiler)
        # Use IBMQRunner instead of SimulationRunner to run on a machine.
        results = quest_runner.run(compiled_circuit)
        print('Output Probability Distribution')
        print(results.probs)
