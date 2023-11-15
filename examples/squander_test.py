from bqskit.ir import Circuit
circuit = Circuit.from_file('heisenberg-16-20.qasm')


from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.passes import Squander
# import

task = CompilationTask(circuit, [
    QuickPartitioner(3),
    ForEachBlockPass([Squander(), ScanningGateRemovalPass()]), # qsearch helyett sajt ami legyen más.
    UnfoldPass(),
])

for gate in circuit.gate_set:
    print(f"{gate} Count:", circuit.count(gate))
# Finally, we construct a compiler and submit the task
with Compiler() as compiler:
    synthesized_circuit = compiler.compile(task)


for gate in synthesized_circuit.gate_set:
    print(f"{gate} Count:", synthesized_circuit.count(gate))

