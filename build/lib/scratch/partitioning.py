from bqskit import Circuit
from bqskit.compiler.passes.simplepartitioner import SimplePartitioner
from bqskit.compiler.passes.logicaltopology import LogicalTopology
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language

# Prepare circuit
with open('scratch/qft_qasm/qft_10.qasm', 'r') as f:
    circ = OPENQASM2Language().decode(f.read())

# Run partitioner on logical topology:w
conx = LogicalTopology()
mach = conx.get_logical_machine(circ)
part = SimplePartitioner(mach, 3)
data = {}
part.run(circ, data)
print('hi')
# Run Synthesis on partitioned circuit

# Do layout phase 1
#   Call mapping algorithm to get a qudit layout that minimizes SWAP globally
#   For edges that are not in the physical topology

# 