from __future__ import annotations

# CALL
#   export PYTHONPATH="$PYTHONPATH:~/Documents/quantum/bqskit"

from bqskit.ir import Circuit
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passes.simplepartitioner import SimplePartitioner
from bqskit.ir.gates.constant.cx import CNOTGate
#     0  1  2  3           #########     
# 0 --o-----o--------    --#-o---o-#-----#######--
# 1 --x--o--x--o-----    --#-x-o-x-#######-o---#--
# 2 -----x--o--x--o-- => --#---x---#---o-#-x-o-#--
# 3 --o-----x-----x--    --#########-o-x-#---x-#--
# 4 --x--------------    ----------#-x---#######--
#                                  #######
num_q = 5
coup_map = set([(0,1),(1,2),(2,3),(3,4)])
circ = Circuit(num_q)
circ.append_gate(CNOTGate(), [0,1])
circ.append_gate(CNOTGate(), [3,4])
circ.append_gate(CNOTGate(), [1,2])
circ.append_gate(CNOTGate(), [0,1])
circ.append_gate(CNOTGate(), [2,3])
circ.append_gate(CNOTGate(), [1,2])
circ.append_gate(CNOTGate(), [2,3])
mach = MachineModel(num_q, coup_map)
part = SimplePartitioner(mach, 3)

data = {}

part.run(circ, data)
print("hello")