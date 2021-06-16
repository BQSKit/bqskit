from bqskit import Circuit
from bqskit.ir.gates.constant.h import HGate

# Prepare circuit
circ = Circuit(3)
circ.append_gate(HGate(), [0])
circ.append_gate(HGate(), [1])
circ.append_gate(HGate(), [0])
circ.append_gate(HGate(), [2])
circ.append_gate(HGate(), [0])
circ.append_gate(HGate(), [1])
circ.append_gate(HGate(), [2])
fold_1 = [(0,0), (0,1)]
fold_2 = [(1,0), (1,2)]
fold_3 = [(2,0), (1,1), (1,2)]

# Expected boundaries:
# 0 : (-1,1)
# 1 : (-1,1)
# Observed boundaries:
# 0 : (-1,1)
# 1 : (-1,1)
circ.fold(fold_1)

# Expected boundaries:
# 0 : (0,2)
# 2 : (-1,1)
# Observed boundaries:
# 0 : (0,2)
# 2 : (0,3)
circ.fold(fold_2)

# Expected boundaries:
# 0 : (1,3)
# 1 : (0,2)
# 2 : (0,2)
# Observed boundaries:
# 0 : (0,3)
# 1 : (0,3)
# 2 : (0,3)
circ.fold(fold_3)