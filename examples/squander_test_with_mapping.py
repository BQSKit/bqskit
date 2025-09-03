from bqskit.ir import Circuit
from bqskit.ir.gates import U3Gate, CNOTGate, SwapGate
from bqskit.compiler import CompilationTask, Compiler
from bqskit.compiler.compile import build_seqpam_mapping_optimization_workflow
from bqskit.compiler.machine import MachineModel
from bqskit.passes import (
    QuickPartitioner, ClusteringPartitioner, GreedyPartitioner, ScanPartitioner,
    ForEachBlockPass, QSearchSynthesisPass, LEAPSynthesisPass, QFASTDecompositionPass,
    QPredictDecompositionPass, SquanderSynthesisPass, ScanningGateRemovalPass, UnfoldPass,
    RestoreModelConnectivityPass, NOOPPass, IfThenElsePass, NotPredicate, WidthPredicate,
    EmbedAllPermutationsPass, GeneralizedSabreLayoutPass, SetModelPass, PAMRoutingPass,
    PAMLayoutPass, ApplyPlacement, SubtopologySelectionPass, LogPass,
    ExtractModelConnectivityPass
)
from bqskit.passes.mapping.verify import PAMVerificationSequence
from bqskit.qis.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.state import StateVector, StateSystem
from bqskit.qis.graph import CouplingGraph
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.workflow import Workflow
from bqskit.passes.mapping.setmodel import SetModelPass
from qiskit import transpile
import numpy as np
from numpy import linalg as LA
import time
import pickle
import math
from scipy.sparse import csr_array


bqskit_circuit_original = Circuit.from_file(circuit_name + '.qasm')


def generate_sparse_perm_matrix(num_qudits,location):

    row,col = [],[]
    data = np.ones(2**num_qudits,dtype=np.complex128)
    for i in range(2**num_qudits):
        bitstring = format(i, f'0{num_qudits}b')  
        bits = list(map(int, bitstring))

        # Apply permutation (on qubit positions, not integer values)
        permuted_bits = [bits[p] for p in location]

        # Convert permuted bitstring back to integer
        j = int("".join(map(str, permuted_bits)), 2)

        row.append(j)
        col.append(i)
    mat = csr_array((data, (row, col)), shape=(2**num_qudits, 2**num_qudits))
    return mat

# define the largest partition in circuit
largest_partition = 3



print("\n the original gates are: \n")

    
original_gates = []

for gate in bqskit_circuit_original.gate_set:
    case_original = {f"{gate}count:": bqskit_circuit_original.count(gate)}
    original_gates.append(case_original)
    
print(original_gates, "\n")

Allowed_gate_set = bqskit_circuit_original.gate_set.union({SwapGate(),U3Gate()})
###########################################################################
# SQUANDER Tree seach synthesis 
 
start_squander = time.time()
  
config = {  'strategy': "Tree_search", 
            'parallel': 0,
            'verbosity' : 0,
            'optimization_tolerance': 1e-8
         }
         

def generate_squander_seqpam(squander_config,block_size):

    squander = SquanderSynthesisPass(squander_config = squander_config)

    post_pam_seq: BasePass = PAMVerificationSequence(8)
    
    return Workflow(
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Caching permutation-aware synthesis results.'),
                ExtractModelConnectivityPass(),
                QuickPartitioner(block_size),
                ForEachBlockPass(
                    EmbedAllPermutationsPass(
                        inner_synthesis = squander,
                        input_perm = True,
                        output_perm = False,
                        vary_topology = False,
                    ),              
                ),
                LogPass('Preoptimizing with permutation-aware mapping.'),
                PAMRoutingPass(),
                #post_pam_seq,
                UnfoldPass(),
                RestoreModelConnectivityPass(),
    
                LogPass('Recaching permutation-aware synthesis results.'),
                SubtopologySelectionPass(block_size),
                QuickPartitioner(block_size),
                ForEachBlockPass(
                    EmbedAllPermutationsPass(
                        inner_synthesis = squander,
                        input_perm = False,
                        output_perm = True,
                        vary_topology = True,
                    ),
                ),
                LogPass('Performing permutation-aware mapping.'),
                ApplyPlacement(),
                PAMLayoutPass(100), # originally 3
                PAMRoutingPass(2.0), # originally 0.1
                post_pam_seq,
                ApplyPlacement(),
                UnfoldPass(),
            ],
        ),
        name='SeqPAM Mapping',
    )



quditnumber = bqskit_circuit_original.num_qudits



def heavy_hex_coupling_for_qubits(num_qubits):
    """
    Generate heavy-hex connectivity edges for at least `num_qubits` qubits,
    choosing num_rows and num_cols to be as close to square as possible.
    
    Returns:
        edges: list of (u, v) tuples
        num_rows: chosen rows
        num_cols: chosen cols
    """
    def gen_edges(num_rows, num_cols):
        edges = []
        def qid(r, c):
            return r * num_cols + c

        for r in range(num_rows):
            for c in range(num_cols):
                q = qid(r, c)
                # Horizontal neighbor
                if c + 1 < num_cols:
                    edges.append((q, qid(r, c + 1)))
                # Zig-zag diagonal
                if r + 1 < num_rows:
                    if r % 2 == 0 and c + 1 < num_cols:
                        edges.append((q, qid(r + 1, c + 1)))
                    elif r % 2 == 1 and c > 0:
                        edges.append((q, qid(r + 1, c - 1)))
        return edges

    # Find best square-ish size
    best_rows, best_cols = None, None
    best_diff = float('inf')
    for rows in range(2, num_qubits + 3): 
        cols = math.ceil(num_qubits / rows)
        qubits_in_grid = rows * cols
        if qubits_in_grid >= num_qubits:
            diff = abs(rows - cols)
            if diff < best_diff:
                best_diff = diff
                best_rows, best_cols = rows, cols
    edges = gen_edges(best_rows, best_cols)
    return edges, best_rows, best_cols

#edges, rows, cols = heavy_hex_coupling_for_qubits(quditnumber) 

edges32 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(0,14),(4,15),(8,16),(12,17),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(14,18),(15,22),(16,26),(17,30)]
edges20 = [
    (0, 1), (1, 2), (2, 3), (0, 5), (1, 6), (1, 7), (2, 6), (3, 8), (3, 9), (4, 9),
    (5, 6), (5, 10), (5, 11), (6, 7), (6, 10), (6, 11), (7, 8), (7, 12), (7, 13),
    (8, 9), (8, 12), (8, 13), (10, 11), (10, 15), (11, 12), (11, 16), (11, 17),
    (12, 13), (13, 14), (13, 18), (13, 19), (14, 18), (14, 19), (15, 16), (16, 17),(4,8),(12,16)
]
edges32 = [(int(u), int(v)) for (u, v) in edges32]
edges20 = [(int(u), int(v)) for (u, v) in edges20]

#coupling_graph = CouplingGraph(edges)
#print(coupling_graph)

print(quditnumber)
model = MachineModel(quditnumber,gate_set = Allowed_gate_set,coupling_graph = edges20)




workflow = [
    SetModelPass(model),
    generate_squander_seqpam(config,largest_partition)
]
#Finally, we construct a compiler and submit the task
with Compiler(num_workers = 1) as compiler:
    with Compiler() as compiler:
        circuit_squander_tree, data_tree = compiler.compile(bqskit_circuit_original, workflow, request_data=True,)

Circuit.save(circuit_squander_tree, circuit_name + '_squander_tree_search_with_mapping.qasm')
pi_tree = data_tree['initial_mapping']
pf_tree = data_tree['final_mapping']

PI_tree = generate_sparse_perm_matrix(quditnumber, pi_tree)
PF_tree = generate_sparse_perm_matrix(quditnumber, pf_tree)


print("\n Circuit optimized with squander tree search:")



squander_gates = []

for gate in circuit_squander_tree.gate_set:
    case_squander = {f"{gate}count:":  circuit_squander_tree.count(gate)}
    squander_gates.append(case_squander)
 
end_squander = time.time()
time_squander = "the execution time with squander tree search:" + str(end_squander-start_squander)

print(squander_gates, "\n")
print( time_squander )
print(' ')
print(' ')
 

    
    

###########################################################################
# SQUANDER Tabu seach synthesis 
 
start_squander = time.time()

config = {  'strategy': "Tabu_search", 
            'parallel': 0,
            'optimization_tolerance': 1e-8
         }




workflow = [
    SetModelPass(model),
    generate_squander_seqpam(config,largest_partition)
]


# Finally, we construct a compiler and submit the task
with Compiler(num_workers=1) as compiler:
    with Compiler() as compiler:
        circuit_squander_tabu,data_tabu = compiler.compile(bqskit_circuit_original, workflow,request_data=True,)


Circuit.save(circuit_squander_tabu, circuit_name + '_squander_tabu_search_with_mapping.qasm')

pi_tabu = data_tabu['initial_mapping']
pf_tabu = data_tabu['final_mapping']

PI_tabu = generate_sparse_perm_matrix(quditnumber, pi_tabu)
PF_tabu = generate_sparse_perm_matrix(quditnumber, pf_tabu)


print("\n Circuit optimized with squander tabu search:")


squander_gates = []

for gate in circuit_squander_tabu.gate_set:
    case_squander = {f"{gate}count:":  circuit_squander_tabu.count(gate)}
    squander_gates.append(case_squander)
 
end_squander = time.time()
time_squander = "the execution time with squander with tabu search:" + str(end_squander-start_squander)

print(squander_gates, "\n")
print( time_squander )
print(' ')
print(' ')
  





###########################################################################
# QSearch synthesis

start_qsearch = time.time()





workflow = [
    SetModelPass(model),
    build_seqpam_mapping_optimization_workflow(block_size=largest_partition)
]


# Finally, we construct a compiler and submit the task
with Compiler() as compiler:
    synthesized_circuit_qsearch, data_qsearch = compiler.compile(bqskit_circuit_original, workflow, request_data=True)


# save the circuit is qasm format
Circuit.save(synthesized_circuit_qsearch, circuit_name + '_qsearch_with_mapping.qasm')

pi_qsearch = data_qsearch['initial_mapping']
pf_qsearch = data_qsearch['final_mapping']

PI_qsearch = generate_sparse_perm_matrix(quditnumber, pi_qsearch)
PF_qsearch = generate_sparse_perm_matrix(quditnumber, pf_qsearch)

print("\n Circuit optimized with qsearch:")

qsearch_gates = []

for gate in synthesized_circuit_qsearch.gate_set:
    case_qsearch = {f"{gate}count:":  synthesized_circuit_qsearch.count(gate)}   
    qsearch_gates.append(case_qsearch)
 
end_qsearch = time.time()
time_qsearch = "the execution time with qsearch:" + str(end_qsearch-start_qsearch)

print(qsearch_gates, "\n")
print( time_qsearch )
print(' ')
print(' ')




##############################################################################
#################### Test the generated circuits #############################

import qiskit
qiskit_version = qiskit.version.get_version_info()

from qiskit import QuantumCircuit
import qiskit_aer as Aer    
   

if qiskit_version[0] == '0':
    from qiskit import execute
else:
    from qiskit import transpile



# load the circuit from QASM format

qc_original = Circuit.from_file( circuit_name + '.qasm')
qc_squander_tabu = Circuit.from_file( circuit_name + '_squander_tabu_search_with_mapping.qasm' )
qc_squander_tree = Circuit.from_file( circuit_name + '_squander_tree_search_with_mapping.qasm' )
qc_qsearch  = Circuit.from_file( circuit_name + '_qsearch_with_mapping.qasm' )


#qc_original      = QuantumCircuit.from_qasm_file( circuit_name +  '.qasm' )
#qc_squander_tabu = QuantumCircuit.from_qasm_file( circuit_name +  '_squander_tabu_search.qasm' )
#qc_squander_tree = QuantumCircuit.from_qasm_file( circuit_name +  '_squander_tree_search.qasm' )
#qc_qsearch       = QuantumCircuit.from_qasm_file( circuit_name +  '_qsearch.qasm' )



# generate random initial state on which we test the circuits

num_qubits = qc_original.num_qudits 
matrix_size = 1 << num_qubits 
initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state = initial_state_real + initial_state_imag*1j
initial_state = initial_state/np.linalg.norm(initial_state)


# statevectors:

sv_original = qc_original.get_statevector(initial_state).numpy
sv_tabu =  qc_squander_tabu.get_statevector(PI_tabu @ initial_state ).numpy @ PF_tabu.T
sv_tree =  qc_squander_tree.get_statevector(PI_tree @ initial_state ).numpy @ PF_tree.T
sv_qsearch = qc_qsearch.get_statevector(PI_qsearch @ initial_state ).numpy @ PF_qsearch.T


# Compute overlaps (fidelity)
    
def compute_overlap(state1, state2) -> float:
    # Ensure both inputs are raw numpy arrays

    inner_product = np.conjugate(state1).T @ state2
    return LA.norm(inner_product)





    
    
overlap_squander_tree = compute_overlap(sv_original, sv_tree)
overlap_squander_tabu = compute_overlap(sv_original, sv_tabu)
overlap_qsearch = compute_overlap(sv_original, sv_qsearch)
overlap_tree_tabu = compute_overlap(sv_tree, sv_tabu)
overlap_tree_qsearch = compute_overlap(sv_tree, sv_qsearch)
overlap_tabu_qsearch = compute_overlap(sv_tabu, sv_qsearch)

# Display results
print()
print('The overlap of states (original vs squander tree search): ', overlap_squander_tree)
print('The overlap of states (original vs squander tabu search): ', overlap_squander_tabu)
print('The overlap of states (original vs qsearch): ', overlap_qsearch)
print('The overlap of states (tree vs tabu): ', overlap_tree_tabu)
print('The overlap of states (tree vs qsearch): ', overlap_tree_qsearch)
print('The overlap of states (tabu vs qsearch): ', overlap_tabu_qsearch)
