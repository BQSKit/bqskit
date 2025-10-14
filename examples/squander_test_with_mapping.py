from bqskit.ir import Circuit
from bqskit.ir.gates import U3Gate, CNOTGate, SwapGate
from bqskit.compiler import CompilationTask, Compiler
from bqskit.compiler.compile import build_seqpam_mapping_optimization_workflow
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.basepass import BasePass
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

circuit_name = 'ising_n10'
bqskit_circuit_original = Circuit.from_file(circuit_name + '.qasm')
print("current circuit is:",bqskit_circuit_original)

from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

def gen_swap_unitary():
    """
    Generate a unitary matrix that swaps the state of two qudits.

    Args:
        radix (int): The base of the qudits being swapped.
            Defaults to qubits or base 2. (Default: 2)

    Raises:
        ValueError: If radix is less than two.
    """
    dim = 2 * 2
    mat = [[0 for _j in range(dim)] for _i in range(dim)]
    for col in range(dim):
        # col = a * radix + b; a, b < radix
        a = col // 2
        b = col % 2
        row = b * 2 + a
        mat[row][col] = 1

    return UnitaryMatrix(mat, [2, 2])

def apply_permutation_sparse(state_vector, location):
    """
    Apply permutation to state vector in a sparse manner
    state_vector: numpy array of amplitudes
    permutation: list where permutation[i] = j means qubit i goes to position j
    Returns: new numpy array with permuted amplitudes
    """
    num_qudits = len(location)
    swap_utry = gen_swap_unitary()
    current_perm = list(location)
    for i in range(num_qudits):
        if i not in current_perm:
            current_perm.append(i)

    for index, qudit in enumerate(current_perm):
        if index != qudit:
            current_pos = current_perm.index(index)
            state_vector.apply(swap_utry, (index, current_pos))
            tmp = current_perm[index]
            current_perm[index] = current_perm[current_pos]
            current_perm[current_pos] = tmp
    return state_vector

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
                post_pam_seq,
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
                PAMLayoutPass(3), # originally 3
                PAMRoutingPass(0.1), # originally 0.1
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


#edges20 = [
#    (0, 1), (1, 2), (2, 3), (0, 5), (1, 6), (1, 7), (2, 6), (3, 8), (3, 9), (4, 9),
#    (5, 6), (5, 10), (5, 11), (6, 7), (6, 10), (6, 11), (7, 8), (7, 12), (7, 13),
#    (8, 9), (8, 12), (8, 13), (10, 11), (10, 15), (11, 12), (11, 16), (11, 17),
#    (12, 13), (13, 14), (13, 18), (13, 19), (14, 18), (14, 19), (15, 16), (16, 17),(4,8),(12,16)
#]
edges = [(i, i+1) for i in range(quditnumber-1)]
#edges.extend([(0,3), (0,2),(1,6),(2,12),(2,14)])
edges = [(int(u), int(v)) for (u, v) in edges]
#edges_heavy = [
#    (0, 1), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7),
#    (3, 4), (3, 8), (5, 6), (6, 7), (7, 8), (8, 9),
#    (5, 10), (6, 11), (7, 12), (8, 13),
#    (10, 11), (11, 12), (12, 13), (13, 14), (10, 15)
#]

#edges_heavy = [(int(u), int(v)) for (u, v) in edges_heavy]



print(quditnumber)



#coupling_graph = CouplingGraph(edges)
#print(coupling_graph)

#model = MachineModel(quditnumber, gate_set=Allowed_gate_set, coupling_graph=coupling_graph)
#model = MachineModel(quditnumber,gate_set = (coupling_graph = edges)
model = MachineModel(quditnumber, coupling_graph=edges)






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
print("TABU MAPPINGS")
print(pi_tree)
print(pf_tree)
inv_pf_tree = [0] * quditnumber
for i, j in enumerate(pf_tree):
    inv_pf_tree[j] = i

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
print("TABU MAPPINGS")
print(pi_tabu)
print(pf_tabu)
inv_pf_tabu = [0] * quditnumber
for i, j in enumerate(pf_tabu):
    inv_pf_tabu[j] = i

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

inv_pf_qsearch = [0] * quditnumber
for i, j in enumerate(pf_qsearch):
    inv_pf_qsearch[j] = i

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


# generate random initial state on which we test the circuits

num_qubits = qc_original.num_qudits 
matrix_size = 1 << num_qubits 
initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
initial_state = initial_state_real + initial_state_imag*1j
initial_state = initial_state/np.linalg.norm(initial_state)


# statevectors:


sv_original = qc_original.get_statevector(initial_state )
sv_tree = apply_permutation_sparse(qc_squander_tree.get_statevector(apply_permutation_sparse(StateVector(initial_state), pi_tree)), inv_pf_tree)
sv_tabu = apply_permutation_sparse(qc_squander_tabu.get_statevector(apply_permutation_sparse(StateVector(initial_state), pi_tabu)), inv_pf_tabu)
sv_qsearch = apply_permutation_sparse(qc_qsearch.get_statevector(apply_permutation_sparse(StateVector(initial_state), pi_qsearch)), inv_pf_qsearch)






def compute_overlap(state1, state2) -> float:
    # Extract numpy arrays safely
    if hasattr(state1, 'vec'):
        state1 = state1.vec
    if hasattr(state2, 'vec'):
        state2 = state2.vec
    
    # Normalize to avoid numerical drift
    state1 = state1 / np.linalg.norm(state1)
    state2 = state2 / np.linalg.norm(state2)

    return abs(np.vdot(state1, state2))



'''
print("\n \n squander tree:",np.abs(sv_tree[0:4]))

print("\n squander tree:",np.abs(sv_tree[4:8]))

print("\n squander tree:",np.abs(sv_tree[8:12]))

print("\n squander tree:",np.abs(sv_tree[12:16]))




print("\n \n original:",np.abs(sv_original[0:4]))

print("\n original:",np.abs(sv_original[4:8]))

print("\n original:",np.abs(sv_original[8:12]))

print("\n original:",np.abs(sv_original[12:16]))
'''
#tree =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#original = [0,2,1,3,12,15,14,13,11,9,10,8,4,7,6,5]
#original2 = [0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15]

#sv_shuffled = sv_original[original]
#sv_shuffled2 = sv_original[original2]
    
#print("\n \n shuffled:",np.abs(sv_shuffled))  
    
#print("\n \n squander tree:",np.abs(sv_tree))    

#print("\n \n shuffled2:",np.abs(sv_shuffled2))  

#print("\n \n original:",np.abs(sv_original))

overlap_squander_tree = compute_overlap(sv_original, sv_tree)
overlap_squander_tabu = compute_overlap(sv_original, sv_tabu)
overlap_qsearch = compute_overlap(sv_original, sv_qsearch)
overlap_tree_tabu = compute_overlap(sv_tree, sv_tabu)
overlap_tree_qsearch = compute_overlap(sv_tree, sv_qsearch)
overlap_tabu_qsearch = compute_overlap(sv_tabu, sv_qsearch)

# Display results
print()
print('The overlap of states (shuffled vs squander tree search): ', overlap_squander_tree)
print('The overlap of states (original vs squander tabu search): ', overlap_squander_tabu)
print('The overlap of states (original vs qsearch): ', overlap_qsearch)
print('The overlap of states (tree vs tabu): ', overlap_tree_tabu)
print('The overlap of states (tree vs qsearch): ', overlap_tree_qsearch)
print('The overlap of states (tabu vs qsearch): ', overlap_tabu_qsearch)
