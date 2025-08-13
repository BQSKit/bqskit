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
from bqskit.qis.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
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


circuit_name = 'tutorials/heisenberg-16-20'  # or '9symml_195'
bqskit_circuit_original = Circuit.from_file(circuit_name + '.qasm')




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
         }
         

def generate_squander_seqpam(squander_config,block_size):

    squander = SquanderSynthesisPass(squander_config=squander_config)

    post_pam_seq = NOOPPass() # pam= permutation aware mapping
    
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
                PAMLayoutPass(3),
                PAMRoutingPass(0.1),
                post_pam_seq,
                ApplyPlacement(),
                UnfoldPass(),
            ],
        ),
        name='SeqPAM Mapping',
    )



quditnumber = bqskit_circuit_original.num_qudits
coupling_graph = CouplingGraph([
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15),
    (0, 8),
])
print(coupling_graph)
model = MachineModel(quditnumber,gate_set = Allowed_gate_set,coupling_graph=coupling_graph)




workflow = [
    SetModelPass(model),
    generate_squander_seqpam(config,largest_partition)
]
#Finally, we construct a compiler and submit the task
with Compiler(num_workers = 1) as compiler:
    with Compiler() as compiler:
        circuit_squander_tree = compiler.compile(bqskit_circuit_original, workflow)


Circuit.save(circuit_squander_tree, circuit_name + '_squander_tree_search.qasm')


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
         }




workflow = [
    SetModelPass(model),
    generate_squander_seqpam(config,largest_partition)
]


# Finally, we construct a compiler and submit the task
with Compiler(num_workers=1) as compiler:
    with Compiler() as compiler:
        circuit_squander_tabu = compiler.compile(bqskit_circuit_original, workflow)


Circuit.save(circuit_squander_tabu, circuit_name + '_squander_tabu_search.qasm')




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
    synthesized_circuit_qsearch = compiler.compile(bqskit_circuit_original, workflow)


# save the circuit is qasm format
Circuit.save(synthesized_circuit_qsearch, circuit_name + '_qsearch.qasm')



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
qc_squander_tabu = Circuit.from_file( circuit_name + '_squander_tabu_search.qasm' )
qc_squander_tree = Circuit.from_file( circuit_name + '_squander_tree_search.qasm' )
qc_qsearch  = Circuit.from_file( circuit_name + '_qsearch.qasm' )


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

sv_original = qc_original.get_statevector(initial_state)
sv_tabu = qc_squander_tabu.get_statevector(initial_state)
sv_tree = qc_squander_tree.get_statevector(initial_state)
sv_qsearch = qc_qsearch.get_statevector(initial_state)




# Compute overlaps (fidelity)
    
def compute_overlap(state1, state2) -> float:
    # Ensure both inputs are raw numpy arrays
    state1 = state1.vec if hasattr(state1, 'vec') else state1
    state2 = state2.vec if hasattr(state2, 'vec') else state2

    inner_product = np.conjugate(state1) @ state2
    return LA.norm(inner_product)


    
    
overlap_squander_tree = compute_overlap(sv_original, sv_tree)
overlap_squander_tabu = compute_overlap(sv_original, sv_tabu)
overlap_qsearch = compute_overlap(sv_original, sv_qsearch)

# Display results
print()
print('The overlap of states (original vs squander tree search): ', overlap_squander_tree)
print('The overlap of states (original vs squander tabu search): ', overlap_squander_tabu)
print('The overlap of states (original vs qsearch): ', overlap_qsearch)

        
print('SQUANDER circuit compatibility with model: ', model.is_compatible(circuit_squander))
print('QSEARCH circuit compatibility with model: ', model.is_compatible(synthesized_circuit_qsearch))
