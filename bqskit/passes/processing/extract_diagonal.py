"""This module implements the ExtractDiagonalPass."""
from __future__ import annotations

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir.operation import Operation
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate, DiagonalGate
from bqskit.ir.gates.constant import CNOTGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from typing import Any


theorized_bounds = [0, 0, 3, 14, 61, 252]
def construct_linear_ansatz(num_qudits: int):
    theorized_num = theorized_bounds[num_qudits]
    circuit = Circuit(num_qudits)
    circuit.append_gate(DiagonalGate(num_qudits), tuple(range(num_qudits)))
    for i in range(num_qudits):
        circuit.append_gate(VariableUnitaryGate(1), (i,))
    for _ in range(theorized_num // (num_qudits - 1)):
        # Apply n - 1 linear CNOTs
        for i in range(num_qudits - 1):
            circuit.append_gate(CNOTGate(), (i, i+1))
            circuit.append_gate(VariableUnitaryGate(1), (i,))
            circuit.append_gate(VariableUnitaryGate(1), (i+1,))
    return circuit


class ExtractDiagonalPass(BasePass):
    """
    A pass that attempts to extract a diagonal matrix from a unitary matrix.

    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1269020

    While there is a known algorithm for 2-qubit gates, we utilize
    synthesis methods instead to scale to wider qubit gates.

    As a heuristic, we attempt to extrac the diagonal using a linear chain
    ansatz of CNOT gates. We have found that up to 5 qubits, this ansatz
    does succeed for most unitaries with fewer CNOTs than the theoretical
    minimum number of CNOTs (utilizing the power of the Diagonal Gate in front).
    """

    def __init__(
            self,
            success_threshold: float = 1e-8,
            cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
            instantiate_options: dict[str, Any] = {},
        ) -> None:
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
            'min_iters': 0,
            'diff_tol_r':1e-4,
            'multistarts': 16,
            'method': 'qfactor'
        }
        self.instantiate_options.update(instantiate_options)
        super().__init__()


    async def decompose(
        self,
        op: Operation, 
        cost:CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        target: UnitaryMatrix = None,
        success_threshold: float = 1e-14,
        instantiate_options: dict[str, Any] = {}) -> tuple[Operation | None, 
                                                           Circuit]:
        '''
        Return the circuit that is generated from one levl of QSD. 
        '''
        
        circ = Circuit(op.gate.num_qudits)

        if op.gate.num_qudits == 2:
            # For now just try for 2 qubit
            circ.append_gate(DiagonalGate(op.gate.num_qudits), (0,1))
            circ.append_gate(VariableUnitaryGate(op.gate.num_qudits - 1), (0,))
            circ.append_gate(VariableUnitaryGate(op.gate.num_qudits - 1), (1,))
            circ.append_gate(CNOTGate(), (0, 1))
            circ.append_gate(VariableUnitaryGate(op.gate.num_qudits - 1), (0,))
            circ.append_gate(VariableUnitaryGate(op.gate.num_qudits - 1), (1,))
            circ.append_gate(CNOTGate(), (0, 1))
            circ.append_gate(VariableUnitaryGate(op.gate.num_qudits - 1), (0,))
            circ.append_gate(VariableUnitaryGate(op.gate.num_qudits - 1), (1,))
        elif op.gate.num_qudits == 3:
            circ = construct_linear_ansatz(op.gate.num_qudits)
        else:
            circ = construct_linear_ansatz(op.gate.num_qudits)

        instantiated_circ = circ.instantiate(target=target,  
                                             **instantiate_options)


        if cost.calc_cost(instantiated_circ, target) < success_threshold:
            print("Success")
            diag_op = instantiated_circ.pop((0,0))
            return diag_op, instantiated_circ
        
        default_circ = Circuit(op.gate.num_qudits)
        default_circ.append_gate(op.gate, 
                                 tuple(range(op.gate.num_qudits)), op.params)
        return None, default_circ
    

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        num_ops = 0
        print(circuit.gate_counts)

        j = circuit.count(VariableUnitaryGate(4))

        while j > 1:
            # Find last Unitary
            all_ops = list(circuit.operations_with_cycles(reverse=True))
            found = False
            for cyc, op in all_ops:
                if isinstance(op.gate, VariableUnitaryGate) and op.gate.num_qudits in [2,3,4]:
                    print("Replacing op at cyc", cyc)
                    if found:
                        merge_op = op
                        merge_pt = (cyc, op.location[0])
                        merge_location = op.location
                        break
                    else:
                        num_ops += 1
                        gate = op
                        pt = (cyc, op.location[0])
                        found = True
            # print(self.cost.calc_cost(circuit, data.target))
            # print(f"Decomposing {num_ops}th gate", flush=True)
            diag_op, circ = await self.decompose(gate, 
                                                 cost=self.cost, 
                                                 target=gate.get_unitary(), 
                                                 success_threshold=self.success_threshold, 
                                                 instantiate_options=self.instantiate_options)
            
            print(self.cost.calc_cost(circuit, data.target))
            circuit.replace_with_circuit(pt, circ, as_circuit_gate=True)
            print(self.cost.calc_cost(circuit, data.target))
            j -= 1
            # Commute Diagonal into next op
            if diag_op:
                print(self.cost.calc_cost(circuit, data.target))
                new_mat = diag_op.get_unitary() @ merge_op.get_unitary()
                circuit.replace_gate(merge_pt, merge_op.gate, merge_location, merge_op.gate.calc_params(new_mat))
                print(self.cost.calc_cost(circuit, data.target))
                print("Inserted Diagonal")

            print(circuit.gate_counts)
        circuit.unfold_all()