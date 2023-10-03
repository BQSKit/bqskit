"""This module implements the Multiplex Gate Decomposition Pass for one level."""
from __future__ import annotations

import logging

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir.operation import Operation
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.mcry import MCRYGate
from bqskit.ir.gates.parameterized.mcrz import MCRZGate
from bqskit.ir.gates.parameterized import RYGate, RZGate
from bqskit.ir.gates.constant import CNOTGate
from bqskit.ir.gates import CircuitGate
from bqskit.runtime import get_runtime
import time

_logger = logging.getLogger(__name__)

class MGDPass(BasePass):
    """
    A pass performing one round of decomposition of the MCRY and MCRZ gates in a circuit.

    References:
        C.C. Paige, M. Wei,
        History and generality of the CS decomposition,
        Linear Algebra and its Applications,
        Volumes 208–209,
        1994,
        Pages 303-326,
        ISSN 0024-3795,
        https://arxiv.org/pdf/quant-ph/0406176.pdf
    """

    decompose_time = 0
    init_time = 0
    replace_time = 0

    @staticmethod
    def decompose(op: Operation) -> Circuit:
        '''
        Return the circuit that is generated from one levl of QSD. 
        '''
        start = time.time()
        gate: MCRYGate | MCRZGate = op.gate
        if (op.num_qudits > 2):
            gate_type = type(gate)
        elif (isinstance(gate, MCRYGate)):
            gate_type = RYGate
        else:
            gate_type = RZGate
    
        left_params, right_params =  MCRYGate.get_decomposition(op.params)
        circ = Circuit(gate.num_qudits)
        new_gate_location = list(range(1, gate.num_qudits))
        cx_location = (0, gate.num_qudits - 1)
        circ.append_gate(gate_type(gate.num_qudits - 1, gate.num_qudits - 2), new_gate_location, left_params)
        circ.append_gate(CNOTGate(), cx_location)
        circ.append_gate(gate_type(gate.num_qudits - 1, gate.num_qudits - 2), new_gate_location, right_params)
        circ.append_gate(CNOTGate(), cx_location)

        MGDPass.decompose_time += time.time() - start

        return circ

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # while num_ops > 0:
        start = time.time()
        gates = []
        pts = []
        locations = []
        num_ops = 0
        all_ops = list(circuit.operations_with_cycles(reverse=True))
        # Gather all of the unitaries
        for cyc, op in all_ops:
            if isinstance(op.gate, MCRYGate) or isinstance(op.gate, MCRZGate):
                num_ops += 1
                gates.append(op)
                pts.append((cyc, op.location[0]))
                locations.append(op.location)
        
        MGDPass.init_time += (time.time() - start)

        start = time.time()
        if len(gates) > 0:
            # Do a bulk QSDs -> circs
            circs = await get_runtime().map(MGDPass.decompose, gates)
            # Do bulk replace (single threaded)
            circ_gates = [CircuitGate(x) for x in circs]
            circ_ops = [Operation(x, locations[i], x._circuit.params) for i,x in enumerate(circ_gates)]
            circuit.batch_replace(pts, circ_ops)
            circuit.unfold_all()

        MGDPass.replace_time += (time.time() - start)

        # print(f"Init Time: {QSDPass.init_time}")
        # print(f"Decompose Time: {QSDPass.cs_time}")
        # print(f"Replace Time: {QSDPass.replace_time}")

        circuit.unfold_all()