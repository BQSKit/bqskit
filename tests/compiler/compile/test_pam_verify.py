from __future__ import annotations

from bqskit.compiler.compile import build_seqpam_mapping_optimization_workflow
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.passes import SetModelPass
from bqskit.qis.permutation import PermutationMatrix

from bqskit.ir.gates import *
import logging
_logger = logging.getLogger(__name__)

def test_pam_verify(compiler: Compiler, medium_qasm_file: str) -> None:

    _logger.info(f' Reading from file: {medium_qasm_file}')

    circuit = Circuit.from_file(medium_qasm_file)
    out_circuit, data = compiler.compile(
        circuit,
        [
            SetModelPass(MachineModel(circuit.num_qudits)),
            build_seqpam_mapping_optimization_workflow(1, error_sim_size=8),
        ],
        request_data=True,
    )

    _logger.info(f' Finished compiling, CNOT count: {out_circuit.count(CNOTGate())}')

    upper_bound_error = data.error
    pi = data['initial_mapping']
    pf = data['final_mapping']
    out_utry = out_circuit.get_unitary()
    PI = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pi)
    PF = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pf)
    exact_error = out_utry.get_distance_from(PF.T @ circuit.get_unitary() @ PI)

    _logger.info(f' Upper bound error: {upper_bound_error}')
    _logger.info(f' Exact error: {exact_error}')
    _logger.info(f' |upper_bound-exact| : {abs(upper_bound_error - exact_error)}')
    
    assert upper_bound_error >= exact_error or abs(
        upper_bound_error - exact_error,
    ) < 5e-7
