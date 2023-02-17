
import pytest

from bqskit import compile
from bqskit import MachineModel
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.machine import default_gate_set
from bqskit.ext.cirq.models import google_gate_set
from bqskit.ext.honeywell import honeywell_gate_set
from bqskit.ext.rigetti import rigetti_gate_set
from bqskit.qis.state import StateVector

def test_state_prep() -> None:
    state = StateVector.random(3)
    out_circuit = compile(state)
    out_state = out_circuit.get_statevector(StateVector.zero(3))
    assert out_state.get_distance_from(state) < 1e-7