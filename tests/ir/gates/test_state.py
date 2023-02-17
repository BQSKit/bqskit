import pytest
from bqskit.ir.gates.state import StateGate
from bqskit.qis.state import StateVector


@pytest.mark.parametrize('radixes', [[2], [2, 2], [2, 3], [3, 2], [2, 2, 3], [3, 3, 3]])
def test_state_unitary(radixes: list[int]) -> None:
    num_qudits = len(radixes)
    state = StateVector.random(num_qudits, radixes)
    gate = StateGate(state)
    assert gate.state.get_distance_from(state) < 1e-7
